"""
오케스트레이터 — 단계 배선(파이프라인 본체).

데이터 흐름(스펙 2):
  영상
   ├ 영상 갈래: (성긴 검출 →) motion-guided 샘플링 → YOLO 검출 → VLM 행동
   └ 음성 갈래: STT → transcript 정제(근거발화 태깅)
                       ↓ (타임스탬프로 정렬)
            윈도우 정렬 → LLM 융합 → 암묵지 후보 JSON(스키마 검증)

각 단계는 registry로 만든 구현을 호출만 한다(구현 교체는 config로).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .config import PipelineConfig
from .interfaces.detector import FrameRef
from .registry import (
    build_aligner,
    build_detector,
    build_llm,
    build_refiner,
    build_sampler,
    build_stt,
    build_vlm,
)
from .schema.intermediate import Detection, FrameMeta
from .schema.tacit_schema import TacitKnowledgeDocument


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        # 구현 인스턴스화(아직 모델 로딩 안 함 — 각 어댑터가 지연 로딩)
        self.sampler = build_sampler(cfg.sampler)
        self.detector = build_detector(cfg.detector)
        self.vlm = build_vlm(cfg.vlm)
        self.stt = build_stt(cfg.stt)
        self.refiner = build_refiner(cfg.transcript_refine)
        self.aligner = build_aligner(cfg.aligner)
        self.llm = build_llm(cfg.llm)

    # ── 영상 메타 ─────────────────────────────────────────────────────
    def _probe_meta(self, video_path: str) -> FrameMeta:
        video_id = Path(video_path).name
        fps = self.cfg.fps_override
        width = height = n_frames = 0
        if fps is None:
            import cv2  # noqa: 지연 import

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return FrameMeta(
            video_id=video_id, path=video_path, fps=fps,
            width=width, height=height, n_frames=n_frames,
        )

    def run(self, video_path: str | None = None) -> TacitKnowledgeDocument:
        video_path = video_path or self.cfg.video_path
        if not video_path:
            raise ValueError("video_path 가 비었습니다. config.video_path 또는 --video 로 지정하세요.")
        meta = self._probe_meta(video_path)

        # ── 음성 갈래 ────────────────────────────────────────────────
        transcript = self.stt.transcribe(video_path, meta.video_id)
        transcript = self.refiner.refine(transcript)  # 정규화(정렬 전) + 근거 태깅

        # ── 영상 갈래 ────────────────────────────────────────────────
        # 1) (옵션) 성긴 검출로 ego-motion 무시 샘플링 근거 마련.
        #    TODO(decision): 성긴 검출 간격. 지금은 sampler가 coarse 없이도 transcript만으로 동작 가능.
        coarse = None  # 필요 시 self._coarse_detect(meta) 구현(uniform 샘플→detect)

        # 2) motion-guided 샘플링 (YOLO 위치힌트 + 부품주입 소스용. 본선 VLM은 자체 fps 샘플)
        frames: List[FrameRef] = self.sampler.sample(
            meta, coarse_detections=coarse, transcript=transcript
        )
        self._fill_images(meta, frames)

        # 3) YOLO 검출(위치 힌트 + 부품 자동 주입 소스)
        detections_by_frame = self.detector.detect(frames, meta)
        flat_dets: List[Detection] = [d for fd in detections_by_frame for d in fd.detections]

        # 4) VLM 관찰 추출 — 본선=네이티브 비디오, 옵션=프레임 리스트
        input_mode = getattr(self.vlm, "input_mode", "native_video")
        if input_mode == "native_video":
            injected = self._injected_parts(meta.video_id, flat_dets)
            actions = self.vlm.observe_video(video_path, meta, injected_parts=injected)
        else:
            actions = self.vlm.describe_actions(frames, detections_by_frame)

        # 5) 순차 실행(단일 GPU): VLM 언로드 후 LLM 로드 (VLM→언로드→LLM)
        if hasattr(self.vlm, "unload"):
            self.vlm.unload()

        # ── 융합 ──────────────────────────────────────────────────────
        windows = self.aligner.align(actions, transcript, flat_dets)
        doc = self.llm.fuse(windows, meta)

        # 할루시네이션 규율 교차검증(경고만 — 후보 누락 방지)
        for c in doc.candidates:
            for w in c.cross_check():
                print(f"[CROSS-CHECK][{c.id}] {w}")

        self._save(doc)
        return doc

    def _injected_parts(self, video_id: str, flat_dets: List[Detection]):
        """부품 주입 대상 결정 — VLM 어댑터의 정책(videos_map/토글)을 그대로 위임."""
        detected = [d.cls for d in flat_dets]
        if hasattr(self.vlm, "_injected_parts_for"):
            return self.vlm._injected_parts_for(video_id, detected)
        return sorted(set(detected)) or None

    def _fill_images(self, meta: FrameMeta, frames: List[FrameRef]) -> None:
        """샘플 프레임의 image를 한 번의 순차 디코드로 채운다(랜덤 시킹보다 효율적)."""
        if not frames:
            return
        import cv2  # noqa

        want = {f.frame_idx for f in frames}
        by_idx = {f.frame_idx: f for f in frames}
        cap = cv2.VideoCapture(meta.path)
        idx = 0
        ok, img = cap.read()
        while ok:
            if idx in want:
                by_idx[idx].image = img
            idx += 1
            ok, img = cap.read()
        cap.release()

    def _save(self, doc: TacitKnowledgeDocument) -> str:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{doc.video_id}.tacit.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 암묵지 후보 {len(doc.candidates)}건 저장 → {path}")
        return str(path)
