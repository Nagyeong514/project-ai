"""
오케스트레이터 — 단계 배선(파이프라인 본체).

데이터 흐름(본선 = native_video):
  영상
   ├ 음성 갈래: STT → transcript 정제(정규화+반복감지)
   └ 영상 갈래: ffmpeg 프레임 추출(YOLO·VLM 공용) → YOLO 검출 → VLM 관찰
                       ↓ (타임스탬프로 정렬)
            윈도우 정렬 → LLM 융합(VLM unload 후 로드) → 암묵지 후보 JSON

이 env선 torchcodec/pyav/cv2 디코딩이 불안정 → 프레임은 ffmpeg CLI로 한 번만 추출해 공용.
단일 GPU 순차: VLM 추론 → unload → LLM 로드.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .components.frame_extract import extract_frames, probe_duration
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
        self.sampler = build_sampler(cfg.sampler)  # (옵션 모드용; 본선은 미사용)
        self.detector = build_detector(cfg.detector)
        self.vlm = build_vlm(cfg.vlm)
        self.stt = build_stt(cfg.stt)
        self.refiner = build_refiner(cfg.transcript_refine)
        self.aligner = build_aligner(cfg.aligner)
        self.llm = build_llm(cfg.llm)

    def _probe_meta(self, video_path: str) -> FrameMeta:
        video_id = Path(video_path).name
        # TODO(decision): 실제 영상 fps로 교체 — observe_video/motion 경로 쓸 때 위험
        fps = self.cfg.fps_override or 30.0
        return FrameMeta(video_id=video_id, path=video_path, fps=fps,
                         width=0, height=0, n_frames=0)

    def run(self, video_path: str | None = None) -> TacitKnowledgeDocument:
        video_path = video_path or self.cfg.video_path
        if not video_path:
            raise ValueError("video_path 가 비었습니다. config.video_path 또는 --video 로 지정하세요.")
        meta = self._probe_meta(video_path)

        # ── 음성 갈래 ────────────────────────────────────────────────
        print("[1/5] STT...")
        transcript = self.stt.transcribe(video_path, meta.video_id)
        transcript = self.refiner.refine(transcript)  # 정규화 + 반복감지

        # ── 영상 갈래: ffmpeg 프레임 추출(YOLO·VLM 공용) ─────────────
        print("[2/5] 프레임 추출(ffmpeg)...")
        ffmpeg_bin = getattr(self.vlm, "ffmpeg_bin", None)
        long_side = getattr(self.vlm, "long_side", 480)
        frames_dir = getattr(self.vlm, "frames_dir", f"{self.cfg.output_dir}/_frames")
        dur = probe_duration(video_path, ffmpeg_bin)
        fps = self.vlm.fps_for_duration(dur) if hasattr(self.vlm, "fps_for_duration") else 0.5
        frame_paths, times = extract_frames(video_path, fps, frames_dir,
                                            long_side=long_side, ffmpeg_bin=ffmpeg_bin)
        print(f"      duration={dur:.0f}s fps={fps} → {len(frame_paths)}프레임")

        # ── YOLO 검출(공용 프레임) ───────────────────────────────────
        print("[3/5] YOLO 검출...")
        frame_refs = [FrameRef(frame_idx=i, timestamp=times[i], image=p)
                      for i, p in enumerate(frame_paths)]
        try:
            detections_by_frame = self.detector.detect(frame_refs, meta)
            flat_dets: List[Detection] = [d for fd in detections_by_frame for d in fd.detections]
        except Exception as e:  # YOLO 실패해도 파이프라인 진행(부품주입/위치힌트만 손해)
            print(f"      [WARN] YOLO 건너뜀: {e}")
            flat_dets = []

        # ── VLM 관찰(공용 프레임) ────────────────────────────────────
        print("[4/5] VLM 관찰...")
        injected = self._injected_parts(meta.video_id, flat_dets)
        actions = self.vlm.observe_frames(frame_paths, times, injected_parts=injected)
        self._save_observations(meta.video_id, actions)  # VLM 원시 관찰 디스크 저장(디버깅/검토용)
        if hasattr(self.vlm, "unload"):
            self.vlm.unload()  # 단일 GPU 순차: LLM 로드 전 비움

        # ── 융합 ──────────────────────────────────────────────────────
        print("[5/5] LLM 융합...")
        windows = self.aligner.align(actions, transcript, flat_dets)
        doc = self.llm.fuse(windows, meta)

        # 우리가 이미 아는 결정적 메타데이터는 코드가 채운다(LLM 추측 금지 — #1).
        self._finalize_metadata(doc, dur)

        # 교차검증: VLM 관찰문/STT 발화문을 넘겨 '관찰 둔갑'·'발화 지어냄'까지 잡는다(#3).
        obs_texts = [a.action for a in actions]
        utt_texts = [u.raw_text for u in transcript.utterances] + \
                    [u.normalized_text for u in transcript.utterances]
        for c in doc.candidates:
            for w in c.cross_check(observation_texts=obs_texts, utterance_texts=utt_texts):
                print(f"[CROSS-CHECK][{c.id}] {w}")

        self._save(doc)
        return doc

    def _finalize_metadata(self, doc: TacitKnowledgeDocument, dur: float) -> None:
        """LLM이 추측하면 안 되는(우리가 이미 아는) 메타데이터를 authoritative 하게 채운다.

        id / scenario_id / source.video_id / transcript_ref / clip_start·clip_end 는
        전부 코드가 아는 값이다. LLM 출력을 신뢰하지 않고 여기서 덮어쓴다.
        """
        from .schema.intermediate import seconds_to_hhmmss

        video_id = doc.video_id
        stem = video_id.rsplit(".", 1)[0].replace("-", "_")
        transcript_dir = getattr(self.stt, "transcript_dir", "transcripts")
        transcript_ref = f"{transcript_dir}/{video_id}.json"
        clip_end = seconds_to_hhmmss(dur) if dur else None

        for i, c in enumerate(doc.candidates, start=1):
            c.id = f"tk_{stem}_{i:03d}"
            c.metadata.scenario_id = stem
            src = c.metadata.source
            src.video_id = video_id
            src.transcript_ref = transcript_ref
            src.clip_start = "00:00:00"
            src.clip_end = clip_end

    def _injected_parts(self, video_id: str, flat_dets: List[Detection]):
        detected = [d.cls for d in flat_dets]
        if hasattr(self.vlm, "_injected_parts_for"):
            return self.vlm._injected_parts_for(video_id, detected)
        return sorted(set(detected)) or None

    def _save_observations(self, video_id: str, actions: List["ActionDescription"]) -> str:
        """VLM 관찰 로그를 output/<video_id>.observations.json 로 저장(VLM이 뭘 적고 뭘 놓치는지 검토용)."""
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{video_id}.observations.json"
        payload = {"video_id": video_id, "n_observations": len(actions),
                   "observations": [a.model_dump() for a in actions]}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] VLM 관찰 {len(actions)}건 저장 → {path}")
        return str(path)

    def _save(self, doc: TacitKnowledgeDocument) -> str:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{doc.video_id}.tacit.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 암묵지 후보 {len(doc.candidates)}건 저장 → {path}")
        return str(path)
