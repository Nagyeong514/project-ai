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

    @staticmethod
    def _log_gpu_mem(label: str) -> None:
        """스테이지 경계마다 실제 GPU 메모리를 찍는다(2026-07-02: 반복 OOM을 감으로
        max_new_tokens만 깎아가며 재시도하던 걸 멈추고, unload()가 실제로 메모리를
        비우는지·파편화인지 용량부족인지 계측으로 확인하기 위함).

        torch.cuda.memory_allocated()는 PyTorch 캐싱 allocator가 아는 것만 잡는다 —
        ctranslate2(faster-whisper 백엔드)처럼 자체 CUDA 메모리 풀을 쓰는 라이브러리는
        안 잡힌다. 그래서 nvidia-smi 실측치를 같이 찍어 진짜 전체 사용량과 대조한다.
        """
        import subprocess
        import torch  # noqa
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().split("\n")[0]
            used_mib, total_mib = (int(x.strip()) for x in out.split(","))
            real = f"nvidia-smi 실측={used_mib/1024:.2f}/{total_mib/1024:.2f}GiB"
        except Exception as e:
            real = f"nvidia-smi 실패({e})"
        print(f"      [GPU-MEM] {label}: torch_allocated={alloc:.2f}GiB torch_reserved={reserved:.2f}GiB "
              f"torch_peak={max_alloc:.2f}GiB | {real} "
              f"(reserved-allocated={reserved - alloc:.2f}GiB = torch 파편화 추정치, "
              f"실측-torch_reserved = torch가 모르는 타 라이브러리 점유분)")

    def run(self, video_path: str | None = None) -> TacitKnowledgeDocument:
        video_path = video_path or self.cfg.video_path
        if not video_path:
            raise ValueError("video_path 가 비었습니다. config.video_path 또는 --video 로 지정하세요.")
        meta = self._probe_meta(video_path)

        import os
        print(f"      [ENV] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}")
        self._log_gpu_mem("시작")

        # ── 음성 갈래 ────────────────────────────────────────────────
        print("[1/5] STT...")
        transcript = self.stt.transcribe(video_path, meta.video_id)
        transcript = self.refiner.refine(transcript)  # 정규화 + 반복감지
        self._log_gpu_mem("STT 완료(unload 전)")
        if hasattr(self.stt, "unload"):
            self.stt.unload()  # cuda 로드된 채 방치하면 뒤 단계(특히 LLM) OOM 여유 감소
        self._log_gpu_mem("STT unload 후")

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
        self._log_gpu_mem("YOLO 완료(unload 전)")
        if hasattr(self.detector, "unload"):
            self.detector.unload()  # VLM/LLM 단계 여유 확보(STT와 동일한 이유)
        self._log_gpu_mem("YOLO unload 후")

        # ── VLM 관찰(공용 프레임) ────────────────────────────────────
        print("[4/5] VLM 관찰...")
        injected = self._injected_parts(meta.video_id, flat_dets)
        actions = self.vlm.observe_frames(frame_paths, times, injected_parts=injected)
        self._save_observations(meta.video_id, actions)  # VLM 원시 관찰 디스크 저장(디버깅/검토용)
        self._log_gpu_mem("VLM 완료(unload 전)")
        if hasattr(self.vlm, "unload"):
            self.vlm.unload()  # 단일 GPU 순차: LLM 로드 전 비움
        self._log_gpu_mem("VLM unload 후")

        # ── 융합 ──────────────────────────────────────────────────────
        print("[5/5] LLM 융합...")
        windows = self.aligner.align(actions, transcript, flat_dets)
        doc = self.llm.fuse(windows, meta)
        self._log_gpu_mem("LLM 융합 완료")

        # 우리가 이미 아는 결정적 메타데이터는 코드가 채운다(LLM 추측 금지 — #1).
        self._finalize_metadata(doc, dur)

        # 교차검증: VLM 관찰문/STT 발화문을 넘겨 '관찰 둔갑'·'발화 지어냄'까지 잡는다(#3).
        # (2026-07-04: 시간 근접성 기반 evidence 강제 정정(enforce_evidence_grounding)은
        # 제거됨 — 새 설계는 시간이 아니라 순서로 매칭하므로 "행동/발화가 4초 넘게 떨어지면
        # 무관하다"는 전제가 더 이상 성립하지 않는다. 이제 timestamp는 LLM이 지어내는 게
        # 아니라 코드가 인덱스로 원본에서 verbatim 복사하므로, 이 함수가 막으려던 사고
        # 자체가 구조적으로 재발 불가능해졌다. cross_check()는 시간 근접성이 아니라
        # 정합성만 보므로 그대로 유지.)
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

        id / scenario_id / equipment / source.video_id / transcript_ref / clip_start·clip_end 는
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
            c.metadata.equipment = self.cfg.equipment
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
        """VLM 관찰 로그를 output/vlm_observations/<video_id>.observations.json 로 저장
        (STT는 transcripts/, 최종 융합은 output/tacit_json/ — 결과물을 단계별 폴더로 분리해
        한눈에 보기 쉽게 함. 2026-07-02)."""
        out_dir = Path(self.cfg.output_dir) / "vlm_observations"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{video_id}.observations.json"
        payload = {"video_id": video_id, "n_observations": len(actions),
                   "observations": [a.model_dump() for a in actions]}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] VLM 관찰 {len(actions)}건 저장 → {path}")
        return str(path)

    def _save(self, doc: TacitKnowledgeDocument) -> str:
        """최종 LLM 융합 결과를 output/tacit_json/ 에 저장(VLM 관찰과 폴더 분리)."""
        out_dir = Path(self.cfg.output_dir) / "tacit_json"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{doc.video_id}.tacit.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 암묵지 후보 {len(doc.candidates)}건 저장 → {path}")
        return str(path)
