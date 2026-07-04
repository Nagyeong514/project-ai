"""
STEP3 실행 본체 — STT → 정제 → 프레임 추출(YOLO/VLM 공용, ffmpeg).

산출물(항상 파일로 남긴다 — 단독 실행/통합 실행 둘 다 이 파일들을 거친다):
  - transcripts/<video_id>.json           (정제까지 끝난 버전)
  - output/_frames/<video_id>/*.jpg       (video_id별 폴더 — 예전 덮어쓰기 버그 수정판)
  - output/_frames/<video_id>/frames_meta.json  (STEP4가 읽는 프레임 경로+시각 목록)
"""

from __future__ import annotations

from pathlib import Path

from tacit_common import artifacts
from tacit_common.config import PipelineConfig
from step3_components.frame_extract import extract_frames, probe_duration
from step3_registry import build_refiner, build_stt


class Step3Runner:
    def __init__(self, cfg: PipelineConfig, stt=None, refiner=None):
        self.cfg = cfg
        self.stt = stt or build_stt(cfg.stt)
        self.refiner = refiner or build_refiner(cfg.transcript_refine)
        # cwd가 STEP3_전처리든 파이프라인_통합실행이든 항상 같은 절대경로를 보게 강제
        # (STT 어댑터 자체의 params.transcript_dir은 상대경로라 cwd에 따라 달라짐 — 그
        # 값을 신뢰하지 않고 여기서 덮어쓴다).
        self.transcript_dir = cfg.paths.resolve(cfg.paths.step3_transcript_dir)
        self.frames_dir = cfg.paths.resolve(cfg.paths.step3_frames_dir)
        if hasattr(self.stt, "transcript_dir"):
            self.stt.transcript_dir = self.transcript_dir

    def _fps_for_duration(self, duration_sec: float) -> float:
        fe = self.cfg.frame_extraction
        if fe.fps_override is not None:
            return fe.fps_override
        return fe.fps_long if duration_sec >= fe.long_video_threshold_sec else fe.fps_short

    def run(self, video_path: str) -> str:
        """video_id를 반환한다 — STEP4/5가 이 video_id로 파일을 찾는다."""
        video_id = Path(video_path).name

        print(f"[STEP3][1/2] STT + 정제: {video_id}")
        transcript = self.stt.transcribe(video_path, video_id)
        transcript = self.refiner.refine(transcript)  # 정규화 + 반복감지
        artifacts.save_transcript(self.transcript_dir, transcript)  # 정제본으로 덮어쓰기
        if hasattr(self.stt, "unload"):
            self.stt.unload()

        print(f"[STEP3][2/2] 프레임 추출(ffmpeg): {video_id}")
        fe = self.cfg.frame_extraction
        dur = probe_duration(video_path, fe.ffmpeg_bin)
        fps = self._fps_for_duration(dur)
        video_frames_dir = str(Path(self.frames_dir) / video_id)
        frame_paths, times = extract_frames(
            video_path, fps, video_frames_dir,
            long_side=fe.long_side, ffmpeg_bin=fe.ffmpeg_bin,
        )
        artifacts.save_frames_meta(self.frames_dir, video_id, frame_paths, times, fps, dur)
        print(f"      duration={dur:.0f}s fps={fps} → {len(frame_paths)}프레임 → {video_frames_dir}")

        return video_id
