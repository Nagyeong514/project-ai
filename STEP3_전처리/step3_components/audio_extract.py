"""
ffmpeg CLI 기반 오디오 추출 유틸 (frame_extract.py와 같은 이유로 subprocess 방식 —
이 환경선 Python 디코더 라이브러리가 불안정하고, ffmpeg CLI가 경로/코덱 문제를 통째로 우회).

기본 출력은 16kHz mono WAV(pcm_s16le):
  - STT(faster-whisper) 표준 입력 포맷이라 재샘플링 없이 바로 쓸 수 있고,
  - imageio-ffmpeg 정적 바이너리에 mp3 인코더(libmp3lame)가 없어도 항상 동작한다.
"""

from __future__ import annotations

import os
import subprocess

from step3_components.frame_extract import _ffmpeg_bin, probe_duration


def extract_audio(
    video: str,
    out_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_bin: str | None = None,
) -> str:
    """video에서 오디오 트랙만 뽑아 out_path(WAV)로 저장하고 경로를 돌려준다.

    실패(오디오 트랙 없음/ffmpeg 에러)면 RuntimeError — 조용히 빈 파일을 남기지 않는다.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        _ffmpeg_bin(ffmpeg_bin), "-y",
        "-i", video,
        "-vn",                     # 비디오 스트림 제거
        "-acodec", "pcm_s16le",    # WAV — 정적 ffmpeg에도 항상 있음
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-loglevel", "error",
        out_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 오디오 추출 실패({video}): {p.stderr.strip()[:300]}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"오디오 추출 결과가 비어 있음: {out_path}")
    return out_path
