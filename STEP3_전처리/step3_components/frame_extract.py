"""
ffmpeg CLI 기반 프레임 추출 유틸 (팀 검증 방식 — 이 환경선 torchcodec/pyav/cv2 디코딩이 불안정).

이 머신엔 ffmpeg 바이너리가 있고(`/home/piai/anaconda3/envs/deep/bin/ffmpeg`), CLI subprocess가
Python 디코더 라이브러리 문제(libnvrtc/libavutil/경로인코딩)를 통째로 우회한다.
추출 jpg를 YOLO와 VLM이 공용으로 쓴다.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from typing import List, Tuple

DEFAULT_FFMPEG = "/home/piai/anaconda3/envs/deep/bin/ffmpeg"


def _ffmpeg_bin(ffmpeg_bin: str | None) -> str:
    return ffmpeg_bin or os.environ.get("FFMPEG_BIN") or DEFAULT_FFMPEG


def probe_duration(video: str, ffmpeg_bin: str | None = None) -> float:
    """ffmpeg stderr의 'Duration: HH:MM:SS.ss' 파싱 → 초. 실패 시 0.0."""
    p = subprocess.run([_ffmpeg_bin(ffmpeg_bin), "-i", video],
                       capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", p.stderr)
    if not m:
        return 0.0
    h, mm, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return h * 3600 + mm * 60 + s


def extract_frames(
    video: str,
    fps: float,
    out_dir: str,
    long_side: int = 480,
    jpeg_q: int = 5,
    ffmpeg_bin: str | None = None,
) -> Tuple[List[str], List[float]]:
    """video를 fps로 균등 추출 → (jpg경로 리스트, 각 프레임 timestamp(초) 리스트).

    timestamp는 fps 그리드에서 파생(i/fps) — 시간은 여기서 한 번만 부여하고 이후 단계는 그대로 들고 감(5.6).
    """
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "frame_*.jpg")):
        os.remove(f)
    vf = (f"fps={fps},scale="
          f"'if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'")
    subprocess.run(
        [_ffmpeg_bin(ffmpeg_bin), "-y", "-v", "error", "-i", video,
         "-vf", vf, "-q:v", str(jpeg_q), os.path.join(out_dir, "frame_%04d.jpg")],
        check=True,
    )
    paths = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
    times = [i / fps for i in range(len(paths))]
    return paths, times
