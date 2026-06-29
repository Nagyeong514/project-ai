"""프레임 추출 — 저수준 유틸. 조건별 조립은 run_experiment 가 한다.

  extract(...)      : 한 시간창을 fps로 전부 추출 (상한 없음)
  uniform_pick(...) : 균등하게 n장만 남기고 나머지 삭제 (8GB 대응 다운샘플)
  chunk(...)        : 리스트를 size씩 나눔 (A 천장 청크용)
"""
from __future__ import annotations
import os
import subprocess

_FFMPEG = os.environ.get("FFMPEG_BIN") or "/home/piai/anaconda3/envs/deep/bin/ffmpeg"


def extract(video: str, start: float, end: float, out_dir: str, cfg: dict) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    f = cfg["frames"]
    ls = f["max_long_side"]
    vf = (f"fps={f['fps']},scale="
          f"'if(gt(iw,ih),{ls},-2)':'if(gt(iw,ih),-2,{ls})'")
    cmd = [
        _FFMPEG, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", video,
        "-vf", vf, "-q:v", str(int(31 - f["jpeg_quality"] * 30 / 100)),
        os.path.join(out_dir, "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(os.path.join(out_dir, x) for x in os.listdir(out_dir) if x.endswith(".jpg"))


def uniform_pick(paths: list[str], n) -> list[str]:
    """균등하게 n장만 남기고 나머지는 디스크에서 삭제."""
    if not n or len(paths) <= n:
        return paths
    step = len(paths) / n
    keep = {int(i * step) for i in range(n)}
    kept = []
    for i, p in enumerate(paths):
        (kept.append(p) if i in keep else os.remove(p))
    return kept


def chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]
