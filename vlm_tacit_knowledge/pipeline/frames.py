"""A/B/C 프레임 추출.

세 조건 모두 동일 fps(2)로 추출 — 차이는 '어느 구간을 보느냐'뿐.
  A: 통영상 한 번에 2fps (upper bound)
  B: 발화 구간(pad400)별 2fps          ← 가설
  C: B 각 구간과 동일 길이의 '랜덤 위치' 창 2fps ← 대조군(위치 효과 분리)

C의 STT 텍스트는 cfg.control.c_text_source 에 따름:
  'segment'(권장): B와 같은 구간 텍스트 동봉 → 텍스트 고정, 영상 위치만 변수.
  'window'        : 랜덤창의 실제 전사(대개 무음).
"""
from __future__ import annotations
import os
import random
import subprocess

_FFMPEG = os.environ.get("FFMPEG_BIN") or "/home/piai/anaconda3/envs/deep/bin/ffmpeg"


def _scale_filter(long_side: int) -> str:
    # 긴 변을 long_side로 제한 (세로/가로 자동), 짝수 보정(-2)
    return (f"fps={{fps}},scale="
            f"'if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'")


def _extract_window(video: str, start: float, end: float, out_dir: str, cfg: dict) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    f = cfg["frames"]
    vf = _scale_filter(f["max_long_side"]).format(fps=f["fps"])
    cmd = [
        _FFMPEG, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", video,
        "-vf", vf, "-q:v", str(int(31 - f["jpeg_quality"] * 30 / 100)),
        os.path.join(out_dir, "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(
        os.path.join(out_dir, x) for x in os.listdir(out_dir) if x.endswith(".jpg")
    )


def extract_A(video: str, dur: float, work: str, cfg: dict) -> dict:
    out = os.path.join(work, "A")
    frames = _extract_window(video, 0.0, dur, out, cfg)
    return {"condition": "A", "window": [0.0, dur], "frames": frames}


def extract_B(video: str, segments, seg_texts, work: str, cfg: dict) -> list[dict]:
    items = []
    for i, ((a, b), txt) in enumerate(zip(segments, seg_texts)):
        out = os.path.join(work, "B", f"seg{i:02d}")
        frames = _extract_window(video, a, b, out, cfg)
        items.append({"condition": "B", "seg_idx": i, "window": [a, b],
                      "stt": txt, "frames": frames})
    return items


def extract_C(video: str, dur: float, segments, seg_texts, work: str, cfg: dict) -> list[dict]:
    """각 구간 i에 대해 동일 '길이'의 랜덤 위치 창을 뽑아 페어 구성."""
    rng = random.Random(cfg["control"]["c_random_seed"])
    text_src = cfg["control"]["c_text_source"]
    items = []
    for i, ((a, b), txt) in enumerate(zip(segments, seg_texts)):
        length = b - a
        if length >= dur:
            ra, rb = 0.0, dur
        else:
            ra = rng.uniform(0.0, dur - length)
            rb = ra + length
        out = os.path.join(work, "C", f"seg{i:02d}")
        frames = _extract_window(video, ra, rb, out, cfg)
        items.append({"condition": "C", "seg_idx": i, "window": [ra, rb],
                      "stt": txt if text_src == "segment" else None,  # window 전사는 호출측에서
                      "stt_source": text_src, "frames": frames})
    return items
