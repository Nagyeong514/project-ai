"""S2: ffmpeg 균등 프레임 추출. 모델 없음.

- 클립별 폴더(frames/<clip>/) — 예전 공용 _frames/ 덮어쓰기 버그 원천 차단.
- fps 단일값(0.5). 조건분기(긴 영상 0.15) 폐지 — 청크 구조라 프레임 수가 OOM 과 무관해졌고,
  0.3/0.4 대역의 '관찰 0건' 미스터리 fps 를 아예 안 밟는다.
- times.json 에 그리드를 저장 — S4 의 snap 기준이자 최종 timestamp 의 원천.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess

from core.config import Cfg
from core.paths import Contract, clip_id


def run(cfg: Cfg, force: bool = False):
    contract = Contract(cfg.paths.output_dir)
    contract.ensure_dirs()
    fps = cfg.frames.fps

    for video in cfg.paths.videos:
        cid = clip_id(video)
        fdir = contract.frames_dir(cid)
        if not force and contract.times(cid).exists():
            print(f"[S2] skip (exists): {cid}")
            continue

        fdir.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(fdir / "frame_*.jpg")):  # 재실행 시 이전 산출물 청소
            os.remove(f)

        ls = cfg.frames.long_side
        vf = f"fps={fps},scale='if(gt(iw,ih),{ls},-2)':'if(gt(iw,ih),-2,{ls})'"
        cmd = [
            cfg.paths.ffmpeg_bin, "-y", "-i", video,
            "-vf", vf, "-q:v", str(cfg.frames.jpeg_q),
            str(fdir / "frame_%04d.jpg"),
        ]
        print(f"[S2] extract: {cid} (fps={fps})")
        subprocess.run(cmd, check=True, capture_output=True)

        paths = sorted(glob.glob(str(fdir / "frame_*.jpg")))
        times = [i / fps for i in range(len(paths))]
        contract.times(cid).write_text(
            json.dumps({"fps": fps, "n_frames": len(paths), "times": times},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[S2] saved {len(paths)} frames → {fdir}")
