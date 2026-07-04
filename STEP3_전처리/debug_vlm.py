"""
VLM 단독 디버그 스크립트. STT/YOLO/LLM 전부 건너뛰고 VLM 관찰만 돌린다
(2026-07-04: CLIP1 VLM 0건 원인 확인용 — 무거운 다른 단계 로딩 없이 빠르게 반복 확인).

사용: srun -p RTX6000 -w n5 --gres=gpu:1 --time=00:10:00 \
      .venv/bin/python3 debug_vlm.py --config config.yaml --video <영상경로>
"""
from __future__ import annotations

import argparse
import time

from preflight import run_preflight
from tacit_pipeline.components.frame_extract import extract_frames, probe_duration
from tacit_pipeline.registry import build_vlm


def _gpu_mem_str() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "(GPU 없음)"
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return f"allocated={alloc:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB"
    except Exception as e:
        return f"(측정 실패: {e})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", required=True)
    args = ap.parse_args()

    cfg = run_preflight(args.config, [args.video])
    vlm = build_vlm(cfg.vlm)

    ffmpeg_bin = getattr(vlm, "ffmpeg_bin", None)
    long_side = getattr(vlm, "long_side", 480)
    frames_dir = getattr(vlm, "frames_dir", f"{cfg.output_dir}/_frames")

    dur = probe_duration(args.video, ffmpeg_bin)
    fps = vlm.fps_for_duration(dur) if hasattr(vlm, "fps_for_duration") else 0.5
    print(f"[DEBUG] duration={dur:.1f}s fps={fps} threshold={getattr(vlm, 'long_video_threshold_sec', None)}")

    frame_paths, times = extract_frames(args.video, fps, frames_dir, long_side=long_side, ffmpeg_bin=ffmpeg_bin)
    print(f"[DEBUG] n_frames={len(frame_paths)} first_times={times[:5]} last_times={times[-5:]}")

    print(f"[DEBUG][GPU-MEM] VLM 관찰 시작 전: {_gpu_mem_str()}")
    t0 = time.monotonic()
    actions = vlm.observe_frames(frame_paths, times, injected_parts=None)
    elapsed = time.monotonic() - t0
    print(f"[DEBUG][TIMING] VLM 관찰 소요시간: {elapsed:.1f}s ({len(frame_paths)}프레임, fps={fps})")
    print(f"[DEBUG][GPU-MEM] VLM 관찰 완료 후: {_gpu_mem_str()}")

    print("=" * 80)
    print(f"[PARSED] {len(actions)}건")
    for a in actions:
        print(f"  - ts={a.timestamp} actor={a.actor} action={a.action!r} objects={a.objects}")
    print("=" * 80)


if __name__ == "__main__":
    main()
