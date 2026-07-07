"""
VLM 단독 디버그 스크립트. STT/YOLO/LLM 전부 건너뛰고 VLM 관찰만 돌린다
(2026-07-04: CLIP1 VLM 0건 원인 확인용 — 무거운 다른 단계 로딩 없이 빠르게 반복 확인).

STEP3/4 분할 후: fps 정책은 STEP3 소관(config.frame_extraction)이라 VLM 객체가 아니라
cfg에서 직접 읽는다. 프레임은 이 스크립트가 임시 폴더에 즉석으로 뽑는다(실제 파이프라인은
STEP3가 만든 output/_frames/<video_id>/를 읽지만, 이건 임의 영상을 바로 찍어보는 용도).

사용: srun -p RTX6000 -w n5 --gres=gpu:1 --time=00:10:00 \
      ../파이프라인_통합실행/.venv/bin/python3 debug_vlm.py --video <영상경로>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "STEP3_전처리"))  # step3_components.frame_extract
sys.path.insert(0, str(ROOT))  # tacit_common

from step4_preflight import run_preflight
from step4_registry import build_vlm
from step3_components.frame_extract import extract_frames, probe_duration


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
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    ap.add_argument("--video", required=True)
    args = ap.parse_args()

    cfg = run_preflight(args.config)
    vlm = build_vlm(cfg.vlm)
    fe = cfg.frame_extraction

    dur = probe_duration(args.video, fe.ffmpeg_bin)
    fps = fe.fps_override if fe.fps_override is not None else (
        fe.fps_long if dur >= fe.long_video_threshold_sec else fe.fps_short)
    print(f"[DEBUG] duration={dur:.1f}s fps={fps} threshold={fe.long_video_threshold_sec}")

    frame_paths, times = extract_frames(args.video, fps, "/tmp/debug_vlm_frames",
                                         long_side=fe.long_side, ffmpeg_bin=fe.ffmpeg_bin)
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
