"""
STEP3 단독 실행 엔트리포인트.

사용:
    python run_step3.py --config ../파이프라인_통합실행/config.yaml --video /path/to/CLIP1.mp4

산출물: transcripts/<video_id>.json, output/_frames/<video_id>/*.jpg + frames_meta.json
→ STEP4_YOLO_VLM관찰/run_step4.py 가 이 산출물을 읽어서 이어 돈다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import argparse

from step3_preflight import run_preflight
from step3_runner import Step3Runner


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP3 전처리(STT+정제, 프레임 추출) 단독 실행")
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml", help="config.yaml 경로")
    ap.add_argument("--video", required=True, help="영상 경로")
    args = ap.parse_args()

    cfg = run_preflight(args.config)
    runner = Step3Runner(cfg)
    video_id = runner.run(args.video)
    print(f"[OK] STEP3 완료: {video_id}")


if __name__ == "__main__":
    main()
