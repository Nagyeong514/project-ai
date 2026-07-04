"""
STEP4 단독 실행 엔트리포인트.

사용:
    python run_step4.py --config ../파이프라인_통합실행/config.yaml --video-id CLIP1_정상조립과정.mp4

STEP3_전처리/run_step3.py 를 먼저 돌려서 output/_frames/<video_id>/ 가 있어야 한다.
산출물: output/detections/<video_id>.json, output/vlm_observations/<video_id>.observations.json
→ STEP5_LLM으로_암묵지_후보생성/run_step5.py 가 이 산출물을 읽어서 이어 돈다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import argparse

from step4_preflight import run_preflight
from step4_runner import Step4Runner


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP4(YOLO 검출 + VLM 관찰) 단독 실행")
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml", help="config.yaml 경로")
    ap.add_argument("--video-id", required=True, help="STEP3가 만든 video_id(파일명 그대로)")
    args = ap.parse_args()

    cfg = run_preflight(args.config)
    runner = Step4Runner(cfg)
    runner.run(args.video_id)
    print(f"[OK] STEP4 완료: {args.video_id}")


if __name__ == "__main__":
    main()
