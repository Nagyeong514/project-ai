"""
STEP5 단독 실행 엔트리포인트.

사용:
    python run_step5.py --config ../파이프라인_통합실행/config.yaml --video-id CLIP1_정상조립과정.mp4

STEP3_전처리/run_step3.py + STEP4_YOLO_VLM관찰/run_step4.py 를 먼저 돌려서
transcript/detections/observations 가 있어야 한다.
산출물: output/aligned_windows/<video_id>.windows.json, output/tacit_json/<video_id>.tacit.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import argparse

from step5_preflight import run_preflight
from step5_runner import Step5Runner


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP5(정렬 + LLM 융합) 단독 실행")
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml", help="config.yaml 경로")
    ap.add_argument("--video-id", required=True, help="STEP3가 만든 video_id(파일명 그대로)")
    args = ap.parse_args()

    cfg = run_preflight(args.config)
    runner = Step5Runner(cfg)
    doc = runner.run(args.video_id)
    print(f"[OK] STEP5 완료: {args.video_id} → 후보 {len(doc.candidates)}건")


if __name__ == "__main__":
    main()
