"""
클립 1개만 통합 실행(STEP3→4→5를 이 프로세스 안에서 순서대로).

    python run_one_clip.py --config config.yaml --video /path/to/CLIP1.mp4

여러 클립을 한 번에 돌릴 땐 run_all_clips.py(모델 재로딩 없이 루프)를 쓴다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in ["STEP3_전처리", "STEP4_YOLO_VLM관찰", "STEP5_LLM으로_암묵지_후보생성"]:
    sys.path.insert(0, str(ROOT / _p))
sys.path.insert(0, str(ROOT))  # tacit_common

from preflight import run_preflight


def main() -> None:
    ap = argparse.ArgumentParser(description="암묵지 후보 생성 전처리 파이프라인(클립 1개)")
    ap.add_argument("--config", default="config.yaml", help="config.yaml 경로")
    ap.add_argument("--video", default=None, help="영상 경로(config.video_path 덮어씀)")
    args = ap.parse_args()

    cfg = run_preflight(args.config, [args.video] if args.video else None)
    video = args.video or cfg.video_path

    from step3_runner import Step3Runner
    from step4_runner import Step4Runner
    from step5_runner import Step5Runner

    video_id = Step3Runner(cfg).run(video)
    Step4Runner(cfg).run(video_id)
    doc = Step5Runner(cfg).run(video_id)
    print(f"[OK] 완료: {video_id} → 후보 {len(doc.candidates)}건")


if __name__ == "__main__":
    main()
