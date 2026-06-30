"""
파이프라인 엔트리포인트.

내일 서버에서:
    python run.py --config config.yaml --video /path/to/master_S1_take3.mp4

config.yaml 의 video_path 를 채웠다면 --video 생략 가능:
    python run.py --config config.yaml

(오늘은 실행하지 않는다 — 모델/영상이 없으므로.)
"""

from __future__ import annotations

import argparse

from tacit_pipeline import Pipeline, PipelineConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="암묵지 후보 생성 전처리 파이프라인")
    ap.add_argument("--config", default="config.yaml", help="config.yaml 경로")
    ap.add_argument("--video", default=None, help="영상 경로(config.video_path 덮어씀)")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    pipe = Pipeline(cfg)
    pipe.run(args.video)


if __name__ == "__main__":
    main()
