"""
메인 실험 실행 진입점.

사용법:
  cd stt_comparison_research
  export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
  PYTHONPATH=$(pwd) python scripts/run_experiment.py \
    --metadata data/metadata.csv \
    --config configs/experiment_config.yaml \
    --output results/raw/results.csv

선택 옵션:
  --models large-v3,turbo          # 특정 모델만 실행 (쉼표 구분)
  --kospeech-ckpt /path/to/las.pt  # Kospeech checkpoint 경로
  --skip-api                       # API 호출 건너뜀
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import yaml

# PYTHONPATH가 프로젝트 루트여야 함
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_all_models import run_all_models


RESULT_COLS = [
    "file_id", "utterance_type", "model_key", "model_label",
    "cer", "wer",
    "substitutions", "deletions", "insertions", "hits",
    "ins_rate", "del_rate", "length_ratio",
    "rtf_mean", "rtf_std", "rtf_note", "deployable",
]


def load_metadata(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ground_truth(file_id: str, gt_dir: str) -> str:
    gt_path = Path(gt_dir) / f"{file_id}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT 파일 없음: {gt_path}")
    return gt_path.read_text(encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output", default="results/raw/results.csv")
    parser.add_argument("--gt-dir", default="data/ground_truth")
    parser.add_argument("--models", default=None, help="쉼표 구분 모델 키 (기본: 전체)")
    parser.add_argument("--kospeech-ckpt", default=None)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_configs = cfg["models"]

    # checkpoint 경로 주입
    if args.kospeech_ckpt:
        model_configs["kospeech"]["checkpoint"] = args.kospeech_ckpt

    # 모델 필터
    if args.models:
        keys = [k.strip() for k in args.models.split(",")]
        model_configs = {k: v for k, v in model_configs.items() if k in keys}

    if args.skip_api:
        model_configs = {k: v for k, v in model_configs.items()
                         if v["engine"] not in ("clova_api", "kakao_api")}

    metadata = load_metadata(args.metadata)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if write_header:
            writer.writeheader()

        for row in metadata:
            file_id = row["file_id"]
            wav_path = row["wav_path"]
            utype = row["utterance_type"]

            print(f"\n[{file_id}] ({utype}) {wav_path}")
            gt = load_ground_truth(file_id, args.gt_dir)

            results = run_all_models(
                audio_path=wav_path,
                ground_truth=gt,
                model_configs=model_configs,
                rtf_cfg=cfg["rtf"],
            )

            for res in results:
                writer.writerow({
                    "file_id": file_id,
                    "utterance_type": utype,
                    **{k: res[k] for k in RESULT_COLS if k in res},
                })
                f.flush()
                print(f"  {res['model_key']:40s} CER={res['cer']:.3f}  RTF={res['rtf_mean']:.3f}±{res['rtf_std']:.3f} {res['rtf_note']}")

    print(f"\n완료. 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
