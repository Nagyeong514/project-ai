"""
모델별 전사 결과를 세그먼트 타임스탬프(start/end)와 함께 JSON으로 저장.

기존 실험(run_all_models.py/run_experiment_parallel.py)은 STTResult.segments를
longform stability 계산에만 쓰고 버렸음(파일로 저장 안 함) — 이 스크립트는 그
segments를 그대로 dump한다. RTF 측정용 반복 실행은 하지 않고 모델당 파일당 1회만
transcribe한다(타임스탬프 추출이 목적이라 반복 측정 불필요).

사용법 (01_STT선정/.venv 파이썬, LD_LIBRARY_PATH 필수 — run_experiment_parallel.py와 동일):
  PROJECT=$(pwd)
  CUBLAS_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
  CUDNN_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB:$LD_LIBRARY_PATH" .venv/bin/python3 \
    scripts/export_transcripts_with_timestamps.py \
    --config configs/experiment_config_v2.yaml \
    --metadata data/metadata_v2.csv \
    --output results/STT_v2_최종전사_타임스탬프.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.hallucination_filter import filter_hallucinations
from pipeline.stt import get_stt


def load_metadata(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config_v2.yaml")
    parser.add_argument("--metadata", default="data/metadata_v2.csv")
    parser.add_argument("--output", default="results/STT_v2_최종전사_타임스탬프.json")
    parser.add_argument("--models", default=None, help="쉼표 구분 모델 키 (기본: deployable=true 전체)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_configs = cfg["models"]

    if args.models:
        keys = [k.strip() for k in args.models.split(",")]
        model_configs = {k: v for k, v in model_configs.items() if k in keys}
    else:
        model_configs = {k: v for k, v in model_configs.items() if v.get("deployable", False)}

    metadata = load_metadata(args.metadata)

    output = {}
    for model_key, model_cfg in model_configs.items():
        print(f"[{model_key}] 모델 로딩...", flush=True)
        runner = get_stt(model_cfg)
        output[model_key] = {"label": model_cfg.get("label", model_key), "clips": {}}

        for row in metadata:
            file_id = row["file_id"]
            wav_path = row["wav_path"]
            print(f"[{model_key}] {file_id} 전사 중...", flush=True)

            result = runner.transcribe(wav_path)
            full_text = result.full_text()

            output[model_key]["clips"][file_id] = {
                "audio_duration_s": result.audio_duration_s,
                "processing_time_s": round(result.processing_time_s, 4),
                "full_text": full_text,
                "full_text_filtered": filter_hallucinations(full_text),
                "segments": [
                    {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text}
                    for seg in result.segments
                ],
            }

        del runner  # 다음 모델 로딩 전 GPU 메모리 반환

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
