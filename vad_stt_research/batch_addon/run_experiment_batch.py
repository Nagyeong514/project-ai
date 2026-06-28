"""
배치 추론 추가 실험 드라이버 (4-arm: A', A'_batch, B, B_batch).

기존 비배치 실험(scripts/run_experiment.py)과 분리. RTF 공정 비교를 위해
4조건을 같은 세션에서 새로 측정한다. 기존 results_*.csv는 건드리지 않고
batch_addon/results_batch.csv로 저장한다.

비교 축:
  A'  ↔ A'_batch : 배치 효과 (VAD 없음 계열)
  B   ↔ B_batch  : 배치 효과 (Silero VAD 계열)

실행:
  export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
  PYTHONPATH=/home/piai/project-ai/vad_stt_research \
    /home/piai/anaconda3/bin/python batch_addon/run_experiment_batch.py \
    --metadata data/metadata_10.csv \
    --config configs/experiment_config.yaml \
    --output batch_addon/results_batch.csv \
    --batch_size 8 --batch_size_fallback 4 --repeats 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# batch_addon/을 import 경로에 추가(드라이버를 파일로 직접 실행할 때 condition_batch 해석용)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from condition_batch import run_condition_a_prime_batch, run_condition_b_batch  # noqa: E402

# 기존(비배치) 조건과 평가 코드는 그대로 재사용 — 원본 불변
from experiments.condition_a_prime import run_condition_a_prime  # noqa: E402
from experiments.condition_b import run_condition_b  # noqa: E402
from pipeline.stt.faster_whisper_runner import FasterWhisperRunner  # noqa: E402
from pipeline.vad.base import SpeechSegment  # noqa: E402
from evaluation.wer_cer import evaluate_accuracy  # noqa: E402
from evaluation.hallucination import (  # noqa: E402
    detect_hallucinations,
    compute_hallucination_rate,
    get_silence_intervals_from_gt,
)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ground_truth(file_id: str, gt_dir: str) -> dict:
    gt_path = Path(gt_dir) / f"{file_id}.json"
    if not gt_path.exists():
        return {"text": "", "segments": []}
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    data["segments"] = [SpeechSegment(**s) for s in data.get("segments", [])]
    return data


def _row(res: dict, file_id: str, silence_ratio, gt: dict, audio_duration: float) -> dict:
    """공통 평가 → CSV 행. 배치/비배치 모두 동일 지표(WER/CER/할루시/RTF)로 채점."""
    silence_intervals = get_silence_intervals_from_gt(gt["segments"], audio_duration)
    acc = evaluate_accuracy(res["segments"], gt["text"]) if gt["text"] else {}
    hall = detect_hallucinations(res["segments"], silence_intervals)
    hall_rate = compute_hallucination_rate(hall["events"], audio_duration)
    return {
        "file_id": file_id,
        "silence_ratio": silence_ratio,
        "condition": res["condition"],
        "batched": res.get("batched", False),
        "batch_size": res.get("batch_size", ""),
        "oom_fallback": res.get("oom_fallback", False),
        "wer": acc.get("wer", ""),
        "cer": acc.get("cer", ""),
        "hallucination_per_hour": hall_rate,
        "rtf_mean": res.get("rtf_mean", ""),
        "rtf_std": res.get("rtf_std", ""),
        "vad_time_s": res.get("vad_time_s", ""),
        "n_chunks": res.get("n_chunks", ""),
    }


def process_file(row: dict, cfg: dict, gt_dir: str, runner, args) -> list:
    file_id = row["file_id"]
    audio_path = row["file_path"]
    audio_duration = float(row["duration_min"]) * 60.0
    silence_ratio = row.get("silence_ratio", "")
    gt = load_ground_truth(file_id, gt_dir)

    seq_kwargs = dict(runner=runner, n_repeats=args.repeats, warmup=1)
    batch_kwargs = dict(
        runner=runner, n_repeats=args.repeats, warmup=1,
        batch_size=args.batch_size, batch_size_fallback=args.batch_size_fallback,
    )

    # 4조건을 같은 세션에서 측정(RTF 공정 비교)
    run_results = [
        run_condition_a_prime(audio_path, **seq_kwargs),
        run_condition_a_prime_batch(audio_path, **batch_kwargs),
        run_condition_b(audio_path, vad_params={**cfg["vad"]}, **seq_kwargs),
        run_condition_b_batch(audio_path, vad_params={**cfg["vad"]}, **batch_kwargs),
    ]
    return [_row(res, file_id, silence_ratio, gt, audio_duration) for res in run_results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata_10.csv")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--gt_dir", default="data/ground_truth")
    parser.add_argument("--output", default="batch_addon/results_batch.csv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size_fallback", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta_df = pd.read_csv(args.metadata)
    os.makedirs(Path(args.output).parent, exist_ok=True)

    # STT 모델 1회 로드 후 4조건·전 파일 공유(비배치 runner와 배치 pipeline이 같은 모델 사용)
    runner = FasterWhisperRunner(
        cfg["stt"]["model"], cfg["stt"]["device"], cfg["stt"]["compute_type"]
    )

    all_rows = []
    for _, row in meta_df.iterrows():
        print(f"\n{'='*60}\n파일: {row['file_id']} (무음 {row.get('silence_ratio','?')})")
        try:
            rows = process_file(row.to_dict(), cfg, args.gt_dir, runner, args)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"  [저장] {row['file_id']} 완료 → {args.output} (누적 {len(all_rows)}행)")
        except Exception as e:
            print(f"  [오류] {row['file_id']}: {e}")

    print(f"\n결과 저장 완료: {args.output} (총 {len(all_rows)}행)")


if __name__ == "__main__":
    main()
