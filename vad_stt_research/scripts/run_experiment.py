"""
전체 실험 실행 스크립트 (3-arm: A, A', B).
metadata.csv를 읽어 조건별로 모든 파일 처리 후 results/raw/results.csv 저장.

사용법:
  python scripts/run_experiment.py --metadata data/metadata.csv --config configs/experiment_config.yaml
"""
import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from experiments.condition_a import run_condition_a
from experiments.condition_a_prime import run_condition_a_prime
from experiments.condition_b import run_condition_b
from pipeline.stt.faster_whisper_runner import FasterWhisperRunner
from evaluation.wer_cer import evaluate_accuracy, segments_to_text
from evaluation.hallucination import (
    detect_hallucinations,
    compute_hallucination_rate,
    get_silence_intervals_from_gt,
)
from evaluation.timestamp_eval import compute_timestamp_drift
from pipeline.vad.base import SpeechSegment


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ground_truth(file_id: str, gt_dir: str) -> dict:
    """
    정답 데이터 로드.
    {text: str, segments: [{start, end}, ...]} 형식 JSON 예상.
    """
    gt_path = Path(gt_dir) / f"{file_id}.json"
    if not gt_path.exists():
        return {"text": "", "segments": []}
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    data["segments"] = [SpeechSegment(**s) for s in data.get("segments", [])]
    return data


def process_file(
    row: dict,
    cfg: dict,
    gt_dir: str,
    drift_dir: str,
    runner: FasterWhisperRunner,
    n_repeats: int = 3,
    warmup: int = 1,
) -> list:
    file_id = row["file_id"]
    audio_path = row["file_path"]
    audio_duration = float(row["duration_min"]) * 60.0

    gt = load_ground_truth(file_id, gt_dir)
    silence_intervals = get_silence_intervals_from_gt(gt["segments"], audio_duration)

    # 모델은 main에서 1회 로드해 재사용(중복 로드 방지). RTF 측정 구간 밖이라 결과 불변.
    stt_kwargs = dict(runner=runner, n_repeats=n_repeats, warmup=warmup)

    run_results = {
        "A": run_condition_a(audio_path, **stt_kwargs),
        "A_prime": run_condition_a_prime(audio_path, **stt_kwargs),
        "B": run_condition_b(audio_path, vad_params={**cfg["vad"]}, **stt_kwargs),
    }

    rows = []
    for cond, res in run_results.items():
        acc = ts_drift = {}
        if gt["text"]:
            acc = evaluate_accuracy(res["segments"], gt["text"])
        hall = detect_hallucinations(res["segments"], silence_intervals)
        hall_rate = compute_hallucination_rate(hall["events"], audio_duration)
        if gt["segments"]:
            ts_drift = compute_timestamp_drift(res["segments"], gt["segments"])
            # 그래프 5(타임스탬프 드리프트 추이)용 버킷 데이터를 별도 JSON 저장
            drift_path = Path(drift_dir) / f"{file_id}_{cond}.json"
            with open(drift_path, "w", encoding="utf-8") as f:
                json.dump(ts_drift.get("drift_by_time", []), f, ensure_ascii=False, indent=2)

        rows.append({
            "file_id": file_id,
            "silence_ratio": row.get("silence_ratio", ""),
            "condition": cond,
            "wer": acc.get("wer", ""),
            "cer": acc.get("cer", ""),
            "hallucination_per_hour": hall_rate,
            "timestamp_drift_late_s": ts_drift.get("mean_drift_late_s", ""),
            "rtf_mean": res.get("rtf_mean", ""),
            "rtf_std": res.get("rtf_std", ""),
            "vad_time_s": res.get("vad_time_s", ""),
            "n_chunks": res.get("n_chunks", ""),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--gt_dir", default="data/ground_truth")
    parser.add_argument("--output", default="results/raw/results.csv")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta_df = pd.read_csv(args.metadata)

    raw_dir = Path(args.output).parent
    drift_dir = raw_dir / "drift_by_time"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(drift_dir, exist_ok=True)

    # STT 모델 1회 로드 후 전 파일·전 조건 공유 (중복 로드 방지)
    runner = FasterWhisperRunner(
        cfg["stt"]["model"], cfg["stt"]["device"], cfg["stt"]["compute_type"]
    )

    all_rows = []
    for _, row in meta_df.iterrows():
        print(f"\n{'='*60}\n파일: {row['file_id']} (무음 {row.get('silence_ratio','?')})")
        try:
            rows = process_file(
                row.to_dict(), cfg, args.gt_dir, str(drift_dir), runner, n_repeats=args.repeats
            )
            all_rows.extend(rows)
            # 파일별 incremental 저장: 중간에 죽어도 완료분 보존
            pd.DataFrame(all_rows).to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"  [저장] {row['file_id']} 완료 → {args.output} (누적 {len(all_rows)}행)")
        except Exception as e:
            print(f"  [오류] {e}")

    print(f"\n결과 저장 완료: {args.output} (총 {len(all_rows)}행)")


if __name__ == "__main__":
    main()
