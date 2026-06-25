"""
VAD 파라미터 민감도 분석 (가설 3 검증).
threshold × speech_pad_ms 스윕 후 WER 변화 관찰.

사용법 (단일 파일):
  python scripts/sensitivity_analysis.py --audio path/to/audio.wav --gt path/to/gt.json

사용법 (메타데이터 CSV):
  python scripts/sensitivity_analysis.py --metadata data/metadata_8.csv \
      --gt_dir data/ground_truth --output results/figures/sensitivity_wer.csv
"""
import argparse
import itertools
import json
import os
import tempfile

import pandas as pd

from pipeline.vad.silero_vad import SileroVAD
from pipeline.merge.chunk_extractor import extract_chunks
from pipeline.stt.faster_whisper_runner import FasterWhisperRunner
from experiments.condition_a_prime import DECODING_PARAMS_UNIFIED
from evaluation.wer_cer import evaluate_accuracy


THRESHOLD_VALUES = [0.3, 0.5, 0.7]
SPEECH_PAD_VALUES = [0, 200, 400]


def run_sweep_single(audio_path: str, gt_text: str, runner: FasterWhisperRunner) -> list[dict]:
    rows = []
    for threshold, pad_ms in itertools.product(THRESHOLD_VALUES, SPEECH_PAD_VALUES):
        print(f"    threshold={threshold}, speech_pad_ms={pad_ms} ...", end=" ", flush=True)
        vad = SileroVAD(
            threshold=threshold,
            speech_pad_ms=pad_ms,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            merge_gap_ms=200,
            max_chunk_s=30.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            segments = vad.detect(audio_path)
            chunk_info = extract_chunks(audio_path, segments, output_dir=tmp)
            all_segs = []
            for chunk_path, offset in chunk_info:
                result = runner.transcribe(chunk_path, DECODING_PARAMS_UNIFIED, chunk_offset=offset)
                all_segs.extend(result.segments)

        acc = evaluate_accuracy(all_segs, gt_text)
        print(f"WER={acc['wer']:.4f}  n_chunks={len(segments)}")
        rows.append({
            "threshold": threshold,
            "speech_pad_ms": pad_ms,
            "wer": acc["wer"],
            "cer": acc["cer"],
            "n_chunks": len(segments),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--gt_dir", default="data/ground_truth")
    parser.add_argument("--file_ids", nargs="*", default=None, help="분석할 file_id 목록 (미지정 시 전체)")
    parser.add_argument("--output", default="results/figures/sensitivity_wer.csv")
    parser.add_argument("--model", default="large-v3-turbo")
    args = parser.parse_args()

    runner = FasterWhisperRunner(args.model)
    all_rows = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def _save():
        pd.DataFrame(all_rows).to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.metadata:
        meta = pd.read_csv(args.metadata, encoding="utf-8-sig")  # BOM 안전
        if args.file_ids:
            meta = meta[meta.file_id.isin(args.file_ids)]
        for _, row in meta.iterrows():
            fid = row["file_id"]
            audio_path = os.path.join("data/raw", f"{fid}.wav")
            gt_path = os.path.join(args.gt_dir, f"{fid}.json")
            with open(gt_path, encoding="utf-8") as f:
                gt_text = json.load(f)["text"]
            sr = row.get("silence_ratio", None)
            sr_str = f"{sr:.4f}" if isinstance(sr, (int, float)) else str(sr)
            print(f"\n[{fid}] silence_ratio={sr_str}")
            rows = run_sweep_single(audio_path, gt_text, runner)
            for r in rows:
                r["file_id"] = fid
                r["silence_ratio"] = sr
            all_rows.extend(rows)
            _save()  # 파일별 incremental 저장 (긴 스윕 중간 실패 보호)
            print(f"  [저장] {fid} 스윕 완료 → {args.output} (누적 {len(all_rows)}행)")
    else:
        with open(args.gt, encoding="utf-8") as f:
            gt_text = json.load(f)["text"]
        print(f"민감도 분석 시작: {args.audio}")
        rows = run_sweep_single(args.audio, gt_text, runner)
        for r in rows:
            r["file_id"] = os.path.basename(args.audio)
        all_rows.extend(rows)
        _save()

    print(f"\n결과 저장: {args.output}")

    for fid in df.file_id.unique():
        sub = df[df.file_id == fid]
        print(f"\n[{fid}] WER pivot (threshold × speech_pad_ms):")
        print(sub.pivot(index="threshold", columns="speech_pad_ms", values="wer").round(4).to_string())


if __name__ == "__main__":
    main()
