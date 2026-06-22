"""
VAD 파라미터 민감도 분석 (가설 3 검증).
threshold × speech_pad_ms 스윕 후 WER 변화 관찰.

사용법:
  python scripts/sensitivity_analysis.py --audio path/to/audio.wav --gt path/to/gt.json
"""
import argparse
import itertools
import json

import pandas as pd

from pipeline.vad.silero_vad import SileroVAD
from pipeline.merge.chunk_extractor import extract_chunks
from pipeline.stt.faster_whisper_runner import FasterWhisperRunner
from experiments.condition_a_prime import DECODING_PARAMS_UNIFIED
from evaluation.wer_cer import evaluate_accuracy
from pipeline.vad.base import SpeechSegment


THRESHOLD_VALUES = [0.3, 0.5, 0.7]
SPEECH_PAD_VALUES = [0, 200, 400]


def run_sweep(audio_path: str, gt_text: str, model_size: str = "large-v3") -> pd.DataFrame:
    runner = FasterWhisperRunner(model_size)
    rows = []

    for threshold, pad_ms in itertools.product(THRESHOLD_VALUES, SPEECH_PAD_VALUES):
        print(f"  threshold={threshold}, speech_pad_ms={pad_ms} ...", end=" ", flush=True)
        vad = SileroVAD(
            threshold=threshold,
            speech_pad_ms=pad_ms,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            merge_gap_ms=200,
            max_chunk_s=30.0,
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            segments = vad.detect(audio_path)
            chunk_info = extract_chunks(audio_path, segments, output_dir=tmp)
            all_segs = []
            for chunk_path, offset in chunk_info:
                result = runner.transcribe(chunk_path, DECODING_PARAMS_UNIFIED, chunk_offset=offset)
                all_segs.extend(result.segments)

        acc = evaluate_accuracy(all_segs, gt_text)
        print(f"WER={acc['wer']:.4f}")
        rows.append({
            "threshold": threshold,
            "speech_pad_ms": pad_ms,
            "wer": acc["wer"],
            "cer": acc["cer"],
            "n_segments": len(segments),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--gt", required=True, help="정답 JSON {text: str}")
    parser.add_argument("--output", default="results/figures/sensitivity_wer.csv")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    with open(args.gt, encoding="utf-8") as f:
        gt_text = json.load(f)["text"]

    print(f"민감도 분석 시작: {args.audio}")
    df = run_sweep(args.audio, gt_text, model_size=args.model)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {args.output}")
    print(df.pivot(index="threshold", columns="speech_pad_ms", values="wer").to_string())


if __name__ == "__main__":
    main()
