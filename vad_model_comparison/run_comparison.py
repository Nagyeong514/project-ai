"""
VAD 엔진 비교 실험: 참조(VAD없음) / Silero / WebRTC.
파일별 3조건 처리 후 결과 CSV 저장 (incremental).

측정: VAD RTF, 검출 F1/P/R, 청크수·평균길이, STT RTF, WER/CER, 할루시네이션.
STT·청크추출·평가·디코딩 파라미터는 Phase 1(`vad_stt_research/`) 재사용 → 단일 변수(엔진)만 변동.

사용:
  cd vad_model_comparison
  PYTHONPATH=. python run_comparison.py                       # 전체 10파일
  PYTHONPATH=. python run_comparison.py --file_ids H03 --max_seconds 120   # 스모크
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Phase 1 재사용 — 루트 pipeline.py 충돌 방지 위해 맨 앞에 주입
_PHASE1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vad_stt_research"))
if _PHASE1 not in sys.path:
    sys.path.insert(0, _PHASE1)

import numpy as np
import pandas as pd
import soundfile as sf

from pipeline.stt.faster_whisper_runner import FasterWhisperRunner
from pipeline.merge.chunk_extractor import extract_chunks
from pipeline.vad.base import SpeechSegment
from evaluation.wer_cer import evaluate_accuracy
from evaluation.hallucination import (
    detect_hallucinations,
    compute_hallucination_rate,
    get_silence_intervals_from_gt,
)
from experiments.condition_a_prime import DECODING_PARAMS_UNIFIED

from engines import get_engine
from metrics.detection_f1 import detection_f1

COLUMNS = [
    "file_id", "group", "silence_ratio", "engine",
    "vad_rtf", "precision", "recall", "f1", "n_chunks", "avg_chunk_s",
    "stt_rtf", "wer", "cer", "hallucination_per_hour",
]


def load_gt(file_id: str, gt_dir: str):
    data = json.load(open(Path(gt_dir) / f"{file_id}.json", encoding="utf-8"))
    segs = [SpeechSegment(**s) for s in data.get("segments", [])]
    return data.get("text", ""), segs


def truncate(audio_path: str, gt_segs, max_s: int, tmpdir: str):
    """스모크용: 앞 max_s초만 사용 + GT도 동일 구간으로 자름."""
    audio, sr = sf.read(audio_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio[: sr * max_s]
    out = os.path.join(tmpdir, "trunc.wav")
    sf.write(out, audio, sr, subtype="PCM_16")
    dur = len(audio) / sr
    segs = [SpeechSegment(s.start, min(s.end, dur)) for s in gt_segs if s.start < dur]
    return out, segs, dur


def eval_block(pred_segments, gt_text, sil_intervals, dur):
    acc = evaluate_accuracy(pred_segments, gt_text) if gt_text else {}
    hall = detect_hallucinations(pred_segments, sil_intervals)
    return {
        "wer": acc.get("wer", ""),
        "cer": acc.get("cer", ""),
        "hallucination_per_hour": compute_hallucination_rate(hall["events"], dur),
    }


def run_reference(runner, audio_path, dur, gt_text, sil_intervals):
    t = time.perf_counter()
    res = runner.transcribe(audio_path, DECODING_PARAMS_UNIFIED)
    stt_rtf = (time.perf_counter() - t) / dur
    row = {"engine": "reference", "vad_rtf": "", "precision": "", "recall": "",
           "f1": "", "n_chunks": "", "avg_chunk_s": "", "stt_rtf": stt_rtf}
    row.update(eval_block(res.segments, gt_text, sil_intervals, dur))
    return row


def run_vad_engine(name, runner, audio_path, dur, gt_text, gt_segs, sil_intervals, vad_repeats=3):
    engine = get_engine(name)
    engine.detect(audio_path)  # warmup
    times = []
    segs = []
    for _ in range(vad_repeats):
        t = time.perf_counter()
        segs = engine.detect(audio_path)
        times.append(time.perf_counter() - t)
    vad_rtf = float(np.mean(times)) / dur
    f1 = detection_f1(segs, gt_segs, dur)

    with tempfile.TemporaryDirectory(prefix="vmc_chunks_") as d:
        chunks = extract_chunks(audio_path, segs, output_dir=d)
        t = time.perf_counter()
        res = runner.transcribe_chunks(chunks, DECODING_PARAMS_UNIFIED)
        stt_rtf = (time.perf_counter() - t) / dur

    avg_chunk = float(np.mean([s.duration for s in segs])) if segs else 0.0
    row = {"engine": name, "vad_rtf": vad_rtf, "precision": f1["precision"],
           "recall": f1["recall"], "f1": f1["f1"], "n_chunks": len(segs),
           "avg_chunk_s": avg_chunk, "stt_rtf": stt_rtf}
    row.update(eval_block(res.segments, gt_text, sil_intervals, dur))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(_PHASE1, "data/metadata_10.csv"))
    ap.add_argument("--gt_dir", default=os.path.join(_PHASE1, "data/ground_truth"))
    ap.add_argument("--output", default="results/comparison.csv")
    ap.add_argument("--file_ids", nargs="*", default=None)
    ap.add_argument("--max_seconds", type=int, default=None, help="스모크: 앞 N초만")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="int8_float16")
    args = ap.parse_args()

    runner = FasterWhisperRunner(args.model, args.device, args.compute_type)
    meta = pd.read_csv(args.metadata, encoding="utf-8-sig")
    if args.file_ids:
        meta = meta[meta.file_id.isin(args.file_ids)]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_rows = []
    for _, r in meta.iterrows():
        fid = r["file_id"]
        ident = {"file_id": fid, "group": r.get("group", ""), "silence_ratio": r.get("silence_ratio", "")}
        gt_text, gt_segs = load_gt(fid, args.gt_dir)

        audio_path = r["file_path"]
        tmpdir = None
        if args.max_seconds:
            tmpdir = tempfile.mkdtemp(prefix="vmc_trunc_")
            audio_path, gt_segs, dur = truncate(r["file_path"], gt_segs, args.max_seconds, tmpdir)
        else:
            dur = sf.info(audio_path).duration

        sil = get_silence_intervals_from_gt(gt_segs, dur)
        print(f"\n[{fid}] {dur/60:.1f}분 — 참조/Silero/WebRTC 처리 중...")
        try:
            rows = [
                run_reference(runner, audio_path, dur, gt_text, sil),
                run_vad_engine("silero", runner, audio_path, dur, gt_text, gt_segs, sil),
                run_vad_engine("webrtc", runner, audio_path, dur, gt_text, gt_segs, sil),
            ]
            for row in rows:
                all_rows.append({**ident, **row})
            pd.DataFrame(all_rows)[COLUMNS].to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"  [저장] {fid} → {args.output} (누적 {len(all_rows)}행)")
        except Exception as e:
            print(f"  [오류] {fid}: {e}")
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n완료: {args.output} (총 {len(all_rows)}행)")


if __name__ == "__main__":
    main()
