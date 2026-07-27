"""
VAD 재검증 실험 (v2 4-arm 대상) — CLIP1~4에 Silero VAD 트리밍을 전처리로 붙였을 때
CER/WER이 "필터전용"(01_STT선정 v2, VAD 없음) 대비 어떻게 바뀌는지 비교한다.

파이프라인: VAD 트리밍(pipeline/vad_silero.py) -> 전사(01_STT선정 STT 러너 재사용) ->
환각 필터(01_STT선정 evaluation/hallucination_filter.py 재사용, 그대로) -> 채점(evaluate 재사용).
01_STT선정의 어떤 파일도 수정하지 않고 sys.path로 import만 재사용한다.

사용법 (반드시 01_STT선정/.venv 파이썬 + LD_LIBRARY_PATH, 그 venv에 silero-vad 설치해둠):
  PROJECT=$(pwd)/../01_STT선정
  CUBLAS_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
  CUDNN_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB" "$PROJECT/.venv/bin/python3" scripts/run_vad_experiment.py --gpus 0,1
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STT_PROJECT = HERE.parent / "01_STT선정"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STT_PROJECT))

import yaml  # noqa: E402

MODEL_KEYS = [
    "faster_whisper_large_v3",
    "faster_whisper_large_v3_turbo",
    "seamless_m4t_v2",
    "wav2vec2_xlsr_ko",
]
CLIPS = ["CLIP1", "CLIP2", "CLIP3", "CLIP4"]

RESULT_COLS = [
    "file_id", "model_key", "model_label",
    "cer", "wer", "substitutions", "deletions", "insertions", "hits",
    "ins_rate", "del_rate", "length_ratio", "cs_wer", "cs_ref_tokens",
    "rtf_mean", "rtf_std", "vad_speech_ratio", "vad_n_segments",
]


def run_vad_trim(source_wav_dir: Path, out_dir: Path) -> dict:
    """CLIP1~4를 Silero VAD로 트리밍해 out_dir에 저장. {clip_id: stats} 반환."""
    from vad_pipeline.vad_silero import trim_with_vad

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for clip in CLIPS:
        src = source_wav_dir / f"{clip}.wav"
        dst = out_dir / f"{clip}_vad.wav"
        s = trim_with_vad(str(src), str(dst))
        stats[clip] = s
        print(f"  {clip}: {s['original_duration_s']}s -> {s['trimmed_duration_s']}s "
              f"(발화비율 {s['speech_ratio']*100:.1f}%, 구간 {s['n_segments']}개)")
    return stats


def _run_one_model(model_key, model_cfg, gpu_id, vad_dir, gt_dir, rtf_cfg, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch  # GPU 핀닝 확인 로그
    uuids = [torch.cuda.get_device_properties(i).uuid for i in range(torch.cuda.device_count())]
    print(f"[{model_key}] CUDA_VISIBLE_DEVICES={gpu_id} -> UUID={uuids}", flush=True)

    from pipeline.stt import get_stt
    from experiments.run_all_models import _run_with_rtf
    from evaluation.metrics import evaluate
    from evaluation.hallucination_filter import filter_hallucinations

    runner = get_stt(model_cfg)
    warmup = rtf_cfg.get("warmup_runs", 1)
    repeats = rtf_cfg.get("measurement_runs", 3)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()

        for clip in CLIPS:
            wav_path = str(vad_dir / f"{clip}_vad.wav")
            gt = (Path(gt_dir) / f"{clip}.txt").read_text(encoding="utf-8").strip()

            result, mean_rtf, std_rtf = _run_with_rtf(runner, wav_path, warmup, repeats)
            filtered_text = filter_hallucinations(result.full_text())
            m = evaluate(filtered_text, gt, segments=result.segments)

            writer.writerow({
                "file_id": clip,
                "model_key": model_key,
                "model_label": model_cfg.get("label", model_key),
                "cer": round(m.cer, 4),
                "wer": round(m.wer, 4),
                "substitutions": m.substitutions,
                "deletions": m.deletions,
                "insertions": m.insertions,
                "hits": m.hits,
                "ins_rate": round(m.ins_rate, 4),
                "del_rate": round(m.del_rate, 4),
                "length_ratio": round(m.length_ratio, 4),
                "cs_wer": round(m.cs_wer, 4) if m.cs_wer is not None else "",
                "cs_ref_tokens": m.cs_ref_tokens,
                "rtf_mean": round(mean_rtf, 4),
                "rtf_std": round(std_rtf, 4),
                "vad_speech_ratio": "",  # main 프로세스가 병합 후 채움
                "vad_n_segments": "",
            })
            f.flush()


def merge_and_annotate(partial_paths, vad_stats, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RESULT_COLS)
        writer.writeheader()
        for path in partial_paths:
            if not Path(path).exists():
                continue
            with open(path, encoding="utf-8") as in_f:
                for row in csv.DictReader(in_f):
                    s = vad_stats.get(row["file_id"], {})
                    row["vad_speech_ratio"] = s.get("speech_ratio", "")
                    row["vad_n_segments"] = s.get("n_segments", "")
                    writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--source-wav-dir", default=str(STT_PROJECT / "data" / "raw_v2"),
        help="VAD 트리밍 대상 원본 WAV(16kHz mono) 디렉터리",
    )
    parser.add_argument("--vad-dir", default=str(HERE / "data" / "vad_trimmed"))
    parser.add_argument("--gt-dir", default=str(HERE / "data" / "ground_truth"))
    parser.add_argument("--config", default=str(STT_PROJECT / "configs" / "experiment_config_v2.yaml"))
    parser.add_argument("--output", default=str(HERE / "results" / "vad_filtered_results.csv"))
    args = parser.parse_args()

    print("== 1단계: Silero VAD 트리밍 ==")
    vad_stats = run_vad_trim(Path(args.source_wav_dir), Path(args.vad_dir))
    with open(Path(args.vad_dir).parent / "vad_stats.json", "w", encoding="utf-8") as f:
        json.dump(vad_stats, f, ensure_ascii=False, indent=2)

    print("\n== 2단계: 4개 모델 전사 -> 환각 필터 -> 채점 (GPU 병렬) ==")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_configs = {k: cfg["models"][k] for k in MODEL_KEYS}
    rtf_cfg = cfg.get("rtf", {})
    gpus = [g.strip() for g in args.gpus.split(",")]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vad_dir = Path(args.vad_dir)

    pending = list(model_configs.items())
    running = {}
    partial_paths = []

    def launch(gpu_id, model_key, model_cfg):
        partial_path = out_path.parent / f"{out_path.stem}_{model_key}.csv"
        partial_paths.append(partial_path)
        p = mp.Process(
            target=_run_one_model,
            args=(model_key, model_cfg, gpu_id, vad_dir, args.gt_dir, rtf_cfg, partial_path),
        )
        p.start()
        running[gpu_id] = (p, model_key, time.perf_counter())
        print(f"  배정: {model_key} -> GPU {gpu_id}")

    for gpu_id in gpus:
        if not pending:
            break
        model_key, model_cfg = pending.pop(0)
        launch(gpu_id, model_key, model_cfg)

    while running:
        for gpu_id, (p, model_key, started_at) in list(running.items()):
            if not p.is_alive():
                p.join()
                elapsed = time.perf_counter() - started_at
                print(f"  완료: {model_key} (GPU {gpu_id}, exitcode={p.exitcode}, {elapsed:.1f}s)")
                del running[gpu_id]
                if pending:
                    next_key, next_cfg = pending.pop(0)
                    launch(gpu_id, next_key, next_cfg)
        if running:
            time.sleep(1)

    merge_and_annotate(partial_paths, vad_stats, out_path)
    print(f"\n병합 완료: {out_path}")


if __name__ == "__main__":
    main()
