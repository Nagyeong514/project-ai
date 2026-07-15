"""
VAD 기반 4-arm 어블레이션 실험 — 단계별로 조건을 하나씩 얹어가며 CER/WER을 재본다.

  Baseline(01_STT선정 v2, 필터 후)
    -> Stage1 +VAD만
    -> Stage2 +VAD +디코딩파라미터(no_speech_threshold/temperature, faster-whisper만)
    -> Stage3 +VAD +디코딩파라미터 +initial_prompt   (예정)
    -> Stage4 +VAD +디코딩파라미터 +initial_prompt +LLM필터   (예정)

VAD 트리밍은 기본적으로 한 번만 하고(이미 있으면 재사용), 이후 단계는 같은 VAD 오디오에
조건만 바꿔가며 재실행한다. 매 단계마다 (원문 전사, 필터후 전사) 텍스트 파일을 남긴다 —
나중에 보고서 쓸 때 다시 GPU 안 돌려도 되게.

01_STT선정 파일은 어떤 것도 수정하지 않고 sys.path로 import만 재사용한다.

사용법 (01_STT선정/.venv 파이썬 + LD_LIBRARY_PATH 필수):
  PROJECT=$(pwd)/../01_STT선정
  CUBLAS_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
  CUDNN_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

  # Stage1 (+VAD만, 텍스트도 저장):
  LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB" "$PROJECT/.venv/bin/python3" scripts/run_ablation_stage.py \
    --stage-name stage1_vad_only

  # Stage2 (+VAD +디코딩파라미터, faster-whisper만 파라미터 변경):
  LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB" "$PROJECT/.venv/bin/python3" scripts/run_ablation_stage.py \
    --stage-name stage2_decoding --no-speech-threshold 0.8 --temperature 0.0
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


def _build_runner(model_key, model_cfg, decoding_overrides):
    """
    기본은 01_STT선정의 get_stt() 팩토리를 그대로 쓴다(수정 없음).
    faster_whisper 엔진 + decoding_overrides가 있을 때만 FasterWhisperRunner를 직접
    생성해 no_speech_threshold/temperature 같은 디코딩 파라미터를 얹는다 — get_stt()는
    현재 model_id만 넘기게 돼 있어 이 값들이 안 먹기 때문(01_STT선정 원본 동작, 안 건드림).
    """
    engine = model_cfg["engine"]
    if engine == "faster_whisper" and decoding_overrides:
        from pipeline.stt.faster_whisper_runner import FasterWhisperRunner
        return FasterWhisperRunner(model_id=model_cfg["model_id"], decoding_params=decoding_overrides)

    from pipeline.stt import get_stt
    return get_stt(model_cfg)


def _run_one_model(model_key, model_cfg, gpu_id, vad_dir, gt_dir, rtf_cfg, out_path,
                    transcripts_dir, decoding_overrides):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch  # GPU 핀닝 확인 로그
    uuids = [torch.cuda.get_device_properties(i).uuid for i in range(torch.cuda.device_count())]
    print(f"[{model_key}] CUDA_VISIBLE_DEVICES={gpu_id} -> UUID={uuids}", flush=True)

    from experiments.run_all_models import _run_with_rtf
    from evaluation.metrics import evaluate
    from evaluation.hallucination_filter import filter_hallucinations

    runner = _build_runner(model_key, model_cfg, decoding_overrides)
    warmup = rtf_cfg.get("warmup_runs", 1)
    repeats = rtf_cfg.get("measurement_runs", 3)

    raw_dir = Path(transcripts_dir) / "raw" / model_key
    filtered_dir = Path(transcripts_dir) / "filtered" / model_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()

        for clip in CLIPS:
            wav_path = str(vad_dir / f"{clip}_vad.wav")
            gt = (Path(gt_dir) / f"{clip}.txt").read_text(encoding="utf-8").strip()

            result, mean_rtf, std_rtf = _run_with_rtf(runner, wav_path, warmup, repeats)
            raw_text = result.full_text()
            filtered_text = filter_hallucinations(raw_text)
            m = evaluate(filtered_text, gt, segments=result.segments)

            (raw_dir / f"{clip}.txt").write_text(raw_text, encoding="utf-8")
            (filtered_dir / f"{clip}.txt").write_text(filtered_text, encoding="utf-8")

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
    parser.add_argument("--stage-name", required=True, help="결과/전사 파일명에 쓸 단계 이름 (예: stage2_decoding)")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--source-wav-dir", default=str(STT_PROJECT / "data" / "raw_v2"))
    parser.add_argument("--vad-dir", default=str(HERE / "data" / "vad_trimmed"))
    parser.add_argument("--retrim", action="store_true", help="VAD 트리밍을 다시 수행(기본은 기존 파일 재사용)")
    parser.add_argument("--gt-dir", default=str(HERE / "data" / "ground_truth"))
    parser.add_argument("--config", default=str(STT_PROJECT / "configs" / "experiment_config_v2.yaml"))
    parser.add_argument("--no-speech-threshold", type=float, default=None,
                        help="faster_whisper 엔진 모델에만 적용 (seamless/wav2vec2는 대응 파라미터 없음)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="faster_whisper 엔진 모델에만 적용")
    args = parser.parse_args()

    out_dir = HERE / "results" / args.stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.csv"
    transcripts_dir = out_dir / "transcripts"

    vad_dir = Path(args.vad_dir)
    vad_stats_path = vad_dir.parent / "vad_stats.json"
    if args.retrim or not vad_stats_path.exists():
        print("== VAD 트리밍 ==")
        vad_stats = run_vad_trim(Path(args.source_wav_dir), vad_dir)
        with open(vad_stats_path, "w", encoding="utf-8") as f:
            json.dump(vad_stats, f, ensure_ascii=False, indent=2)
    else:
        print(f"== VAD 트리밍 재사용: {vad_dir} (--retrim으로 다시 할 수 있음) ==")
        with open(vad_stats_path, encoding="utf-8") as f:
            vad_stats = json.load(f)

    decoding_overrides = {}
    if args.no_speech_threshold is not None:
        decoding_overrides["no_speech_threshold"] = args.no_speech_threshold
    if args.temperature is not None:
        decoding_overrides["temperature"] = args.temperature
    if decoding_overrides:
        print(f"디코딩 파라미터 오버라이드(faster_whisper만): {decoding_overrides}")

    print(f"\n== [{args.stage_name}] 4개 모델 전사 -> 환각 필터 -> 채점 (GPU 병렬) ==")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_configs = {k: cfg["models"][k] for k in MODEL_KEYS}
    rtf_cfg = cfg.get("rtf", {})
    gpus = [g.strip() for g in args.gpus.split(",")]

    pending = list(model_configs.items())
    running = {}
    partial_paths = []

    def launch(gpu_id, model_key, model_cfg):
        partial_path = out_dir / f"results_{model_key}.csv"
        partial_paths.append(partial_path)
        p = mp.Process(
            target=_run_one_model,
            args=(model_key, model_cfg, gpu_id, vad_dir, args.gt_dir, rtf_cfg, partial_path,
                  transcripts_dir, decoding_overrides),
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
    print(f"\n[{args.stage_name}] 병합 완료: {out_path}")
    print(f"전사 텍스트: {transcripts_dir}/raw/, {transcripts_dir}/filtered/")


if __name__ == "__main__":
    main()
