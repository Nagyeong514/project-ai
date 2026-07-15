"""
모델 단위 GPU 병렬 실험 실행기 (v2, RTX6000 x2).

scripts/run_experiment.py의 파일 루프(run_all_models/evaluate)는 그대로 재사용한다 —
평가 지표·프로세스를 손대지 않고, "어떤 모델이 어느 GPU를 쓰는지"만 병렬화한다.
모델마다 별도 OS 프로세스(multiprocessing.Process)를 새로 띄워 CUDA_VISIBLE_DEVICES를
그 프로세스 안에서만 고정한다 — 프로세스풀 재사용 시 발생할 수 있는
"이전 모델이 이미 CUDA를 초기화해 GPU 배정이 안 바뀌는" 문제를 피하기 위함.

사용법 (반드시 .venv 파이썬으로, LD_LIBRARY_PATH 필수):
  이 venv의 torch는 CUDA 13 계열 nvidia 패키지를 쓰는데, faster-whisper(ctranslate2 백엔드)는
  CUDA 12용 libcublas.so.12/libcudnn.so.9를 요구한다. `pip install nvidia-cublas-cu12
  nvidia-cudnn-cu12`로 받은 뒤 아래처럼 LD_LIBRARY_PATH를 잡아줘야 faster_whisper_* 모델이
  GPU에서 죽지 않는다(실제로 이 문제로 크래시 재현·수정함, 2026-07-01).

  PROJECT=$(pwd)
  CUBLAS_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
  CUDNN_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB" .venv/bin/python3 scripts/run_experiment_parallel.py \
    --metadata data/metadata_v2.csv \
    --config configs/experiment_config_v2.yaml \
    --output results/raw/results_v2.csv \
    --gt-dir data/ground_truth \
    --gpus 0,1

선택 옵션:
  --models seamless_m4t_v2,wav2vec2_xlsr_ko   # 특정 모델만 실행 (기본: deployable=true 전체)
  --dry-run                                    # GPU/모델 로딩 없이 스케줄링만 점검
  --skip-scoring                               # GT 없이 전사만 실행 (CER/WER 등은 빈칸,
                                                #   전사 텍스트는 --transcripts-dir에 저장)
  --denylist-filter                            # evaluate() 전에 evaluation/hallucination_filter.py로
                                                #   하이포시스 정제(환각 상투구·반복구 제거) 후 채점
                                                #   (--skip-scoring과 동시 사용 불가)
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_experiment import RESULT_COLS, load_config, load_ground_truth, load_metadata


# RESULT_COLS 중 GT가 있어야만 계산되는 채점 지표 — skip-scoring 모드에서는 빈칸으로 둔다.
SCORING_ONLY_COLS = [
    "cer", "wer", "substitutions", "deletions", "insertions", "hits",
    "ins_rate", "del_rate", "length_ratio",
    "cer_early", "cer_late", "cer_degradation",
    "cs_wer", "cs_ref_tokens",
]


def _run_one_model(
    model_key, model_cfg, gpu_id, metadata, gt_dir, rtf_cfg, out_path,
    dry_run, skip_scoring, transcripts_dir, denylist_filter,
):
    """서브프로세스 진입점 — model_key 하나만 GPU 하나에 고정해 전체 파일 루프를 돈다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not dry_run:
        import torch  # GPU 핀닝이 실제로 먹혔는지 로그로 남긴다 (배정 gpu_id vs 프로세스가 보는 물리 카드)
        uuids = [torch.cuda.get_device_properties(i).uuid for i in range(torch.cuda.device_count())]
        print(
            f"[{model_key}] CUDA_VISIBLE_DEVICES={gpu_id} -> "
            f"{torch.cuda.device_count()}개 GPU 보임, UUID={uuids}",
            flush=True,
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()

        if dry_run:
            return

        if skip_scoring:
            from pipeline.stt import get_stt
            from experiments.run_all_models import _run_with_rtf

            runner = get_stt(model_cfg)
            warmup = rtf_cfg.get("warmup_runs", 1)
            repeats = rtf_cfg.get("measurement_runs", 3)
            model_transcripts_dir = Path(transcripts_dir) / model_key
            model_transcripts_dir.mkdir(parents=True, exist_ok=True)

            for row in metadata:
                file_id = row["file_id"]
                wav_path = row["wav_path"]
                utype = row["utterance_type"]

                result, mean_rtf, std_rtf = _run_with_rtf(runner, wav_path, warmup, repeats)
                (model_transcripts_dir / f"{file_id}.txt").write_text(
                    result.full_text(), encoding="utf-8"
                )

                row_out = {col: "" for col in SCORING_ONLY_COLS}
                row_out.update({
                    "file_id": file_id,
                    "utterance_type": utype,
                    "model_key": model_key,
                    "model_label": model_cfg.get("label", model_key),
                    "rtf_mean": round(mean_rtf, 4),
                    "rtf_std": round(std_rtf, 4),
                    "rtf_note": "skip_scoring",
                    "deployable": model_cfg.get("deployable", False),
                })
                writer.writerow(row_out)
                f.flush()
            return

        if denylist_filter:
            # run_all_models()/evaluate()는 여전히 그대로 재사용한다 — 다만 evaluate()에 넘기기 전에
            # 하이포시스 텍스트만 필터링한다. run_all_models()는 내부에서 자체적으로 evaluate()를
            # 호출해버려서 끼어들 수 없으므로, 이 모드에서는 get_stt+_run_with_rtf+evaluate를
            # skip-scoring 모드와 동일하게 직접 호출한다(둘 다 v1 파일은 수정하지 않음).
            from pipeline.stt import get_stt
            from experiments.run_all_models import _run_with_rtf
            from evaluation.metrics import evaluate
            from evaluation.hallucination_filter import filter_hallucinations

            runner = get_stt(model_cfg)
            warmup = rtf_cfg.get("warmup_runs", 1)
            repeats = rtf_cfg.get("measurement_runs", 3)

            for row in metadata:
                file_id = row["file_id"]
                wav_path = row["wav_path"]
                utype = row["utterance_type"]
                gt = load_ground_truth(file_id, gt_dir)

                result, mean_rtf, std_rtf = _run_with_rtf(runner, wav_path, warmup, repeats)
                filtered_text = filter_hallucinations(result.full_text())
                m = evaluate(filtered_text, gt, segments=result.segments)

                writer.writerow({
                    "file_id": file_id,
                    "utterance_type": utype,
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
                    "cer_early": round(m.cer_early, 4) if m.cer_early is not None else "",
                    "cer_late": round(m.cer_late, 4) if m.cer_late is not None else "",
                    "cer_degradation": round(m.cer_degradation, 4) if m.cer_degradation is not None else "",
                    "cs_wer": round(m.cs_wer, 4) if m.cs_wer is not None else "",
                    "cs_ref_tokens": m.cs_ref_tokens,
                    "rtf_mean": round(mean_rtf, 4),
                    "rtf_std": round(std_rtf, 4),
                    "rtf_note": "denylist_filtered",
                    "deployable": model_cfg.get("deployable", False),
                })
                f.flush()
            return

        from experiments.run_all_models import run_all_models

        for row in metadata:
            file_id = row["file_id"]
            wav_path = row["wav_path"]
            utype = row["utterance_type"]
            gt = load_ground_truth(file_id, gt_dir)

            results = run_all_models(
                audio_path=wav_path,
                ground_truth=gt,
                model_configs={model_key: model_cfg},
                rtf_cfg=rtf_cfg,
            )
            for res in results:
                writer.writerow({
                    "file_id": file_id,
                    "utterance_type": utype,
                    **{k: res[k] for k in RESULT_COLS if k in res},
                })
                f.flush()


def merge_partial_csvs(partial_paths, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RESULT_COLS)
        writer.writeheader()
        for path in partial_paths:
            if not Path(path).exists():
                continue
            with open(path, encoding="utf-8") as in_f:
                for row in csv.DictReader(in_f):
                    writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata_v2.csv")
    parser.add_argument("--config", default="configs/experiment_config_v2.yaml")
    parser.add_argument("--output", default="results/raw/results_v2.csv")
    parser.add_argument("--gt-dir", default="data/ground_truth")
    parser.add_argument("--models", default=None, help="쉼표 구분 모델 키 (기본: deployable=true 전체)")
    parser.add_argument("--gpus", default="0,1", help="쉼표 구분 GPU 인덱스")
    parser.add_argument("--dry-run", action="store_true", help="GPU/모델 로딩 없이 스케줄링만 점검")
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="GT 없이 전사만 실행 (CER/WER 등 채점 지표는 빈칸, 전사 텍스트는 --transcripts-dir에 저장)",
    )
    parser.add_argument("--transcripts-dir", default="results/raw/transcripts_v2",
                        help="--skip-scoring일 때 전사 텍스트 저장 위치 (모델별 하위 폴더 생성)")
    parser.add_argument(
        "--denylist-filter", action="store_true",
        help="evaluate() 채점 전에 evaluation/hallucination_filter.py로 하이포시스를 정제 "
             "(--skip-scoring과 동시 사용 불가)",
    )
    args = parser.parse_args()

    if args.skip_scoring and args.denylist_filter:
        sys.exit("[오류] --skip-scoring과 --denylist-filter는 같이 쓸 수 없습니다 (전자는 채점 자체를 안 함).")

    cfg = load_config(args.config)
    model_configs = cfg["models"]

    if args.models:
        keys = [k.strip() for k in args.models.split(",")]
        model_configs = {k: v for k, v in model_configs.items() if k in keys}
    else:
        model_configs = {k: v for k, v in model_configs.items() if v.get("deployable", False)}

    if not model_configs:
        sys.exit("[오류] 실행할 모델이 없습니다 (--models 확인).")

    gpus = [g.strip() for g in args.gpus.split(",")]
    metadata = load_metadata(args.metadata)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pending = list(model_configs.items())
    running = {}  # gpu_id -> (Process, model_key)
    partial_paths = []

    mode_tag = "[DRY-RUN]" if args.dry_run else ("[SKIP-SCORING]" if args.skip_scoring else "")
    print(f"모델 {len(model_configs)}개, GPU {len(gpus)}개 {mode_tag}")

    def launch(gpu_id, model_key, model_cfg):
        partial_path = out_path.parent / f"{out_path.stem}_{model_key}.csv"
        partial_paths.append(partial_path)
        p = mp.Process(
            target=_run_one_model,
            args=(
                model_key, model_cfg, gpu_id, metadata, args.gt_dir, cfg.get("rtf", {}),
                partial_path, args.dry_run, args.skip_scoring, args.transcripts_dir,
                args.denylist_filter,
            ),
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

    merge_partial_csvs(partial_paths, out_path)
    print(f"\n병합 완료: {out_path}")


if __name__ == "__main__":
    main()
