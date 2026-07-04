"""
Seamless-M4T 대량 누락 진단(seamless_누락진단/SEAMLESS_누락원인.md)에서 나온 두 가지 수정을
독립적으로/함께 적용해 개선폭을 분리한다.

  수정 A (디코딩): num_beams=5, repetition_penalty=1.3 (기본은 그리디+반복억제 없음)
  수정 B (VAD, CLIP2만): threshold를 낮춰(0.2) 61~77s·95.6~108.6s 구간을 다시 잡도록 재트리밍

CLIP2는 4개 조건(Stage1 그대로 / A만 / B만 / A+B), CLIP4는 2개 조건(Stage1 그대로 / A만)
— 진단 결과 CLIP4는 VAD 문제가 아니었으므로 B는 적용하지 않음.

Stage1(vad_default_gen_default)은 재실행하지 않고 기존 결과를 그대로 가져와 표에 합친다.
01_STT선정 파일은 seamless_runner.py에 생성 파라미터 오버라이드용 인자만 추가했고(이미
반영됨), run_all_models/evaluate 등은 그대로 재사용한다.

사용법 (01_STT선정/.venv 파이썬 + LD_LIBRARY_PATH):
  LD_LIBRARY_PATH=... .venv/bin/python3 scripts/run_seamless_fix.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STT_PROJECT = HERE.parent / "01_STT선정"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STT_PROJECT))

SEAMLESS_MODEL_ID = "facebook/seamless-m4t-v2-large"
GEN_FIX = {"num_beams": 5, "repetition_penalty": 1.3}
VAD_LOWTHRESH = {"threshold": 0.2}

OUT_DIR = HERE / "results" / "stage_seamless_fix"
TRANSCRIPTS_DIR = OUT_DIR / "transcripts"


def run_condition(cond_name, clip, wav_path, gt_dir, generation_kwargs, runner_cache):
    from experiments.run_all_models import _run_with_rtf
    from evaluation.metrics import evaluate
    from evaluation.hallucination_filter import filter_hallucinations

    key = json.dumps(generation_kwargs or {}, sort_keys=True)
    if key not in runner_cache:
        from pipeline.stt.seamless_runner import SeamlessM4Tv2Runner
        runner_cache[key] = SeamlessM4Tv2Runner(
            model_id=SEAMLESS_MODEL_ID, tgt_lang="kor", device="cuda",
            generation_kwargs=generation_kwargs,
        )
    runner = runner_cache[key]

    gt = (Path(gt_dir) / f"{clip}.txt").read_text(encoding="utf-8").strip()
    result, mean_rtf, std_rtf = _run_with_rtf(runner, str(wav_path), 1, 3)
    raw_text = result.full_text()
    filtered_text = filter_hallucinations(raw_text)
    m = evaluate(filtered_text, gt, segments=result.segments)

    cond_dir_raw = TRANSCRIPTS_DIR / "raw" / cond_name
    cond_dir_filtered = TRANSCRIPTS_DIR / "filtered" / cond_name
    cond_dir_raw.mkdir(parents=True, exist_ok=True)
    cond_dir_filtered.mkdir(parents=True, exist_ok=True)
    (cond_dir_raw / f"{clip}.txt").write_text(raw_text, encoding="utf-8")
    (cond_dir_filtered / f"{clip}.txt").write_text(filtered_text, encoding="utf-8")

    return {
        "condition": cond_name, "clip": clip,
        "cer": round(m.cer, 4), "wer": round(m.wer, 4),
        "ins_rate": round(m.ins_rate, 4), "del_rate": round(m.del_rate, 4),
        "rtf_mean": round(mean_rtf, 4),
        "raw_text": raw_text, "filtered_text": filtered_text,
    }


def load_stage1_baseline(clip):
    """기존 stage1_vad_only 결과에서 seamless_m4t_v2 / clip 행을 가져온다(재실행 안 함)."""
    import csv
    stage1_csv = HERE / "results" / "stage1_vad_only" / "results.csv"
    with open(stage1_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model_key"] == "seamless_m4t_v2" and row["file_id"] == clip:
                raw = (HERE / "results" / "stage1_vad_only" / "transcripts" / "raw"
                       / "seamless_m4t_v2" / f"{clip}.txt").read_text(encoding="utf-8")
                filtered = (HERE / "results" / "stage1_vad_only" / "transcripts" / "filtered"
                            / "seamless_m4t_v2" / f"{clip}.txt").read_text(encoding="utf-8")
                return {
                    "condition": "vad_default_gen_default (Stage1)", "clip": clip,
                    "cer": float(row["cer"]), "wer": float(row["wer"]),
                    "ins_rate": float(row["ins_rate"]), "del_rate": float(row["del_rate"]),
                    "rtf_mean": float(row["rtf_mean"]),
                    "raw_text": raw, "filtered_text": filtered,
                }
    raise RuntimeError(f"Stage1 baseline 행을 못 찾음: seamless_m4t_v2 / {clip}")


def main():
    from vad_pipeline.vad_silero import trim_with_vad

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_dir = HERE / "data" / "ground_truth"
    vad_dir = HERE / "data" / "vad_trimmed"  # 기존(threshold 기본값) 트리밍 결과, 그대로 재사용
    vad_lowthresh_dir = HERE / "data" / "vad_trimmed_lowthresh"
    vad_lowthresh_dir.mkdir(parents=True, exist_ok=True)

    results = []
    runner_cache = {}

    # ---- CLIP2: 4조건 ----
    print("== CLIP2 ==")
    results.append(load_stage1_baseline("CLIP2"))
    print("  [vad_default_gen_default] Stage1 재사용")

    results.append(run_condition(
        "vad_default_gen_fixed", "CLIP2", vad_dir / "CLIP2_vad.wav", gt_dir, GEN_FIX, runner_cache,
    ))
    print("  [vad_default_gen_fixed] 완료")

    print("  CLIP2 낮은 threshold로 재트리밍 중...")
    src_wav = STT_PROJECT / "data" / "raw_v2" / "CLIP2.wav"
    lowthresh_wav = vad_lowthresh_dir / "CLIP2_vad.wav"
    vad_stat = trim_with_vad(str(src_wav), str(lowthresh_wav), **VAD_LOWTHRESH)
    print(f"    {vad_stat}")

    results.append(run_condition(
        "vad_lowthresh_gen_default", "CLIP2", lowthresh_wav, gt_dir, {}, runner_cache,
    ))
    print("  [vad_lowthresh_gen_default] 완료")

    results.append(run_condition(
        "vad_lowthresh_gen_fixed", "CLIP2", lowthresh_wav, gt_dir, GEN_FIX, runner_cache,
    ))
    print("  [vad_lowthresh_gen_fixed] 완료")

    # ---- CLIP4: 2조건 (진단상 VAD 문제 아님 -> B 미적용) ----
    print("== CLIP4 ==")
    results.append(load_stage1_baseline("CLIP4"))
    print("  [vad_default_gen_default] Stage1 재사용")

    results.append(run_condition(
        "vad_default_gen_fixed", "CLIP4", vad_dir / "CLIP4_vad.wav", gt_dir, GEN_FIX, runner_cache,
    ))
    print("  [vad_default_gen_fixed] 완료")

    # ---- 저장 ----
    csv_path = OUT_DIR / "results.csv"
    import csv as csv_mod
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=[
            "clip", "condition", "cer", "wer", "ins_rate", "del_rate", "rtf_mean",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    with open(OUT_DIR / "vad_lowthresh_stat_CLIP2.json", "w", encoding="utf-8") as f:
        json.dump(vad_stat, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {csv_path}")
    for r in results:
        print(f"  {r['clip']:6s} {r['condition']:32s} CER={r['cer']:.4f} WER={r['wer']:.4f}")


if __name__ == "__main__":
    main()
