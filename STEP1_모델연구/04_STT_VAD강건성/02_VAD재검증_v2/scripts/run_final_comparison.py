"""
4-arm 최종 비교 라운드 — 모델별 "최종 확정 설정"으로 CLIP1~4 재전사.

  Whisper large-v3-turbo : VAD Stage1 트리밍 그대로, 디코딩파라미터/initial_prompt 스킵
                            (Stage2에서 0% 효과 확인, 용어집 미작성으로 이번 라운드 스킵)
  Seamless-M4T-v2-Large   : VAD Stage1 트리밍 그대로(재트리밍 금지 — CLIP2에서 역효과 확인됨),
                            num_beams=5, repetition_penalty=1.3 필수 적용(stage_seamless_fix 결론)
  Wav2Vec2-XLSR-Korean    : 재전사 없음 — stage1_vad_only 필터후 텍스트 그대로 복사,
                            GT만 재계산

CER/WER은 모델별로 3회 반복 측정 후 평균/표준편차(v2 5장 한계 "반복측정 부족" 보완).
GT는 이번 라운드에서 기존 06_VAD재검증_v2/data/ground_truth 그대로 사용(신규 축어대본 없음,
사용자 확인됨).

사용법 (GPU 노드에서, 01_STT선정/.venv 파이썬 + LD_LIBRARY_PATH 필수):
  PROJECT=$(pwd)/../01_STT선정
  CUBLAS_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
  CUDNN_LIB="$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB" "$PROJECT/.venv/bin/python3" scripts/run_final_comparison.py
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STT_PROJECT = HERE.parent / "01_STT선정"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STT_PROJECT))

import yaml  # noqa: E402

CLIPS = ["CLIP1", "CLIP2", "CLIP3", "CLIP4"]
REPEATS = 3

VAD_DIR = HERE / "data" / "vad_trimmed"
GT_DIR = HERE / "data" / "ground_truth"
OUT_DIR = HERE / "results" / "final_comparison"
TRANSCRIPTS_DIR = OUT_DIR / "transcripts"
STAGE1_DIR = HERE / "results" / "stage1_vad_only"

RESULT_COLS = [
    "model_key", "model_label", "clip",
    "cer_mean", "cer_std", "wer_mean", "wer_std",
    "rtf_mean", "rtf_std", "n_repeats",
    "substitutions", "deletions", "insertions", "hits",
]


def _save_texts(model_key, clip, raw_text, filtered_text):
    raw_dir = TRANSCRIPTS_DIR / "raw" / model_key
    filtered_dir = TRANSCRIPTS_DIR / "filtered" / model_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{clip}.txt").write_text(raw_text, encoding="utf-8")
    (filtered_dir / f"{clip}.txt").write_text(filtered_text, encoding="utf-8")


def _save_diff(model_key, clip, gt_raw, filtered_text):
    """jiwer 단어 정렬 기반 word-level diff. 오류분류_잔차분석.md와 같은 계열 표기."""
    import jiwer
    from evaluation.normalizer import normalize

    ref = normalize(gt_raw)
    hyp = normalize(filtered_text)

    lines = [f"# {model_key} / {clip} — GT vs 필터후 전사, jiwer 단어 정렬 기반 diff", ""]
    if not ref.strip():
        lines.append("(GT 비어있음)")
    else:
        out = jiwer.process_words(ref, hyp)
        ref_words = out.references[0]
        hyp_words = out.hypotheses[0]
        for chunk in out.alignments[0]:
            if chunk.type == "equal":
                for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                    lines.append(f"[유지] {w}")
            elif chunk.type == "substitute":
                r = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                h = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                lines.append(f"[치환] GT={r} -> HYP={h}")
            elif chunk.type == "delete":
                r = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                lines.append(f"[삭제] GT={r} (누락)")
            elif chunk.type == "insert":
                h = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                lines.append(f"[삽입] HYP={h} (없는 단어 추가)")

    diff_dir = TRANSCRIPTS_DIR / "diff" / model_key
    diff_dir.mkdir(parents=True, exist_ok=True)
    (diff_dir / f"{clip}.txt").write_text("\n".join(lines), encoding="utf-8")


def _repeat_transcribe(runner, wav_path, gt_raw, repeats=REPEATS):
    from evaluation.hallucination_filter import filter_hallucinations
    from evaluation.metrics import evaluate

    cers, wers, rtfs = [], [], []
    last_raw = last_filtered = None
    last_m = None
    for _ in range(repeats):
        result = runner.transcribe(wav_path)
        raw_text = result.full_text()
        filtered_text = filter_hallucinations(raw_text)
        m = evaluate(filtered_text, gt_raw, segments=result.segments)
        cers.append(m.cer)
        wers.append(m.wer)
        rtfs.append(result.rtf)
        last_raw, last_filtered, last_m = raw_text, filtered_text, m

    return {
        "cer_mean": statistics.mean(cers), "cer_std": statistics.pstdev(cers) if len(cers) > 1 else 0.0,
        "wer_mean": statistics.mean(wers), "wer_std": statistics.pstdev(wers) if len(wers) > 1 else 0.0,
        "rtf_mean": statistics.mean(rtfs), "rtf_std": statistics.pstdev(rtfs) if len(rtfs) > 1 else 0.0,
        "raw_text": last_raw, "filtered_text": last_filtered, "m": last_m,
        "cer_values": cers,
    }


def run_whisper_turbo(cfg, writer):
    from pipeline.stt import get_stt

    model_key = "faster_whisper_large_v3_turbo"
    model_cfg = cfg["models"][model_key]
    print(f"== {model_key} (VAD Stage1, decoding/prompt 스킵) ==")
    runner = get_stt(model_cfg)

    for clip in CLIPS:
        wav_path = str(VAD_DIR / f"{clip}_vad.wav")
        gt_raw = (GT_DIR / f"{clip}.txt").read_text(encoding="utf-8").strip()
        r = _repeat_transcribe(runner, wav_path, gt_raw)
        _save_texts(model_key, clip, r["raw_text"], r["filtered_text"])
        _save_diff(model_key, clip, gt_raw, r["filtered_text"])
        m = r["m"]
        writer.writerow({
            "model_key": model_key, "model_label": model_cfg.get("label", model_key), "clip": clip,
            "cer_mean": round(r["cer_mean"], 4), "cer_std": round(r["cer_std"], 4),
            "wer_mean": round(r["wer_mean"], 4), "wer_std": round(r["wer_std"], 4),
            "rtf_mean": round(r["rtf_mean"], 4), "rtf_std": round(r["rtf_std"], 4),
            "n_repeats": REPEATS,
            "substitutions": m.substitutions, "deletions": m.deletions,
            "insertions": m.insertions, "hits": m.hits,
        })
        print(f"  {clip}: CER={r['cer_mean']:.4f}(±{r['cer_std']:.4f}) RTF={r['rtf_mean']:.4f} "
              f"repeats={r['cer_values']}")


def run_seamless_fixed(writer):
    from pipeline.stt.seamless_runner import SeamlessM4Tv2Runner

    model_key = "seamless_m4t_v2"
    model_label = "Seamless-M4T v2-Large (num_beams=5, repetition_penalty=1.3)"
    print(f"== {model_key} (VAD Stage1 그대로, num_beams=5/repetition_penalty=1.3 필수 적용) ==")
    runner = SeamlessM4Tv2Runner(
        model_id="facebook/seamless-m4t-v2-large", tgt_lang="kor", device="cuda",
        generation_kwargs={"num_beams": 5, "repetition_penalty": 1.3},
    )

    for clip in CLIPS:
        wav_path = str(VAD_DIR / f"{clip}_vad.wav")  # lowthresh 아님 — Stage1 트리밍 그대로
        gt_raw = (GT_DIR / f"{clip}.txt").read_text(encoding="utf-8").strip()
        r = _repeat_transcribe(runner, wav_path, gt_raw)
        _save_texts(model_key, clip, r["raw_text"], r["filtered_text"])
        _save_diff(model_key, clip, gt_raw, r["filtered_text"])
        m = r["m"]
        writer.writerow({
            "model_key": model_key, "model_label": model_label, "clip": clip,
            "cer_mean": round(r["cer_mean"], 4), "cer_std": round(r["cer_std"], 4),
            "wer_mean": round(r["wer_mean"], 4), "wer_std": round(r["wer_std"], 4),
            "rtf_mean": round(r["rtf_mean"], 4), "rtf_std": round(r["rtf_std"], 4),
            "n_repeats": REPEATS,
            "substitutions": m.substitutions, "deletions": m.deletions,
            "insertions": m.insertions, "hits": m.hits,
        })
        print(f"  {clip}: CER={r['cer_mean']:.4f}(±{r['cer_std']:.4f}) RTF={r['rtf_mean']:.4f} "
              f"repeats={r['cer_values']}")


def run_wav2vec2_reuse(cfg, writer):
    from evaluation.metrics import evaluate

    model_key = "wav2vec2_xlsr_ko"
    model_cfg = cfg["models"][model_key]
    print(f"== {model_key} (재전사 없음, v2 원본 필터후 텍스트 재사용 + GT만 재계산) ==")

    for clip in CLIPS:
        gt_raw = (GT_DIR / f"{clip}.txt").read_text(encoding="utf-8").strip()
        raw_text = (STAGE1_DIR / "transcripts" / "raw" / model_key / f"{clip}.txt").read_text(encoding="utf-8")
        filtered_text = (STAGE1_DIR / "transcripts" / "filtered" / model_key / f"{clip}.txt").read_text(encoding="utf-8")

        m = evaluate(filtered_text, gt_raw)
        _save_texts(model_key, clip, raw_text, filtered_text)
        _save_diff(model_key, clip, gt_raw, filtered_text)

        writer.writerow({
            "model_key": model_key, "model_label": model_cfg.get("label", model_key), "clip": clip,
            "cer_mean": round(m.cer, 4), "cer_std": 0.0,
            "wer_mean": round(m.wer, 4), "wer_std": 0.0,
            "rtf_mean": "", "rtf_std": "",
            "n_repeats": 1,
            "substitutions": m.substitutions, "deletions": m.deletions,
            "insertions": m.insertions, "hits": m.hits,
        })
        print(f"  {clip}: CER={m.cer:.4f} (v2 재사용, 재전사 없음)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = STT_PROJECT / "configs" / "experiment_config_v2.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()

        t0 = time.perf_counter()
        run_whisper_turbo(cfg, writer)
        f.flush()
        run_seamless_fixed(writer)
        f.flush()
        run_wav2vec2_reuse(cfg, writer)
        f.flush()
        print(f"\n총 소요: {time.perf_counter() - t0:.1f}s")

    print(f"\n결과 저장: {csv_path}")
    print(f"전사 텍스트: {TRANSCRIPTS_DIR}/raw|filtered|diff/")


if __name__ == "__main__":
    main()
