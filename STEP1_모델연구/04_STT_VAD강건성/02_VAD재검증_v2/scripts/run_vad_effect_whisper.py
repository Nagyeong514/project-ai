"""
Whisper large-v3-turbo(최종 확정 STT) 고정, VAD 효과를 보여주는 3-arm 비교.

  ① no_vad        : VAD 없음, 원본 오디오 그대로 (01_STT선정 v2 필터후 결과 재사용)
  ② vad_default   : VAD 기본값(threshold=0.5) — 오늘 확정한 최종 설정 (stage1_vad_only 재사용)
  ③ vad_lowthresh : VAD threshold=0.2 — 지금까지 미검증이었던 축, Whisper에서만 새로 실행

디코딩파라미터/initial_prompt는 세 조건 모두 동일(스킵) — Stage2에서 0% 효과 확인된 축이라
이 실험에 다시 넣지 않는다. VAD 자체의 순수 효과(①→②)와 VAD 파라미터 민감도(②→③)를
분리해서 보는 게 목적.

사용법 (GPU 노드, 01_STT선정/.venv 파이썬 + LD_LIBRARY_PATH 필수):
  LD_LIBRARY_PATH=... .venv/bin/python3 scripts/run_vad_effect_whisper.py
"""
import csv
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
MODEL_KEY = "faster_whisper_large_v3_turbo"

GT_DIR = HERE / "data" / "ground_truth"
VAD_DEFAULT_DIR = HERE / "data" / "vad_trimmed"
VAD_LOWTHRESH_DIR = HERE / "data" / "vad_trimmed_lowthresh"
NO_VAD_DIR = STT_PROJECT / "data" / "raw_v2"
STAGE1_DIR = HERE / "results" / "stage1_vad_only"
SMOKE_TEST_DIR = STT_PROJECT / ".smoke_test"

OUT_DIR = HERE / "results" / "vad_effect_whisper"
TRANSCRIPTS_DIR = OUT_DIR / "transcripts"

RESULT_COLS = [
    "condition", "clip", "cer_mean", "cer_std", "wer_mean", "wer_std",
    "rtf_mean", "rtf_std", "n_repeats", "substitutions", "deletions", "insertions", "hits",
]


def _save_texts(condition, clip, raw_text, filtered_text):
    raw_dir = TRANSCRIPTS_DIR / "raw" / condition
    filtered_dir = TRANSCRIPTS_DIR / "filtered" / condition
    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{clip}.txt").write_text(raw_text, encoding="utf-8")
    (filtered_dir / f"{clip}.txt").write_text(filtered_text, encoding="utf-8")


def _save_diff(condition, clip, gt_raw, filtered_text):
    import jiwer
    from evaluation.normalizer import normalize

    ref = normalize(gt_raw)
    hyp = normalize(filtered_text)

    lines = [f"# {condition} / {clip} — GT vs 필터후 전사, jiwer 단어 정렬 기반 diff", ""]
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

    diff_dir = TRANSCRIPTS_DIR / "diff" / condition
    diff_dir.mkdir(parents=True, exist_ok=True)
    (diff_dir / f"{clip}.txt").write_text("\n".join(lines), encoding="utf-8")


def trim_lowthresh_missing():
    from vad_pipeline.vad_silero import trim_with_vad

    VAD_LOWTHRESH_DIR.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        dst = VAD_LOWTHRESH_DIR / f"{clip}_vad.wav"
        if dst.exists():
            print(f"  {clip}: lowthresh 트리밍 이미 있음, 재사용")
            continue
        src = NO_VAD_DIR / f"{clip}.wav"
        stat = trim_with_vad(str(src), str(dst), threshold=0.2)
        print(f"  {clip}: lowthresh 트리밍 신규 생성 -> {stat}")


def run_condition3_vad_lowthresh(cfg, writer):
    from evaluation.hallucination_filter import filter_hallucinations
    from evaluation.metrics import evaluate
    from pipeline.stt import get_stt

    model_cfg = cfg["models"][MODEL_KEY]
    print(f"== ③ vad_lowthresh (threshold=0.2, Whisper turbo 최종 설정) ==")
    runner = get_stt(model_cfg)

    for clip in CLIPS:
        wav_path = str(VAD_LOWTHRESH_DIR / f"{clip}_vad.wav")
        gt_raw = (GT_DIR / f"{clip}.txt").read_text(encoding="utf-8").strip()

        cers, wers, rtfs = [], [], []
        last_raw = last_filtered = last_m = None
        for _ in range(REPEATS):
            result = runner.transcribe(wav_path)
            raw_text = result.full_text()
            filtered_text = filter_hallucinations(raw_text)
            m = evaluate(filtered_text, gt_raw, segments=result.segments)
            cers.append(m.cer)
            wers.append(m.wer)
            rtfs.append(result.rtf)
            last_raw, last_filtered, last_m = raw_text, filtered_text, m

        cer_mean, cer_std = statistics.mean(cers), (statistics.pstdev(cers) if len(cers) > 1 else 0.0)
        wer_mean, wer_std = statistics.mean(wers), (statistics.pstdev(wers) if len(wers) > 1 else 0.0)
        rtf_mean, rtf_std = statistics.mean(rtfs), (statistics.pstdev(rtfs) if len(rtfs) > 1 else 0.0)

        _save_texts("vad_lowthresh", clip, last_raw, last_filtered)
        _save_diff("vad_lowthresh", clip, gt_raw, last_filtered)

        writer.writerow({
            "condition": "vad_lowthresh", "clip": clip,
            "cer_mean": round(cer_mean, 4), "cer_std": round(cer_std, 4),
            "wer_mean": round(wer_mean, 4), "wer_std": round(wer_std, 4),
            "rtf_mean": round(rtf_mean, 4), "rtf_std": round(rtf_std, 4),
            "n_repeats": REPEATS,
            "substitutions": last_m.substitutions, "deletions": last_m.deletions,
            "insertions": last_m.insertions, "hits": last_m.hits,
        })
        print(f"  {clip}: CER={cer_mean:.4f}(±{cer_std:.4f}) RTF={rtf_mean:.4f} repeats={cers}")


def reuse_condition1_no_vad(writer):
    from evaluation.hallucination_filter import filter_hallucinations
    from evaluation.metrics import evaluate

    print("== ① no_vad (01_STT선정 v2 필터후 결과 재사용, 재전사 없음) ==")
    for clip in CLIPS:
        gt_raw = (GT_DIR / f"{clip}.txt").read_text(encoding="utf-8").strip()
        raw_text = (SMOKE_TEST_DIR / "all_transcripts" / MODEL_KEY / f"{clip}.txt").read_text(encoding="utf-8")
        filtered_text = filter_hallucinations(raw_text)
        m = evaluate(filtered_text, gt_raw)

        _save_texts("no_vad", clip, raw_text, filtered_text)
        _save_diff("no_vad", clip, gt_raw, filtered_text)

        writer.writerow({
            "condition": "no_vad", "clip": clip,
            "cer_mean": round(m.cer, 4), "cer_std": 0.0,
            "wer_mean": round(m.wer, 4), "wer_std": 0.0,
            "rtf_mean": "", "rtf_std": "",
            "n_repeats": 1,
            "substitutions": m.substitutions, "deletions": m.deletions,
            "insertions": m.insertions, "hits": m.hits,
        })
        print(f"  {clip}: CER={m.cer:.4f} (재사용, 재전사 없음)")


def reuse_condition2_vad_default(writer):
    import csv as csv_mod

    print("== ② vad_default (stage1_vad_only 재사용, 재전사 없음) ==")
    stage1_csv = STAGE1_DIR / "results.csv"
    rows_by_clip = {}
    with open(stage1_csv, encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            if row["model_key"] == MODEL_KEY:
                rows_by_clip[row["file_id"]] = row

    for clip in CLIPS:
        row = rows_by_clip[clip]
        raw_text = (STAGE1_DIR / "transcripts" / "raw" / MODEL_KEY / f"{clip}.txt").read_text(encoding="utf-8")
        filtered_text = (STAGE1_DIR / "transcripts" / "filtered" / MODEL_KEY / f"{clip}.txt").read_text(encoding="utf-8")

        _save_texts("vad_default", clip, raw_text, filtered_text)
        _save_diff("vad_default", clip, (GT_DIR / f"{clip}.txt").read_text(encoding="utf-8").strip(), filtered_text)

        writer.writerow({
            "condition": "vad_default", "clip": clip,
            "cer_mean": row["cer"], "cer_std": 0.0,
            "wer_mean": row["wer"], "wer_std": 0.0,
            "rtf_mean": row["rtf_mean"], "rtf_std": row["rtf_std"],
            "n_repeats": 1,
            "substitutions": row["substitutions"], "deletions": row["deletions"],
            "insertions": row["insertions"], "hits": row["hits"],
        })
        print(f"  {clip}: CER={row['cer']} (재사용, 재전사 없음)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = STT_PROJECT / "configs" / "experiment_config_v2.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("== VAD lowthresh 트리밍 (CLIP1/3/4 신규, CLIP2 재사용) ==")
    trim_lowthresh_missing()

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()

        t0 = time.perf_counter()
        reuse_condition1_no_vad(writer)
        f.flush()
        reuse_condition2_vad_default(writer)
        f.flush()
        run_condition3_vad_lowthresh(cfg, writer)
        f.flush()
        print(f"\n총 소요: {time.perf_counter() - t0:.1f}s")

    print(f"\n결과 저장: {csv_path}")
    print(f"전사 텍스트: {TRANSCRIPTS_DIR}/raw|filtered|diff/")


if __name__ == "__main__":
    main()
