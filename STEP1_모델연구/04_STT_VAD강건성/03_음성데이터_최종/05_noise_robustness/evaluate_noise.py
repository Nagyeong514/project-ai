# ======================================================================
# [인수인계 헤더] 용도: 노이즈 강건성 실험 평가 — D0~D4 조건별 CER/WER/RTF/핵심키워드 인식률 + 조건별 평균 (evaluate.py 함수 재사용, CPU 가능)
# 입력: results_noise_robustness/stt_outputs/{D0..D4}/, dataset/transcripts/, dataset/keywords/core_keywords.json, results·results_noise_robustness의 timing_log.csv
# 출력: results_noise_robustness/metrics_summary.csv
# 실행 예시: (프로젝트 루트에서) .venv/bin/python scripts/evaluate_noise.py
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
노이즈 강건성 실험 평가 스크립트.

results_noise_robustness/stt_outputs/{D0..D4}/clip{1..4}.txt 를 기존 evaluate.py와
동일한 방식(동일 정규화·동일 계산식·동일 휴리스틱)으로 평가한다.
지표: CER, WER, RTF, 작업 핵심 키워드 인식률.
정답 대본은 기존 transcripts/clip{n}_gt.txt 를 그대로 재사용한다.

D0의 RTF는 기존 results/metrics/timing_log.csv(faster_whisper_prompt)를,
D1~D4의 RTF는 results_noise_robustness/timing_log.csv 를 사용한다.

산출물: results_noise_robustness/metrics_summary.csv
"""

import csv
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import normalize_for_cer_wer, load_gt, score_keywords  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
DATASET = PROJ / "dataset"
NR = PROJ / "results_noise_robustness"
STT_OUT = NR / "stt_outputs"

CONDS = ["D0", "D1", "D2", "D3", "D4"]
CLIPS = ["clip1", "clip2", "clip3", "clip4"]

COND_DESC = {
    "D0": "노이즈 없음 (기존 결과 재사용)",
    "D1": "화이트 노이즈 SNR 20dB",
    "D2": "화이트 노이즈 SNR 10dB",
    "D3": "화이트 노이즈 SNR 0dB",
    "D4": "ESC-50 기계음 혼합 SNR 10dB",
}


def load_rtf() -> dict:
    """(condition, clip_id) -> rtf"""
    rtf = {}
    # D0: 기존 실험의 faster_whisper_prompt RTF 재사용
    with open(PROJ / "results" / "metrics" / "timing_log.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == "faster_whisper_prompt":
                rtf[("D0", row["clip_id"])] = float(row["rtf"])
    # D1~D4: 이번 실험 타이밍 로그
    noise_log = NR / "timing_log.csv"
    if noise_log.exists():
        with open(noise_log, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rtf[(row["condition"], row["clip_id"])] = float(row["rtf"])
    return rtf


def main():
    core_keywords = json.loads((DATASET / "keywords" / "core_keywords.json").read_text(encoding="utf-8"))
    rtf_map = load_rtf()

    rows = []
    for cond in CONDS:
        for clip in CLIPS:
            path = STT_OUT / cond / f"{clip}.txt"
            if not path.exists():
                print(f"[skip] {cond}/{clip}.txt 없음")
                continue
            stt_raw = path.read_text(encoding="utf-8").strip()
            gt_raw = load_gt(clip)

            ref = normalize_for_cer_wer(gt_raw)
            hyp = normalize_for_cer_wer(stt_raw)
            cer = jiwer.cer(ref, hyp)
            wer = jiwer.wer(ref, hyp)

            kw_retained, kw_total, _detail = score_keywords(stt_raw, core_keywords.get(clip, []))
            keyword_rate = kw_retained / kw_total if kw_total else None

            rtf = rtf_map.get((cond, clip))
            rows.append({
                "condition": cond, "condition_desc": COND_DESC[cond], "clip_id": clip,
                "cer": f"{cer:.4f}", "wer": f"{wer:.4f}",
                "rtf": f"{rtf:.4f}" if rtf is not None else "",
                "keyword_rate": f"{keyword_rate:.4f}" if keyword_rate is not None else "",
            })

    # 조건별 4클립 평균 행 추가
    for cond in CONDS:
        cond_rows = [r for r in rows if r["condition"] == cond and r["clip_id"] in CLIPS]
        if len(cond_rows) != len(CLIPS):
            continue
        avg = {
            "condition": cond, "condition_desc": COND_DESC[cond], "clip_id": "평균",
            "cer": f"{sum(float(r['cer']) for r in cond_rows) / len(cond_rows):.4f}",
            "wer": f"{sum(float(r['wer']) for r in cond_rows) / len(cond_rows):.4f}",
            "rtf": f"{sum(float(r['rtf']) for r in cond_rows) / len(cond_rows):.4f}"
                   if all(r["rtf"] for r in cond_rows) else "",
            "keyword_rate": f"{sum(float(r['keyword_rate']) for r in cond_rows) / len(cond_rows):.4f}"
                            if all(r["keyword_rate"] for r in cond_rows) else "",
        }
        rows.append(avg)

    out = NR / "metrics_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "condition_desc", "clip_id", "cer", "wer", "rtf", "keyword_rate"])
        writer.writeheader()
        writer.writerows(rows)
    print("평가 완료:", out)


if __name__ == "__main__":
    main()
