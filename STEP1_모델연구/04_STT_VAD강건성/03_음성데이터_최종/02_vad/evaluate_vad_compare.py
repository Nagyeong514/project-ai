# ======================================================================
# [인수인계 헤더] 용도: VAD 방식 비교(A 무VAD / B Silero / C WebRTC) 평가 — 10개 지표 가중 점수 산출 (evaluate.py 함수 재사용, CPU 가능)
# 입력: results_vad_compare/stt_outputs/{A_no_vad,B_silero,C_webrtc}/, results·results_novad_backup의 timing_log.csv, dataset/audio_vad·audio_webrtc의 trim/timing 로그
# 출력: results_vad_compare/metrics/vad_metrics_summary.csv, qualitative_review.csv, final_vad_report.md
# 실행 예시: (프로젝트 루트에서) .venv/bin/python scripts/evaluate_vad_compare.py
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
VAD 방식 비교 실험(A: VAD 미사용 / B: Silero VAD / C: WebRTC VAD) 평가 스크립트.

CER/WER/삽입오류/핵심키워드/word_alignment 계산은 evaluate.py의 기존 함수를 그대로
재사용한다(계획서 6번: "기존 계산 방식과 동일한 방법으로 계산할 것 — 새로운 방식을
발명하지 말 것").

근사치 한계(계획서 6, 12번): 음성 구간의 초 단위 정답 타임스탬프가 없어 발화 보존률/
과분할 정도는 텍스트(단어/문장 수) 기반 근사치다. 실제 타임스탬프 기반 정밀 지표가
아니다.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_gt, normalize_for_cer_wer, score_keywords, word_alignment  # noqa: E402

import jiwer

PROJ = Path(__file__).resolve().parent.parent
DATASET = PROJ / "dataset"
RESULTS = PROJ / "results_vad_compare"
CLIPS = ["clip1", "clip2", "clip3", "clip4"]
CONDITIONS = ["A_no_vad", "B_silero", "C_webrtc"]

WEIGHTS = {
    "cer": 15, "wer": 10, "insertion": 10, "keyword": 10,
    "speech_recall": 10, "silence_removal": 5, "oversegmentation": 5,
    "total_rtf": 10, "vad_rtf": 5, "compression": 5,
}
# 방향: True=낮을수록 좋음, False=높을수록 좋음
LOWER_IS_BETTER = {
    "cer": True, "wer": True, "insertion": True, "keyword": False,
    "speech_recall": False, "silence_removal": False, "oversegmentation": True,
    "total_rtf": True, "vad_rtf": True, "compression": False,
}


def load_core_keywords():
    return __import__("json").loads((DATASET / "keywords" / "core_keywords.json").read_text(encoding="utf-8"))


def gt_sentence_count(clip: str) -> int:
    path = DATASET / "transcripts" / f"{clip}_gt.txt"
    return len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])


def load_csv_dict(path: Path, key_field: str):
    with open(path, encoding="utf-8") as f:
        return {row[key_field]: row for row in csv.DictReader(f)}


def load_a_data():
    """A(원본, 무 VAD) 근원 데이터: STT는 results_novad_backup, VAD 전처리 없음."""
    timing = load_csv_dict(PROJ / "results_novad_backup" / "metrics" / "timing_log.csv", "clip_id")
    # timing_log.csv에는 model 컬럼도 있어 clip_id가 중복 키가 됨 -> model로 재필터링
    rows = list(csv.DictReader(open(PROJ / "results_novad_backup" / "metrics" / "timing_log.csv", encoding="utf-8")))
    rows = [r for r in rows if r["model"] == "faster_whisper_prompt"]
    timing = {r["clip_id"]: r for r in rows}

    data = {}
    for clip in CLIPS:
        t = timing[clip]
        orig_sec = float(t["audio_sec"])
        data[clip] = {
            "stt_text": (RESULTS / "stt_outputs" / "A_no_vad" / f"{clip}.txt").read_text(encoding="utf-8").strip(),
            "original_sec": orig_sec,
            "trimmed_sec": orig_sec,     # VAD 미적용 -> 트리밍 없음
            "segment_count": 1,          # 관례: VAD 분할이 없으므로 전체를 1개 세그먼트로 취급 (근사, 보고서에 명시)
            "vad_infer_sec": 0.0,
            "stt_infer_sec": float(t["infer_sec"]),
        }
    return data


def load_b_data():
    """B(Silero VAD) 근원 데이터: STT는 results/, 트리밍 로그는 dataset/audio_vad/."""
    rows = list(csv.DictReader(open(PROJ / "results" / "metrics" / "timing_log.csv", encoding="utf-8")))
    rows = [r for r in rows if r["model"] == "faster_whisper_prompt"]
    stt_timing = {r["clip_id"]: r for r in rows}
    trim = load_csv_dict(DATASET / "audio_vad" / "vad_trim_log.csv", "clip")
    vad_timing = load_csv_dict(DATASET / "audio_vad" / "vad_timing_log.csv", "clip")

    data = {}
    for clip in CLIPS:
        data[clip] = {
            "stt_text": (RESULTS / "stt_outputs" / "B_silero" / f"{clip}.txt").read_text(encoding="utf-8").strip(),
            "original_sec": float(trim[clip]["orig_sec"]),
            "trimmed_sec": float(trim[clip]["speech_sec"]),
            "segment_count": int(trim[clip]["n_segments"]),
            "vad_infer_sec": float(vad_timing[clip]["vad_infer_sec"]),
            "stt_infer_sec": float(stt_timing[clip]["infer_sec"]),
        }
    return data


def load_c_data():
    """C(WebRTC VAD) 근원 데이터: 이번에 새로 생성/실행."""
    stt_timing = load_csv_dict(RESULTS / "metrics" / "stt_timing_C_webrtc.csv", "clip_id")
    trim = load_csv_dict(DATASET / "audio_webrtc" / "webrtc_trim_log.csv", "clip_id")
    vad_timing = load_csv_dict(DATASET / "audio_webrtc" / "webrtc_timing_log.csv", "clip")

    data = {}
    for clip in CLIPS:
        data[clip] = {
            "stt_text": (RESULTS / "stt_outputs" / "C_webrtc" / f"{clip}.txt").read_text(encoding="utf-8").strip(),
            "original_sec": float(trim[clip]["original_sec"]),
            "trimmed_sec": float(trim[clip]["trimmed_sec"]),
            "segment_count": int(trim[clip]["segment_count"]),
            "vad_infer_sec": float(vad_timing[clip]["vad_infer_sec"]),
            "stt_infer_sec": float(stt_timing[clip]["infer_sec"]),
        }
    return data


def compute_raw_metrics(clip: str, d: dict, core_keywords: dict) -> dict:
    gt_raw = load_gt(clip)
    ref = normalize_for_cer_wer(gt_raw)
    hyp = normalize_for_cer_wer(d["stt_text"])

    cer = jiwer.cer(ref, hyp)
    wer = jiwer.wer(ref, hyp)
    _subs, _dels, _ins, n_subs, n_dels, n_ins = word_alignment(ref, hyp)

    kw_retained, kw_total, _detail = score_keywords(d["stt_text"], core_keywords.get(clip, []))
    keyword_rate = (kw_retained / kw_total) if kw_total else None

    ref_word_count = len(ref.split())
    speech_recall = 1 - (n_dels / ref_word_count) if ref_word_count else None

    orig, trimmed = d["original_sec"], d["trimmed_sec"]
    silence_removal = (orig - trimmed) / orig if orig else 0.0
    compression = 1 - (trimmed / orig) if orig else 0.0

    n_sent = gt_sentence_count(clip)
    oversegmentation = abs(d["segment_count"] - n_sent) / n_sent if n_sent else None

    total_rtf = (d["vad_infer_sec"] + d["stt_infer_sec"]) / orig if orig else None
    vad_rtf = d["vad_infer_sec"] / orig if orig else None

    return {
        "cer": cer, "wer": wer, "insertion": n_ins, "keyword": keyword_rate,
        "speech_recall": speech_recall, "silence_removal": silence_removal,
        "oversegmentation": oversegmentation, "total_rtf": total_rtf,
        "vad_rtf": vad_rtf, "compression": compression,
        "_n_dels": n_dels, "_n_subs": n_subs, "_n_ins": n_ins,
    }


def normalize_score(values: dict, lower_is_better: bool) -> dict:
    """values: {condition: raw_value} -> {condition: 0~100점}"""
    vals = list(values.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {c: 100.0 for c in values}
    scores = {}
    for c, v in values.items():
        if lower_is_better:
            scores[c] = 100 * (vmax - v) / (vmax - vmin)
        else:
            scores[c] = 100 * (v - vmin) / (vmax - vmin)
    return scores


def main():
    core_keywords = load_core_keywords()
    all_data = {"A_no_vad": load_a_data(), "B_silero": load_b_data(), "C_webrtc": load_c_data()}

    raw = {}  # raw[clip][cond] = {metric: value}
    for clip in CLIPS:
        raw[clip] = {}
        for cond in CONDITIONS:
            raw[clip][cond] = compute_raw_metrics(clip, all_data[cond][clip], core_keywords)

    metric_keys = list(WEIGHTS.keys())

    # 클립별 정규화 점수: norm[clip][cond][metric] = 0~100
    norm = {clip: {cond: {} for cond in CONDITIONS} for clip in CLIPS}
    for clip in CLIPS:
        for metric in metric_keys:
            values = {cond: raw[clip][cond][metric] for cond in CONDITIONS if raw[clip][cond][metric] is not None}
            if not values:
                for cond in CONDITIONS:
                    norm[clip][cond][metric] = None
                continue
            scored = normalize_score(values, LOWER_IS_BETTER[metric])
            for cond in CONDITIONS:
                norm[clip][cond][metric] = scored.get(cond)

    # metrics_summary.csv: 클립×조건별 원지표 + 정규화 점수
    summary_rows = []
    for clip in CLIPS:
        for cond in CONDITIONS:
            row = {"clip_id": clip, "condition": cond}
            for metric in metric_keys:
                row[f"{metric}_raw"] = raw[clip][cond][metric]
                row[f"{metric}_score"] = norm[clip][cond][metric]
            summary_rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip_id", "condition"] + [f"{m}_raw" for m in metric_keys] + [f"{m}_score" for m in metric_keys]
    with open(RESULTS / "metrics" / "vad_metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    # 조건별 평균 정규화 점수(4클립 평균) -> 가중합 (80점 만점)
    cond_scores = {}
    cond_avg_metric = {}
    for cond in CONDITIONS:
        avg_metric = {}
        for metric in metric_keys:
            vals = [norm[clip][cond][metric] for clip in CLIPS if norm[clip][cond][metric] is not None]
            avg_metric[metric] = sum(vals) / len(vals) if vals else 0.0
        cond_avg_metric[cond] = avg_metric
        total = sum(avg_metric[m] * WEIGHTS[m] / 100 for m in metric_keys)
        cond_scores[cond] = total

    # qualitative_review.csv (빈 칸)
    qual_rows = []
    for cond in CONDITIONS:
        for clip in CLIPS:
            qual_rows.append({
                "condition": cond, "clip_id": clip,
                "stt_text": all_data[cond][clip]["stt_text"],
                "문장_자연스러움": "", "작업_맥락_유지": "",
                "핵심_정보_누락_여부": "", "한영_혼용_용어_보존": "", "LLM_입력_적합성": "",
            })
    with open(RESULTS / "qualitative_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "clip_id", "stt_text",
                                           "문장_자연스러움", "작업_맥락_유지", "핵심_정보_누락_여부",
                                           "한영_혼용_용어_보존", "LLM_입력_적합성"])
        w.writeheader()
        w.writerows(qual_rows)

    write_final_report(cond_scores, cond_avg_metric, raw)
    print("완료:", RESULTS / "metrics" / "vad_metrics_summary.csv")


PRIORITY_ORDER = ["speech_recall", "insertion", "cer", "wer", "keyword", "total_rtf"]
PRIORITY_LABEL = {
    "speech_recall": "발화 보존률", "insertion": "삽입 오류(할루시네이션)", "cer": "CER", "wer": "WER",
    "keyword": "핵심 키워드 인식률", "total_rtf": "전체 RTF",
}
COND_LABEL = {"A_no_vad": "A(VAD 미사용)", "B_silero": "B(Silero VAD)", "C_webrtc": "C(WebRTC VAD)"}


def break_tie(tied_conds: list, cond_avg_metric: dict) -> list:
    order = list(tied_conds)
    for metric in PRIORITY_ORDER:
        lower_better = LOWER_IS_BETTER[metric]
        order.sort(key=lambda c: cond_avg_metric[c][metric], reverse=not lower_better)
    return order


def write_final_report(cond_scores: dict, cond_avg_metric: dict, raw: dict) -> None:
    ranking = sorted(CONDITIONS, key=lambda c: -cond_scores[c])
    # 동점 처리(부동소수 오차 감안 소수 4자리 반올림 기준)
    grouped = {}
    for c in ranking:
        grouped.setdefault(round(cond_scores[c], 4), []).append(c)
    final_order = []
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        final_order.extend(group if len(group) == 1 else break_tie(group, cond_avg_metric))

    lines = ["# VAD 방식 비교 최종 리포트 (A: 미사용 / B: Silero / C: WebRTC)", ""]
    lines.append(
        "> **근사치 한계**: 본 실험에는 음성 구간의 초 단위 정답 타임스탬프가 없어, "
        "발화 보존률과 과분할 정도는 정답 대본의 단어/문장 수 기반 텍스트 근사치로 계산했다. "
        "실제 타임스탬프 기반 정밀 Speech Recall/Over-segmentation 지표가 아니라는 한계가 있다."
    )
    lines.append(
        "> **A(VAD 미사용) 세그먼트 수 관례**: A는 VAD 분할이 없으므로 과분할 정도 계산을 위해 "
        "전체 클립을 세그먼트 1개로 취급했다(근사 관례, 실제 VAD 세그먼트 개념이 A에는 적용되지 않음)."
    )
    lines.append(
        "> **지표 중복 참고**: 5-2 '무음·소음 제거율'과 5-3 '입력 오디오 압축률'은 계획서에 "
        "명시된 계산식이 수학적으로 동일하다((원본-트리밍)/원본). 계획서에 명시된 배점 구조를 "
        "그대로 따라 각각 5점씩 별도 반영했다."
    )
    lines.append(
        "> STT 디코딩 파라미터(compute_type=float16, language=ko, initial_prompt=도메인 용어, "
        "device=cuda)는 A/B/C 3개 조건 모두 동일하게 고정했다. A/B는 기존 실행 결과를 재사용했고 "
        "(경로: results_novad_backup/, results/), C(WebRTC VAD)만 이번에 새로 오디오 트리밍과 "
        "GPU STT 실행을 수행했다."
    )
    lines.append("")

    lines.append("## 조건별 정량 점수 랭킹 (80점 만점, 4클립 평균)")
    lines.append("")
    lines.append("| 순위 | 조건 | 정량 총점(80) |")
    lines.append("|---|---|---|")
    for i, cond in enumerate(final_order, 1):
        lines.append(f"| {i} | {COND_LABEL[cond]} | {cond_scores[cond]:.2f} |")
    lines.append("")

    lines.append("## 세부 지표 평균 (0~100 정규화 점수, 4클립 평균)")
    lines.append("")
    header = "| 지표(배점) | " + " | ".join(COND_LABEL[c] for c in CONDITIONS) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(CONDITIONS))
    for metric in WEIGHTS:
        label = f"{metric}({WEIGHTS[metric]})"
        row = f"| {label} | " + " | ".join(f"{cond_avg_metric[c][metric]:.1f}" for c in CONDITIONS) + " |"
        lines.append(row)
    lines.append("")

    lines.append("## 세부 원지표 평균 (4클립 평균, 참고용)")
    lines.append("")
    lines.append("| 조건 | CER | WER | 삽입오류 | 키워드인식률 | 발화보존률 | 무음제거율 | 과분할정도 | 전체RTF | VAD처리RTF |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        def avg(m):
            vals = [raw[clip][cond][m] for clip in CLIPS if raw[clip][cond][m] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        lines.append(
            f"| {COND_LABEL[cond]} | {avg('cer'):.4f} | {avg('wer'):.4f} | {avg('insertion'):.1f} | "
            f"{avg('keyword'):.4f} | {avg('speech_recall'):.4f} | {avg('silence_removal'):.4f} | "
            f"{avg('oversegmentation'):.4f} | {avg('total_rtf'):.4f} | {avg('vad_rtf'):.6f} |"
        )
    lines.append("")

    winner = final_order[0]
    lines.append(f"## 정량 기준(80점) 선정: **{COND_LABEL[winner]}**")
    lines.append("")
    lines.append(f"- 정량 총점 {cond_scores[winner]:.2f}/80으로 최고.")
    if len(final_order) > 1:
        runner = final_order[1]
        lines.append(f"- 2위 {COND_LABEL[runner]}({cond_scores[runner]:.2f})와의 차이: {cond_scores[winner]-cond_scores[runner]:.2f}점.")
    lines.append(
        "- 최종 선정 우선순위(계획서 10번): 1)발화 보존률 2)삽입 오류(할루시네이션) 3)CER/WER "
        "4)핵심 용어 보존 5)RTF. 동점 발생 시 이 순서로 비교해 상위 조건을 우선했다."
    )
    lines.append(
        "- 정성 평가(20점, results_vad_compare/qualitative_review.csv)는 사람이 직접 채점 후 "
        "100점 기준으로 재확인 필요(계획서 12번)."
    )
    lines.append("")
    lines.append("## 한계 및 주의사항")
    lines.append(
        "- 발화 보존률/과분할 정도는 초 단위 타임스탬프 정답이 없어 텍스트(단어/문장 수) 기반 "
        "근사치이며, 실제 VAD 정밀도(Speech Recall/False Alarm) 지표가 아니다."
    )
    lines.append(
        "- WebRTC VAD(C)는 Silero(B)와 병합 규칙(min_silence_duration_ms=500, "
        "speech_pad_ms=300)을 동일하게 맞췄으나, 프레임 단위 에너지 기반 판정이라 신경망 기반 "
        "Silero보다 훨씬 많은 세그먼트(과분할)와 더 많은 잔존 구간을 남기는 경향이 관찰되었다 "
        "(세부 수치는 위 표 참고)."
    )

    (RESULTS / "final_vad_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
