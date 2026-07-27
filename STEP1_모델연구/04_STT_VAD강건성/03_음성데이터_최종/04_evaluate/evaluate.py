# ======================================================================
# [인수인계 헤더] 용도: STT 모델 비교 평가 — CER/WER/한영혼용용어/핵심키워드/RTF 가중합 점수와 최종 리포트 생성 (CPU 가능)
#      ※ evaluate_noise.py, evaluate_vad_compare.py가 이 파일의 함수를 import하므로 원본 위치(scripts/)에서 삭제·이동 금지
# 입력: results/stt_outputs/, dataset/transcripts/clip{n}_gt.txt, dataset/keywords/*.json, results/metrics/timing_log.csv
# 출력: results/metrics/metrics_summary.csv, results/qualitative_review.csv, results/error_analysis.csv, results/final_report.md
# 실행 예시: (프로젝트 루트에서) .venv/bin/python scripts/evaluate.py
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
STT 모델 비교 평가 스크립트.

results/stt_outputs/{model}/clip{n}.txt 를 읽어 CER/WER/한영혼용용어 인식률/
작업 핵심 키워드 인식률/RTF 를 계산하고, 계획서(STT_연구계획서_CLI.md)의 가중합
공식으로 클립별·모델별 최종 점수를 산출한다.

산출물:
  results/metrics/metrics_summary.csv
  results/qualitative_review.csv
  results/error_analysis.csv
  results/final_report.md

주의(휴리스틱 한계, final_report.md에도 명시):
  - "작업 핵심 키워드 인식률"과 "순서바뀜"은 완전한 의미 이해가 아니라 토큰 등장
    여부/순서 기반 근사치다. 최종 확정은 qualitative_review.csv 수동 채점을 참고할 것.
  - 한영 혼용 용어의 "음차 표기(0.5점)" 판정은 사전 정의된 표기 변형 목록에 의존한다.
"""

import csv
import json
import re
from pathlib import Path

import jiwer

PROJ = Path(__file__).resolve().parent.parent
DATASET = PROJ / "dataset"
RESULTS = PROJ / "results"
STT_OUT = RESULTS / "stt_outputs"
METRICS_DIR = RESULTS / "metrics"
TIMING_LOG = METRICS_DIR / "timing_log.csv"

CLIPS = ["clip1", "clip2", "clip3", "clip4"]

MODELS = [
    "faster_whisper_noprompt",
    "faster_whisper_prompt",
    "speechbrain",
    "owsm",
]

WEIGHTS = {"cer": 0.30, "mixed_term": 0.25, "keyword": 0.20, "rtf": 0.15, "wer": 0.10}

# 한영 혼용 용어의 음차/표기 변형 목록 (0.5점 처리 기준). 정확 일치(대소문자 무시)는 1점.
MIXED_TERM_VARIANTS = {
    "Motherboard": ["마더보드", "마더 보드"],
    "slot": ["슬롯"],
    "GPU": ["지피유", "지피 유", "쥐피유"],
    "RAM": ["램"],
    "BIOS": ["바이오스", "바이어스", "바이오스가", "바이오스는"],
}

# 작업 핵심 키워드 토큰 매칭 임계값: 키워드를 구성하는 토큰 중 이 비율 이상이
# STT 텍스트에 등장하면 "의미상 유지"로 판정한다.
KEYWORD_TOKEN_MATCH_THRESHOLD = 0.5

PUNCT_RE = re.compile(r"[.,!?]")


def normalize_for_cer_wer(text: str) -> str:
    text = PUNCT_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_gt(clip: str) -> str:
    path = DATASET / "transcripts" / f"{clip}_gt.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    return " ".join(line.strip() for line in lines if line.strip())


def load_stt(model: str, clip: str) -> str | None:
    path = STT_OUT / model / f"{clip}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def load_timing() -> dict:
    timing = {}
    if not TIMING_LOG.exists():
        return timing
    with open(TIMING_LOG, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timing[(row["model"], row["clip_id"])] = float(row["rtf"])
    return timing


def score_mixed_terms(stt_text: str, terms: list) -> tuple:
    """returns (score_sum, term_count, detail_list[(term, score)])"""
    stripped = strip_spaces(stt_text)
    detail = []
    score_sum = 0.0
    for term in terms:
        if term.lower() in stripped.lower():
            score_sum += 1.0
            detail.append((term, 1.0))
            continue
        variants = MIXED_TERM_VARIANTS.get(term, [])
        if any(strip_spaces(v) in stripped for v in variants):
            score_sum += 0.5
            detail.append((term, 0.5))
        else:
            detail.append((term, 0.0))
    return score_sum, len(terms), detail


def score_keywords(stt_text: str, keywords: list) -> tuple:
    """returns (retained_count, total_count, detail_list[(keyword, retained_bool, matched_tokens, order_swapped)])"""
    stripped = strip_spaces(stt_text)
    detail = []
    retained_count = 0
    for kw in keywords:
        tokens = kw.split()
        positions = [stripped.find(strip_spaces(tok)) for tok in tokens]
        found = [p >= 0 for p in positions]
        match_ratio = sum(found) / len(tokens) if tokens else 0.0
        retained = match_ratio >= KEYWORD_TOKEN_MATCH_THRESHOLD
        order_swapped = False
        found_positions = [p for p, ok in zip(positions, found) if ok]
        if len(found_positions) >= 2 and found_positions != sorted(found_positions):
            order_swapped = True
        if retained:
            retained_count += 1
        detail.append((kw, retained, sum(found), order_swapped))
    return retained_count, len(keywords), detail


def word_alignment(reference: str, hypothesis: str):
    out = jiwer.process_words(reference, hypothesis)
    subs, dels, ins = [], [], []
    ref_tokens = out.references[0]
    hyp_tokens = out.hypotheses[0]
    for chunk in out.alignments[0]:
        if chunk.type == "substitute":
            subs.append((
                " ".join(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx]),
                " ".join(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]),
            ))
        elif chunk.type == "delete":
            dels.append(" ".join(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx]))
        elif chunk.type == "insert":
            ins.append(" ".join(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]))
    return subs, dels, ins, out.substitutions, out.deletions, out.insertions


def main():
    mixed_terms = json.loads((DATASET / "keywords" / "mixed_terms.json").read_text(encoding="utf-8"))
    core_keywords = json.loads((DATASET / "keywords" / "core_keywords.json").read_text(encoding="utf-8"))
    timing = load_timing()

    metrics_rows = []
    qual_rows = []
    error_rows = []

    for model in MODELS:
        for clip in CLIPS:
            stt_raw = load_stt(model, clip)
            if stt_raw is None:
                print(f"[skip] {model}/{clip}.txt 없음")
                qual_rows.append({
                    "clip_id": clip, "model": model,
                    "stt_text": "", "gt_text": "",
                    "오류_사례": "미실시 (시간 제약으로 실행 안 함, STT_연구계획서_CLI.md 5-2절 참고)",
                    "문장_자연스러움": "", "작업_맥락_유지": "",
                    "핵심_정보_누락_여부": "", "한영_혼용_의미_보존": "", "LLM_입력_적합성": "",
                })
                continue
            gt_raw = load_gt(clip)

            ref = normalize_for_cer_wer(gt_raw)
            hyp = normalize_for_cer_wer(stt_raw)

            cer = jiwer.cer(ref, hyp)
            wer = jiwer.wer(ref, hyp)

            mt_score, mt_total, mt_detail = score_mixed_terms(stt_raw, mixed_terms.get(clip, []))
            mixed_term_rate = (mt_score / mt_total) if mt_total > 0 else None

            kw_retained, kw_total, kw_detail = score_keywords(stt_raw, core_keywords.get(clip, []))
            keyword_rate = (kw_retained / kw_total) if kw_total > 0 else None

            rtf = timing.get((model, clip))
            rtf_score = None
            if rtf is not None:
                rtf_score = (1 - rtf) if rtf < 1.0 else 0.0

            # 최종 가중합: 혼용용어 지표가 없는 클립(예: clip3)은 해당 항목을 빼고
            # 남은 가중치를 합이 1이 되도록 재정규화한다.
            components = {"cer": 1 - cer, "wer": 1 - wer}
            weight_sum = WEIGHTS["cer"] + WEIGHTS["wer"]
            if mixed_term_rate is not None:
                components["mixed_term"] = mixed_term_rate
                weight_sum += WEIGHTS["mixed_term"]
            if keyword_rate is not None:
                components["keyword"] = keyword_rate
                weight_sum += WEIGHTS["keyword"]
            if rtf_score is not None:
                components["rtf"] = rtf_score
                weight_sum += WEIGHTS["rtf"]
            final_score = sum(WEIGHTS[k] * v for k, v in components.items()) / weight_sum if weight_sum else None

            metrics_rows.append({
                "clip_id": clip, "model": model,
                "cer": f"{cer:.4f}", "wer": f"{wer:.4f}",
                "mixed_term_rate": f"{mixed_term_rate:.4f}" if mixed_term_rate is not None else "",
                "keyword_rate": f"{keyword_rate:.4f}" if keyword_rate is not None else "",
                "rtf": f"{rtf:.4f}" if rtf is not None else "",
                "rtf_score": f"{rtf_score:.4f}" if rtf_score is not None else "",
                "final_score": f"{final_score:.4f}" if final_score is not None else "",
            })

            subs, dels, ins, n_subs, n_dels, n_ins = word_alignment(ref, hyp)
            mixed_term_errs = sum(1 for _, s in mt_detail if s == 0.5)
            order_swaps = sum(1 for _, retained, _, swapped in kw_detail if swapped)
            error_rows.append({
                "clip_id": clip, "model": model,
                "누락_omission": n_dels, "오인식_substitution": n_subs, "삽입_insertion": n_ins,
                "혼용용어_표기오류": mixed_term_errs, "순서바뀜": order_swaps,
            })

            err_examples = []
            if subs:
                err_examples.append("오인식: " + ", ".join(f"{r}→{h}" for r, h in subs[:5]))
            if dels:
                err_examples.append("누락: " + ", ".join(dels[:5]))
            if ins:
                err_examples.append("삽입: " + ", ".join(ins[:5]))
            qual_rows.append({
                "clip_id": clip, "model": model,
                "stt_text": stt_raw, "gt_text": gt_raw,
                "오류_사례": " | ".join(err_examples),
                "문장_자연스러움": "", "작업_맥락_유지": "",
                "핵심_정보_누락_여부": "", "한영_혼용_의미_보존": "", "LLM_입력_적합성": "",
            })

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS_DIR / "metrics_summary.csv", metrics_rows,
              ["clip_id", "model", "cer", "wer", "mixed_term_rate", "keyword_rate", "rtf", "rtf_score", "final_score"])
    write_csv(RESULTS / "qualitative_review.csv", qual_rows,
              ["clip_id", "model", "stt_text", "gt_text", "오류_사례",
               "문장_자연스러움", "작업_맥락_유지", "핵심_정보_누락_여부", "한영_혼용_의미_보존", "LLM_입력_적합성"])
    write_csv(RESULTS / "error_analysis.csv", error_rows,
              ["clip_id", "model", "누락_omission", "오인식_substitution", "삽입_insertion", "혼용용어_표기오류", "순서바뀜"])

    write_final_report(metrics_rows)
    print("평가 완료:", METRICS_DIR / "metrics_summary.csv")


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_final_report(metrics_rows: list) -> None:
    by_model = {}
    for row in metrics_rows:
        if row["final_score"] == "":
            continue
        by_model.setdefault(row["model"], []).append(row)

    ranking = []
    for model, rows in by_model.items():
        scores = [float(r["final_score"]) for r in rows]
        cers = [float(r["cer"]) for r in rows]
        avg_score = sum(scores) / len(scores)
        avg_cer = sum(cers) / len(cers)
        ranking.append((model, avg_score, avg_cer, len(rows)))
    ranking.sort(key=lambda x: (-x[1], x[2]))

    # 계획서 10번은 "4개 클립 평균"을 최종 선정 기준으로 명시하므로, 클립이 4개 미만인
    # 모델(예: speechbrain — clip1만 실행)은 참고용으로만 표에 표시하고 선정 후보에서 제외한다.
    complete_ranking = [r for r in ranking if r[3] == len(CLIPS)]
    incomplete_ranking = [r for r in ranking if r[3] != len(CLIPS)]

    lines = ["# STT 모델 비교 최종 리포트", ""]
    lines.append(
        "> 실행 환경: 3개 모델 모두 GPU(n5, RTX6000, SLURM srun)에서 동일 세션 순차 실행함. "
        "faster_whisper는 compute_type float16, speechbrain/owsm은 device cuda로 실행."
    )
    lines.append(
        "> 본 실험은 원본 오디오(dataset/audio, 약 3분/클립) 대신 VAD로 무음을 제거한 "
        "dataset/audio_vad(16~36초/클립)를 입력으로 사용했다. 이는 raw vs vad 비교 실험 "
        "(results_raw_vs_vad/)을 통해, 원본 오디오 사용 시 whisper 계열 모델에서 무음 구간의 "
        "할루시네이션(직전 문장 반복, 의미 없는 한영 혼용 텍스트 생성)으로 CER이 심각하게 "
        "왜곡됨을 확인했기 때문이다(예: clip2 faster_whisper_prompt, 삽입오류 37건 → 실제 "
        "텍스트 확인 결과 '연결이 될 수 있습니다' 3회 반복 등). VAD 적용 후 CER·WER·할루시네이션이 "
        "87~94% 개선되었다(results_raw_vs_vad/conclusion.md 참고)."
    )
    lines.append(
        "> '작업 핵심 키워드 인식률'과 '순서바뀜'은 토큰 등장/순서 기반 자동 휴리스틱이며 "
        "완전한 의미 이해가 아니다. 최종 판단은 results/qualitative_review.csv의 수동 채점을 병행할 것."
    )
    lines.append(
        "> **speechbrain 관련 특이사항**: VAD로 무음 제거 후 클립 길이가 16~36초로 줄어들어 4클립 모두 "
        "실행이 가능해졌으나(RTF는 여전히 7.70~14.77로 실시간처리 기준(RTF<1.0)을 크게 초과, 배점 15%는 "
        "0점 확정), CER도 2.39~3.25로 다른 두 모델(0.01~0.18대)에 비해 현저히 나쁘다. 원인은 배포 기본값 "
        "test_beam_size=66 + CTC/TransformerLM joint scoring 구조로 추정되며(계획서 8번 '임의 튜닝 금지' "
        "원칙에 따라 beam size는 조정하지 않음), 이 모델은 RTF·CER 두 기준 모두에서 실사용에 부적합하다고 "
        "판단된다. 자세한 경위는 STT_연구계획서_CLI.md 5-2, 5-3절 참고."
    )
    lines.append("")
    lines.append("## 최종 점수 랭킹 (4개 클립 평균, 동점시 CER 낮은 모델 우선)")
    lines.append("")
    lines.append("| 순위 | 모델 | 평균 최종 점수 | 평균 CER | 평가 클립 수 | 비고 |")
    lines.append("|---|---|---|---|---|---|")
    for i, (model, score, cer, n) in enumerate(complete_ranking, 1):
        lines.append(f"| {i} | {model} | {score:.4f} | {cer:.4f} | {n} | |")
    for model, score, cer, n in incomplete_ranking:
        lines.append(f"| - | {model} | {score:.4f} | {cer:.4f} | {n}/{len(CLIPS)} | 4클립 미완료, 참고용 (선정 후보 제외) |")
    lines.append("")

    if complete_ranking:
        winner = complete_ranking[0][0]
        lines.append(f"## 선정 모델: **{winner}**")
        lines.append("")
        lines.append(
            f"- 선정 사유: 4개 클립 평균 가중합 최종 점수가 {complete_ranking[0][1]:.4f}로 "
            f"(4클립 완료 모델 중) 최고, 평균 CER {complete_ranking[0][2]:.4f}."
        )
        if len(complete_ranking) > 1:
            runner = complete_ranking[1]
            lines.append(
                f"- 2위({runner[0]}, {runner[1]:.4f})와의 점수 차이: {complete_ranking[0][1] - runner[1]:.4f}."
            )
        if incomplete_ranking:
            names = ", ".join(f"{m}({s:.4f}, {n}/{len(CLIPS)}클립)" for m, s, _, n in incomplete_ranking)
            lines.append(
                f"- 4클립을 다 채우지 못해 선정 후보에서 제외된 모델: {names}. "
                "제외 사유는 위 speechbrain 특이사항 및 STT_연구계획서_CLI.md 5-2절 참고."
            )
        lines.append(
            "- 본 모델은 반드시 VAD 무음 트리밍 전처리와 함께 사용해야 하며, "
            "원본 오디오를 직접 입력하는 것은 권장하지 않는다."
        )
        lines.append(
            "- 세부 지표(CER/WER/한영혼용용어/핵심키워드/RTF)는 "
            "results/metrics/metrics_summary.csv 참고."
        )
        lines.append(
            "- 오류 유형별 집계는 results/error_analysis.csv, 정성 평가는 "
            "results/qualitative_review.csv(수동 채점 필요) 참고."
        )
    else:
        lines.append("## 선정 모델: (4클립을 모두 완료한 모델이 없습니다)")

    (RESULTS / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
