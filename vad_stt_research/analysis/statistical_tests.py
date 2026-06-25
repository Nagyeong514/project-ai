"""
통계 검정 — Wilcoxon signed-rank test (α = 0.05).
동일 파일 쌍 대응 비교 (표본 소수·비정규 가정).
"""
from typing import List, Dict

import numpy as np
from scipy import stats


def wilcoxon_test(
    values_a: List[float],
    values_b: List[float],
    alpha: float = 0.05,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    paired Wilcoxon signed-rank test.
    values_a[i], values_b[i] 는 동일 파일에서 나온 값이어야 함.
    """
    if len(values_a) != len(values_b):
        raise ValueError("두 리스트 길이가 달라 대응 비교 불가")
    if len(values_a) < 5:
        print(f"[경고] 표본 수 {len(values_a)} — Wilcoxon 검정력 낮음 (권장: ≥10)")

    stat, p_value = stats.wilcoxon(values_a, values_b, alternative="two-sided")
    significant = p_value < alpha
    diff = [b - a for a, b in zip(values_a, values_b)]

    return {
        "test": "wilcoxon_signed_rank",
        "label_a": label_a,
        "label_b": label_b,
        "n": len(values_a),
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": significant,
        "alpha": alpha,
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "mean_diff_b_minus_a": float(np.mean(diff)),
        "median_diff_b_minus_a": float(np.median(diff)),
    }


def run_all_comparisons(df, metric: str, alpha: float = 0.05) -> Dict[str, dict]:
    """
    DataFrame에서 조건별 metric 값을 추출해 3쌍 비교 수행.
    df 컬럼: file_id, condition (A / A_prime / B), metric

    file_id 기준으로 두 조건을 짝지어 비교한다(같은 파일끼리 대응).
    한쪽이라도 결측인 file_id는 해당 비교에서 제외해 짝을 보존한다.
    """
    results = {}
    pairs = [
        ("A", "A_prime", "배치 효과"),
        ("A_prime", "B", "VAD 효과"),
        ("A", "B", "전체 파이프라인 효과"),
    ]
    wide = df.pivot_table(index="file_id", columns="condition", values=metric)
    for cond_a, cond_b, label in pairs:
        if cond_a not in wide.columns or cond_b not in wide.columns:
            continue
        paired = wide[[cond_a, cond_b]].dropna()
        if len(paired) < 1:
            continue
        try:
            results[label] = wilcoxon_test(
                paired[cond_a].tolist(),
                paired[cond_b].tolist(),
                alpha=alpha,
                label_a=cond_a,
                label_b=cond_b,
            )
        except ValueError as e:
            # 차이가 전부 0이거나 표본이 비어 검정 불가한 경우
            results[label] = {
                "test": "wilcoxon_signed_rank",
                "label_a": cond_a,
                "label_b": cond_b,
                "n": len(paired),
                "error": str(e),
                "mean_a": float(paired[cond_a].mean()),
                "mean_b": float(paired[cond_b].mean()),
                "mean_diff_b_minus_a": float((paired[cond_b] - paired[cond_a]).mean()),
            }
    return results


METRICS = {
    "wer": "WER",
    "cer": "CER",
    "hallucination_per_hour": "할루시네이션/시간",
    "rtf_mean": "RTF",
    "timestamp_drift_late_s": "후반 타임스탬프 드리프트(s)",
}


def main():
    import argparse
    import json

    import pandas as pd

    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank 검정 (조건 A/A'/B 대응 비교)")
    parser.add_argument("--results", default="results/raw/results.csv", help="실험 결과 CSV")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default=None, help="검정 결과 JSON 저장 경로 (선택)")
    args = parser.parse_args()

    df = pd.read_csv(args.results)

    all_out = {}
    for metric, kor in METRICS.items():
        if metric not in df.columns:
            continue
        comparisons = run_all_comparisons(df, metric, alpha=args.alpha)
        if not comparisons:
            continue
        all_out[metric] = comparisons
        print(f"\n{'='*64}\n[{kor}]  (metric={metric})")
        for label, r in comparisons.items():
            if "error" in r:
                print(f"  {label} ({r['label_a']}→{r['label_b']}, n={r['n']}): "
                      f"검정 불가 — {r['error']} | 평균차 {r['mean_diff_b_minus_a']:+.4f}")
                continue
            sig = "유의함 ✔" if r["significant"] else "유의하지 않음"
            print(f"  {label} ({r['label_a']}→{r['label_b']}, n={r['n']}): "
                  f"p={r['p_value']:.4f} [{sig}] | "
                  f"평균 {r['mean_a']:.4f}→{r['mean_b']:.4f} (차 {r['mean_diff_b_minus_a']:+.4f})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_out, f, ensure_ascii=False, indent=2)
        print(f"\n검정 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
