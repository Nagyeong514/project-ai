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
    """
    results = {}
    pairs = [
        ("A", "A_prime", "배치 효과"),
        ("A_prime", "B", "VAD 효과"),
        ("A", "B", "전체 파이프라인 효과"),
    ]
    for cond_a, cond_b, label in pairs:
        vals_a = df[df["condition"] == cond_a].sort_values("file_id")[metric].tolist()
        vals_b = df[df["condition"] == cond_b].sort_values("file_id")[metric].tolist()
        if not vals_a or not vals_b:
            continue
        results[label] = wilcoxon_test(vals_a, vals_b, alpha=alpha, label_a=cond_a, label_b=cond_b)
    return results
