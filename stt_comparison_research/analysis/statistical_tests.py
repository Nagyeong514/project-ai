"""Wilcoxon signed-rank test — 모델 간 CER 유의차 검정."""
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import wilcoxon


def run_pairwise_wilcoxon(
    df: pd.DataFrame,
    metric: str = "cer",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    df 컬럼: file_id, model_key, cer, wer, ...
    동일 file_id 쌍에 대해 모든 모델 조합 Wilcoxon signed-rank 수행.
    반환: DataFrame [model_a, model_b, statistic, p_value, significant]
    """
    models = df["model_key"].unique().tolist()
    records = []

    for m_a, m_b in combinations(models, 2):
        vals_a = df[df["model_key"] == m_a].set_index("file_id")[metric]
        vals_b = df[df["model_key"] == m_b].set_index("file_id")[metric]
        common = vals_a.index.intersection(vals_b.index)
        if len(common) < 3:
            continue
        diff = vals_a[common].values - vals_b[common].values
        if (diff == 0).all():
            continue
        stat, p = wilcoxon(diff)
        records.append(
            {
                "model_a": m_a,
                "model_b": m_b,
                "statistic": stat,
                "p_value": round(p, 4),
                "significant": p < alpha,
            }
        )

    return pd.DataFrame(records)


def self_deployable_vs_api_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    자체구축 최선(CER 최소) vs 상용 API(CLOVA/Kakao) CER 격차 테이블.
    반환: [file_id, best_deployable_cer, clova_cer, kakao_cer, gap_clova, gap_kakao]
    """
    deployable_keys = df[df["deployable"] == True]["model_key"].unique()
    api_keys = ["clova", "kakao"]

    records = []
    for fid in df["file_id"].unique():
        sub = df[df["file_id"] == fid]
        dep = sub[sub["model_key"].isin(deployable_keys)]
        best_cer = dep["cer"].min() if not dep.empty else None
        best_model = dep.loc[dep["cer"].idxmin(), "model_key"] if not dep.empty else None

        row: Dict = {"file_id": fid, "best_deployable_cer": best_cer, "best_model": best_model}
        for api in api_keys:
            api_row = sub[sub["model_key"] == api]
            api_cer = api_row["cer"].values[0] if not api_row.empty else None
            row[f"{api}_cer"] = api_cer
            if best_cer is not None and api_cer is not None:
                row[f"gap_{api}_pct"] = round((best_cer - api_cer) * 100, 2)
            else:
                row[f"gap_{api}_pct"] = None
        records.append(row)

    return pd.DataFrame(records)
