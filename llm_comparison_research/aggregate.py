"""aggregate.py — 집계·통계·순위·시각화 (계획서 3.4 / 3.6 / 5 / 6절).

results/scores.csv를 읽어 모델 선정에 필요한 산출물을 만든다.

- 최종 점수(계획서 3.4): Final Score = 0.25·Faithfulness + 0.20·Accuracy + 0.20·Usefulness
  + 0.15·CodeSwitch + 0.10·Fluency + 0.10·Format  (1~5점 척도). 가중치는 config에서 읽는다.
- 통계(계획서 3.6): 항목별·종합 평균/표준편차, 모델 쌍 간 paired t-test(대응표본, α=0.05).
- 순위·의사결정(계획서 6): 종합 1위 채택. 단 1·2위 차가 유의하지 않으면(p>=α)
  "유의차 없음 — 운영조건 고려 필요"를 표시.
- 시각화: 모델별 항목 평균 막대그래프 → results/plot.png.

입력: results/scores.csv, config.yaml(weights, stats)
출력: results/summary.csv, results/summary.md, results/plot.png (config.paths)

사용:
  python aggregate.py
  python aggregate.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from scipy import stats

ITEMS = ["faithfulness", "accuracy", "usefulness", "codeswitch", "fluency", "format"]
ITEM_KR = {
    "faithfulness": "충실도",
    "accuracy": "정확성",
    "usefulness": "유용성",
    "codeswitch": "한영혼용",
    "fluency": "한국어",
    "format": "형식",
}


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_final_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """(sample, model) 행별 가중합 Final Score (계획서 3.4)."""
    score = sum(df[it] * weights[it] for it in ITEMS)
    return score


def per_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """모델별 항목 평균·표준편차 + Final Score 평균·표준편차 (계획서 3.6)."""
    rows = []
    for model, g in df.groupby("model"):
        row = {"model": model, "n": len(g)}
        for it in ITEMS:
            row[f"{it}_mean"] = g[it].mean()
            row[f"{it}_std"] = g[it].std(ddof=1)
        row["final_mean"] = g["final_score"].mean()
        row["final_std"] = g["final_score"].std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("final_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def paired_ttests(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """모델 쌍 간 Final Score paired t-test (동일 샘플 대응표본, 계획서 3.6).

    각 모델의 sample_id별 Final Score를 정렬해 공통 샘플만 대응 비교한다.
    """
    pivot = df.pivot_table(index="sample_id", columns="model", values="final_score")
    models = list(pivot.columns)
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            pair = pivot[[a, b]].dropna()  # 둘 다 점수 있는 공통 샘플만
            n = len(pair)
            if n < 2:
                rows.append({"model_a": a, "model_b": b, "n_pairs": n,
                             "mean_diff": float("nan"), "t_stat": float("nan"),
                             "p_value": float("nan"), "significant": False})
                continue
            t, p = stats.ttest_rel(pair[a], pair[b])
            rows.append({
                "model_a": a, "model_b": b, "n_pairs": n,
                "mean_diff": pair[a].mean() - pair[b].mean(),
                "t_stat": t, "p_value": p, "significant": bool(p < alpha),
            })
    return pd.DataFrame(rows)


def top2_verdict(model_tbl: pd.DataFrame, ttests: pd.DataFrame, alpha: float) -> str:
    """1·2위 유의성 판정 문구 (계획서 6)."""
    if len(model_tbl) < 2:
        return "모델이 1개뿐 — 비교 불가."
    first = model_tbl.iloc[0]["model"]
    second = model_tbl.iloc[1]["model"]
    mask = (((ttests["model_a"] == first) & (ttests["model_b"] == second)) |
            ((ttests["model_a"] == second) & (ttests["model_b"] == first)))
    sub = ttests[mask]
    if sub.empty or pd.isna(sub.iloc[0]["p_value"]):
        return f"1위 **{first}** vs 2위 **{second}**: 표본 부족으로 검정 불가."
    p = float(sub.iloc[0]["p_value"])
    if p < alpha:
        return (f"1위 **{first}** 채택 — 2위 **{second}** 대비 유의한 우위 "
                f"(p={p:.4f} < {alpha}).")
    return (f"1위 **{first}** vs 2위 **{second}**: **유의차 없음 — 운영조건 고려 필요** "
            f"(p={p:.4f} ≥ {alpha}). 표준편차·VRAM·추론속도 등으로 결정 권장.")


def faithfulness_warnings(model_tbl: pd.DataFrame, floor: float) -> list[str]:
    """충실도 평균이 하한 미만인 모델 경고 (계획서 6: 순위 무관 제외 검토)."""
    warns = []
    for _, r in model_tbl.iterrows():
        fm = r["faithfulness_mean"]
        if fm < floor:
            warns.append(f"**{r['model']}**: 충실도 평균 {fm:.2f} < 하한 {floor:.1f} "
                         f"→ **충실도 하한 미달 — 순위와 무관하게 제외 검토**(지식 DB 오염 위험).")
    return warns


def write_summary_md(path: Path, model_tbl: pd.DataFrame, ttests: pd.DataFrame,
                     verdict: str, weights: dict, alpha: float,
                     fl_warnings: list[str], floor: float) -> None:
    lines = ["# 모델 비교 집계 결과 (계획서 3.4 / 3.6 / 6)", ""]
    lines.append("Final Score = " + " + ".join(
        f"{weights[it]:.2f}·{ITEM_KR[it]}" for it in ITEMS) + "  (1~5점 척도)")
    lines += ["", "## 1. 모델별 종합 순위 (Final Score 내림차순)", ""]

    lines.append("| 순위 | 모델 | n | Final 평균 | Final 표준편차 |")
    lines.append("|---|---|---|---|---|")
    for _, r in model_tbl.iterrows():
        lines.append(f"| {int(r['rank'])} | {r['model']} | {int(r['n'])} | "
                     f"{r['final_mean']:.3f} | {r['final_std']:.3f} |")

    lines += ["", "## 2. 항목별 평균 (표준편차)", ""]
    header = "| 모델 | " + " | ".join(ITEM_KR[it] for it in ITEMS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ITEMS) + 1))
    for _, r in model_tbl.iterrows():
        cells = " | ".join(f"{r[f'{it}_mean']:.2f} ({r[f'{it}_std']:.2f})" for it in ITEMS)
        lines.append(f"| {r['model']} | {cells} |")

    lines += ["", f"## 3. 모델 쌍 paired t-test (대응표본, α={alpha})", ""]
    lines.append("| A | B | n쌍 | 평균차(A-B) | t | p-value | 유의 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in ttests.iterrows():
        p = r["p_value"]
        pstr = "NA" if pd.isna(p) else f"{p:.4f}"
        tstr = "NA" if pd.isna(r["t_stat"]) else f"{r['t_stat']:.3f}"
        dstr = "NA" if pd.isna(r["mean_diff"]) else f"{r['mean_diff']:+.3f}"
        sig = "✔" if r["significant"] else "—"
        lines.append(f"| {r['model_a']} | {r['model_b']} | {int(r['n_pairs'])} | "
                     f"{dstr} | {tstr} | {pstr} | {sig} |")

    lines += ["", "## 4. 1·2위 의사결정 (계획서 6)", "", verdict, ""]

    lines += [f"## 5. 충실도 하한 점검 (임계값 {floor:.1f}, 계획서 6)", ""]
    if fl_warnings:
        for w in fl_warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append(f"- 모든 모델이 충실도 하한({floor:.1f}) 이상 — 제외 검토 대상 없음.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _setup_korean_font() -> bool:
    """설치된 한글 폰트를 찾아 적용. 있으면 True, 없으면 False(→ 영문 라벨 폴백)."""
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("NanumGothic", "Malgun Gothic", "AppleGothic", "NanumBarunGothic",
                 "Noto Sans CJK KR", "UnDotum"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            return True
    return False


def plot_items(path: Path, model_tbl: pd.DataFrame) -> None:
    """모델별 항목 평균 그룹 막대그래프. 한글 폰트 없으면 영문 라벨로 폴백(깨짐 방지)."""
    import numpy as np

    has_kr = _setup_korean_font()
    item_labels = [ITEM_KR[it] for it in ITEMS] if has_kr else [it for it in ITEMS]
    ylabel = "평균 점수 (1~5)" if has_kr else "Mean score (1-5)"
    title = "모델별 항목 평균 (계획서 3.4)" if has_kr else "Per-item mean by model (plan 3.4)"

    models = model_tbl["model"].tolist()
    x = np.arange(len(ITEMS))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, r) in enumerate(model_tbl.iterrows()):
        vals = [r[f"{it}_mean"] for it in ITEMS]
        ax.bar(x + i * width, vals, width, label=r["model"])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(item_labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 5)
    ax.set_title(title)
    ax.legend(title="model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="집계·통계·순위·시각화 (계획서 3.4/3.6/6)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    root = Path(args.config).resolve().parent
    cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    weights = cfg["weights"]
    alpha = float(cfg.get("stats", {}).get("alpha", 0.05))

    # 가중치 합 검증 (계획서 3.4: 합=1)
    wsum = sum(weights[it] for it in ITEMS)
    if abs(wsum - 1.0) > 1e-9:
        raise SystemExit(f"[오류] 가중치 합이 1이 아닙니다: {wsum} (config.weights 확인)")

    scores_path = root / paths["scores_csv"]
    if not scores_path.exists():
        raise SystemExit(f"[오류] {scores_path} 없음 — 먼저 judge.py 를 실행하세요.")
    df = pd.read_csv(scores_path)
    for it in ITEMS:
        if it not in df.columns:
            raise SystemExit(f"[오류] scores.csv 에 '{it}' 컬럼이 없습니다.")

    df["final_score"] = compute_final_score(df, weights)

    floor = float(cfg.get("stats", {}).get("faithfulness_floor", 3.0))
    model_tbl = per_model_table(df)
    ttests = paired_ttests(df, alpha)
    verdict = top2_verdict(model_tbl, ttests, alpha)
    fl_warnings = faithfulness_warnings(model_tbl, floor)

    # 콘솔 출력
    print(f"가중치 합 = {wsum:.2f} (OK)  |  샘플 {df['sample_id'].nunique()}개 × 모델 {df['model'].nunique()}종\n")
    print("[모델별 종합/항목 (평균·표준편차)]")
    show_cols = ["rank", "model", "n", "final_mean", "final_std"] + [f"{it}_mean" for it in ITEMS]
    print(model_tbl[show_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[paired t-test]")
    print(ttests.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n[의사결정]", verdict)
    if fl_warnings:
        print(f"[충실도 하한 {floor:.1f}]")
        for w in fl_warnings:
            print("  ⚠️", w.replace("**", ""))
    else:
        print(f"[충실도 하한 {floor:.1f}] 미달 모델 없음")

    # 저장: summary.csv (모델별 표) + 통계는 별도 컬럼 묶음으로
    out_csv = root / paths["summary_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    model_tbl.to_csv(out_csv, index=False, encoding="utf-8-sig")
    ttests.to_csv(out_csv.with_name("summary_ttests.csv"), index=False, encoding="utf-8-sig")

    write_summary_md(root / paths["summary_md"], model_tbl, ttests, verdict, weights, alpha,
                     fl_warnings, floor)
    plot_items(root / paths["plot_png"], model_tbl)

    print(f"\n  ✔ {out_csv.relative_to(root)}")
    print(f"  ✔ {(out_csv.with_name('summary_ttests.csv')).relative_to(root)}")
    print(f"  ✔ {(root / paths['summary_md']).relative_to(root)}")
    print(f"  ✔ {(root / paths['plot_png']).relative_to(root)}")


if __name__ == "__main__":
    main()
