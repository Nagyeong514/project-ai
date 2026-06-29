"""
3종 시각화 생성:
  1. CER 비교 바차트 (발화유형별 그룹화)
  2. CER vs RTF 산점도 (가설 2: 정확도-속도 트레이드오프)
  3. 자체구축 최선 vs 상용 격차표 (가설 3)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.rcParams["font.family"] = "NanumGothic"  # 한글 폰트 (서버 환경에 설치 필요)
plt.rcParams["axes.unicode_minus"] = False


def plot_cer_by_utterance_type(df: pd.DataFrame, output_path: str) -> None:
    """CER 비교 바차트 — utterance_type 별 그룹화, 모델별 색상."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    types = df["utterance_type"].unique()

    for ax, utype in zip(axes, sorted(types)):
        sub = df[df["utterance_type"] == utype]
        means = sub.groupby("model_label")["cer"].mean().sort_values()
        means.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"유형 {utype}")
        ax.set_xlabel("")
        ax.set_ylabel("CER")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("모델별 CER 비교 (낭독 vs 대화)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cer_vs_rtf(df: pd.DataFrame, output_path: str) -> None:
    """CER vs RTF 산점도 — 자체구축 모델만 (API 제외)."""
    sub = df[df["rtf_note"] != "net"]
    agg = sub.groupby("model_label").agg(cer=("cer", "mean"), rtf=("rtf_mean", "mean")).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in agg.iterrows():
        ax.scatter(row["rtf"], row["cer"], s=120, zorder=3)
        ax.annotate(row["model_label"], (row["rtf"], row["cer"]), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("RTF (낮을수록 빠름)")
    ax.set_ylabel("CER")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("CER vs RTF (자체구축 모델)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_gap_table(gap_df: pd.DataFrame, output_path: str) -> None:
    """자체구축 최선 vs 상용 격차를 테이블 이미지로 저장."""
    display_cols = ["file_id", "best_model", "best_deployable_cer", "clova_cer", "kakao_cer",
                    "gap_clova_pct", "gap_kakao_pct"]
    existing = [c for c in display_cols if c in gap_df.columns]
    sub = gap_df[existing].copy()
    for col in ["best_deployable_cer", "clova_cer", "kakao_cer"]:
        if col in sub.columns:
            sub[col] = sub[col].map(lambda x: f"{x*100:.1f}%" if x is not None else "-")
    for col in ["gap_clova_pct", "gap_kakao_pct"]:
        if col in sub.columns:
            sub[col] = sub[col].map(lambda x: f"{x:+.1f}%p" if x is not None else "-")

    fig, ax = plt.subplots(figsize=(12, max(3, len(sub) * 0.5 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=sub.values,
        colLabels=sub.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title("자체구축 최선 vs 상용 API CER 격차", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
