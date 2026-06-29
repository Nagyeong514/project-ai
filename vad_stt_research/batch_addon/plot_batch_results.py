"""
배치 추론 실험 시각화. results_batch.csv를 직접 읽어 집계 → 그림 생성.
(보고서 수치 하드코딩 X — CSV 재집계로 일관성 보장)

출력: batch_addon/figures/
  fig1_batch_overview.png   — 4조건 × 4지표 grouped bar (2x2)
  fig2_batch_speedup.png    — RTF 짝대비(A'↔A'_batch, B↔B_batch) + 속도배율
  fig3_group_b_batch.png    — high/low_silence별 B vs B_batch (실용 핵심)

실행:
  PYTHONPATH=/home/piai/project-ai/vad_stt_research \
    /home/piai/anaconda3/bin/python batch_addon/plot_batch_results.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis._plot_style import set_korean_font  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_batch.csv")
FIG_DIR = os.path.join(HERE, "figures")

CONDS = ["A_prime", "A_prime_batch", "B", "B_batch"]
LABELS = {"A_prime": "A′\n(비배치·VAD없음)", "A_prime_batch": "A′_batch\n(배치·고정30s)",
          "B": "B\n(비배치·VAD)", "B_batch": "B_batch\n(배치·VAD)"}
# 비배치=무채색, 배치=강조색. 같은 계열(A'/B)은 색상 통일.
COLORS = {"A_prime": "#9bbfc9", "A_prime_batch": "#2a9d8f",
          "B": "#e6b8a6", "B_batch": "#e76f51"}
METRICS = [("wer", "WER", False), ("cer", "CER", False),
           ("hallucination_per_hour", "할루시네이션 / 시간", False),
           ("rtf_mean", "RTF (낮을수록 빠름)", False)]


def load():
    df = pd.read_csv(CSV)
    df["grp"] = df.file_id.str[0].map({"H": "high", "L": "low"})
    return df


def fig1_overview(df):
    means = df.groupby("condition")[[m[0] for m in METRICS]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title, _) in zip(axes.ravel(), METRICS):
        vals = [means.loc[c, key] for c in CONDS]
        bars = ax.bar([LABELS[c] for c in CONDS], vals,
                      color=[COLORS[c] for c in CONDS], zorder=3, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if v < 10 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("배치 추론 실험 — 4조건 × 4지표 (10파일 평균)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG_DIR, "fig1_batch_overview.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2_speedup(df):
    means = df.groupby("condition")["rtf_mean"].mean()
    pairs = [("A_prime", "A_prime_batch"), ("B", "B_batch")]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(pairs))
    w = 0.36
    for i, (seq, bat) in enumerate(pairs):
        ax.bar(i - w / 2, means[seq], w, color=COLORS[seq], zorder=3, edgecolor="white")
        ax.bar(i + w / 2, means[bat], w, color=COLORS[bat], zorder=3, edgecolor="white")
        ax.text(i - w / 2, means[seq], f"{means[seq]:.4f}", ha="center", va="bottom", fontsize=10)
        ax.text(i + w / 2, means[bat], f"{means[bat]:.4f}", ha="center", va="bottom", fontsize=10)
        speedup = means[seq] / means[bat]
        ax.annotate(f"{speedup:.2f}배 빠름", xy=(i, max(means[seq], means[bat]) * 1.08),
                    ha="center", fontsize=12, fontweight="bold", color="#c1121f")
    ax.set_xticks(x)
    ax.set_xticklabels(["A′ → A′_batch\n(VAD 없음)", "B → B_batch\n(Silero VAD)"], fontsize=11)
    ax.set_ylabel("RTF (낮을수록 빠름)", fontsize=12)
    ax.set_title("배치 추론 속도 효과 — RTF (10파일 평균)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, means.max() * 1.25)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#9bbfc9"),
               plt.Rectangle((0, 0), 1, 1, color="#2a9d8f")]
    ax.legend(handles, ["비배치(순차)", "배치(BatchedInferencePipeline)"], frameon=False, fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_batch_speedup.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3_group_b_batch(df):
    sub = df[df.condition.isin(["B", "B_batch"])]
    g = sub.groupby(["grp", "condition"])[["wer", "cer", "hallucination_per_hour", "rtf_mean"]].mean()
    groups = ["high", "low"]
    metric_keys = [("wer", "WER"), ("cer", "CER"),
                   ("hallucination_per_hour", "할루시/시간"), ("rtf_mean", "RTF")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    x = np.arange(len(groups))
    w = 0.36
    for ax, (key, title) in zip(axes, metric_keys):
        b = [g.loc[(grp, "B"), key] for grp in groups]
        bb = [g.loc[(grp, "B_batch"), key] for grp in groups]
        ax.bar(x - w / 2, b, w, color=COLORS["B"], zorder=3, edgecolor="white", label="B (비배치)")
        ax.bar(x + w / 2, bb, w, color=COLORS["B_batch"], zorder=3, edgecolor="white", label="B_batch (배치)")
        ax.set_xticks(x)
        ax.set_xticklabels(["high_silence\n(간헐)", "low_silence\n(연속)"], fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("B vs B_batch — 무음 그룹별 (high_silence에서 배치가 정확도 유지 + 속도↑)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "fig3_group_b_batch.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    set_korean_font()
    df = load()
    print("saved:", fig1_overview(df))
    print("saved:", fig2_speedup(df))
    print("saved:", fig3_group_b_batch(df))


if __name__ == "__main__":
    main()
