"""
시각화 자동 생성 (RESEARCH_PLAN 5.2절 5종).

① 정확도·할루시네이션 Grouped Bar (조건 A/A'/B × 군별)        — 가설 1
② 속도 향상 요인 분해 Waterfall (배치 기여 / VAD 기여)          — 가설 2
③ 무음 비율 vs RTF 이득 Scatter + 회귀선 + 신뢰구간            — 가설 2 손익분기
④ VAD 파라미터 민감도 Multi-line (pad_ms × threshold)         — 가설 3
⑤ 타임스탬프 누적 드리프트 Timeline (조건별 Δt 추이)           — 가설 1

데이터가 없는 그래프(④ 민감도 등)는 건너뛴다.
"""
import argparse
import json
import os
from pathlib import Path

from analysis._plot_style import set_korean_font  # Agg 백엔드 + 한글 폰트

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.breakeven_analysis import load_results, compute_breakeven

COND_ORDER = ["A", "A_prime", "B"]
COND_LABEL = {"A": "A (Vanilla)", "A_prime": "A′ (배치)", "B": "B (VAD+배치)"}


def _group(silence_ratio: float) -> str:
    return "high_silence" if silence_ratio >= 0.5 else "low_silence"


# ── ① 정확도·할루시네이션 ────────────────────────────────────
def plot_accuracy_hallucination(df: pd.DataFrame, outdir: str) -> str:
    set_korean_font()
    d = df.copy()
    d["group"] = d["silence_ratio"].apply(_group)
    d["condition"] = pd.Categorical(d["condition"], categories=COND_ORDER, ordered=True)

    metrics = [("wer", "WER"), ("cer", "CER"), ("hallucination_per_hour", "할루시네이션/시간")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, title) in zip(axes, metrics):
        if col not in d.columns:
            continue
        sns.barplot(data=d, x="condition", y=col, hue="group", ax=ax,
                    order=COND_ORDER, errorbar="sd")
        ax.set_title(f"① {title}")
        ax.set_xlabel("")
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER], rotation=0)
        ax.legend(title="무음군", fontsize=8)
    fig.suptitle("정확도·할루시네이션 (조건 × 무음군)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, "fig1_accuracy_hallucination.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── ② 속도 요인 분해 Waterfall ───────────────────────────────
def plot_speed_waterfall(df: pd.DataFrame, outdir: str, metadata_csv: str | None = None) -> str:
    set_korean_font()
    rtf = df.pivot_table(index="file_id", columns="condition", values="rtf_mean")
    mean_a = rtf["A"].mean()
    mean_ap = rtf["A_prime"].mean()

    # 실질 B: VAD 오버헤드 포함 (metadata 있으면 load_results로 재계산)
    if metadata_csv:
        real = load_results(_current_results, metadata_csv)
        mean_b = real["rtf_B_real"].mean()
        b_title = "B (실질, VAD 포함)"
    else:
        mean_b = rtf["B"].mean()
        b_title = "B (STT만)"

    batch_contrib = mean_ap - mean_a   # 보통 음수 (빨라짐)
    vad_contrib = mean_b - mean_ap     # 보통 양수 (느려짐)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["A", "배치 기여\n(A→A′)", "VAD 기여\n(A′→B)", b_title]
    # 누적 위치 계산
    starts = [0, mean_a, mean_a + batch_contrib, 0]
    heights = [mean_a, batch_contrib, vad_contrib, mean_b]
    colors = ["#4c72b0", "#55a868" if batch_contrib < 0 else "#c44e52",
              "#c44e52" if vad_contrib > 0 else "#55a868", "#4c72b0"]
    for i, (lab, st, h, c) in enumerate(zip(labels, starts, heights, colors)):
        ax.bar(i, h, bottom=st, color=c, edgecolor="black", linewidth=0.6)
        ax.text(i, st + h + (0.001 if h >= 0 else -0.001),
                f"{(st + h):.4f}" if i in (0, 3) else f"{h:+.4f}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("RTF (평균)")
    ax.set_title("② 속도 요인 분해: 배치 기여 vs VAD 기여")
    ax.axhline(0, color="gray", linewidth=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, "fig2_speed_waterfall.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── ③ 손익분기 Scatter + CI ──────────────────────────────────
def plot_breakeven_scatter(results_csv: str, metadata_csv: str, outdir: str) -> str:
    set_korean_font()
    df = load_results(results_csv, metadata_csv)
    be = compute_breakeven(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x="silence_ratio", y="rtf_gain", ax=ax,
                scatter_kws={"s": 80}, line_kws={"color": "crimson"}, ci=95)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    bp = be["breakeven_silence_ratio"]
    if bp is not None and 0 <= bp <= 1:
        ax.axvline(bp, color="navy", linestyle=":", label=f"손익분기점 ≈ {bp:.2f}")
        ax.legend()
    else:
        ax.text(0.5, 0.95, "관측 구간 내 손익분기점 없음 (B가 항상 느림)",
                transform=ax.transAxes, ha="center", va="top", fontsize=9, color="navy")
    ax.set_xlabel("무음 비율")
    ax.set_ylabel("RTF 이득 (A′ − 실질 B, 양수 = B 빠름)")
    ax.set_title("③ VAD 손익분기: 무음 비율 vs RTF 이득 (95% CI)")
    plt.tight_layout()
    path = os.path.join(outdir, "fig3_breakeven_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── ④ VAD 파라미터 민감도 ────────────────────────────────────
def plot_sensitivity(sensitivity_csv: str, outdir: str) -> str | None:
    if not os.path.exists(sensitivity_csv):
        print(f"  [건너뜀] 민감도 데이터 없음: {sensitivity_csv}")
        return None
    set_korean_font()
    d = pd.read_csv(sensitivity_csv)
    agg = d.groupby(["threshold", "speech_pad_ms"])["wer"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for thr, sub in agg.groupby("threshold"):
        sub = sub.sort_values("speech_pad_ms")
        ax.plot(sub["speech_pad_ms"], sub["wer"], marker="o", label=f"threshold={thr}")
    ax.set_xlabel("speech_pad_ms")
    ax.set_ylabel("WER (평균)")
    ax.set_title("④ VAD 파라미터 민감도: padding × threshold")
    ax.legend(title="VAD threshold")
    plt.tight_layout()
    path = os.path.join(outdir, "fig4_sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── ⑤ 타임스탬프 드리프트 Timeline ───────────────────────────
def plot_drift_timeline(drift_dir: str, outdir: str) -> str | None:
    files = list(Path(drift_dir).glob("*_*.json"))
    if not files:
        print(f"  [건너뜀] 드리프트 데이터 없음: {drift_dir}")
        return None
    set_korean_font()

    # 조건별 버킷 누적: {cond: {bucket_start_min: [drift, ...]}}
    by_cond: dict[str, dict[float, list]] = {}
    for fp in files:
        stem = fp.stem
        # {file_id}_{cond}; cond in A / A_prime / B
        for cond in ("A_prime", "A", "B"):
            if stem.endswith(f"_{cond}"):
                break
        else:
            continue
        with open(fp, encoding="utf-8") as f:
            buckets = json.load(f)
        slot = by_cond.setdefault(cond, {})
        for bk in buckets:
            if bk.get("mean_drift_s") is not None:
                slot.setdefault(bk["bucket_start_min"], []).append(bk["mean_drift_s"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in COND_ORDER:
        if cond not in by_cond:
            continue
        xs = sorted(by_cond[cond].keys())
        ys = [np.mean(by_cond[cond][x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=COND_LABEL[cond])
    ax.axvline(50, color="gray", linestyle=":", linewidth=0.8, label="후반부 기준(50분)")
    ax.set_xlabel("오디오 경과 시간 (분)")
    ax.set_ylabel("평균 타임스탬프 드리프트 (s)")
    ax.set_title("⑤ 타임스탬프 누적 드리프트 추이 (조건별)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "fig5_drift_timeline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# 모듈 레벨에서 waterfall이 참조할 현재 results 경로 (main에서 설정)
_current_results = "results/raw/results.csv"


def main():
    global _current_results
    parser = argparse.ArgumentParser(description="VAD STT 연구 시각화 5종 생성")
    parser.add_argument("--results", default="results/raw/results.csv")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--drift_dir", default="results/raw/drift_by_time")
    parser.add_argument("--sensitivity", default="results/figures/sensitivity_wer.csv")
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    _current_results = args.results
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.results)

    generated = []
    generated.append(plot_accuracy_hallucination(df, args.outdir))
    generated.append(plot_speed_waterfall(df, args.outdir, metadata_csv=args.metadata))
    generated.append(plot_breakeven_scatter(args.results, args.metadata, args.outdir))
    generated.append(plot_sensitivity(args.sensitivity, args.outdir))
    generated.append(plot_drift_timeline(args.drift_dir, args.outdir))

    print("\n생성된 그래프:")
    for g in generated:
        if g:
            print(f"  - {g}")


if __name__ == "__main__":
    main()
