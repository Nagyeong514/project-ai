"""
VAD 엔진 비교 시각화. results/comparison.csv → results/figures/ 4종.

① 실행비용 (VAD RTF, 로그)        — 가설1
② 검출정확도 (F1/P/R)             — 가설2
③ downstream WER (엔진×무음군)     — 가설3 + Phase1 연계
④ 비용 vs 정확도 산점 (파일별)      — 종합 매트릭스
"""
import argparse
import os
import sys

_PHASE1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vad_stt_research"))
if _PHASE1 not in sys.path:
    sys.path.insert(0, _PHASE1)

from analysis._plot_style import set_korean_font  # Phase1 재사용 (Agg + 한글폰트)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ENG_LABEL = {"reference": "참조(VAD없음)", "silero": "Silero", "webrtc": "WebRTC"}
ENG_COLOR = {"reference": "#7f7f7f", "silero": "#4c72b0", "webrtc": "#dd8452"}


def _load(path):
    df = pd.read_csv(path)
    for c in ["vad_rtf", "f1", "precision", "recall", "n_chunks", "avg_chunk_s",
              "stt_rtf", "wer", "cer", "hallucination_per_hour", "silence_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["grp"] = np.where(df["silence_ratio"] >= 0.5, "high_silence", "low_silence")
    return df


def fig_cost(df, outdir):
    set_korean_font()
    m = df[df.engine.isin(["silero", "webrtc"])].groupby("engine")["vad_rtf"].mean()
    m = m.reindex(["silero", "webrtc"])
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([ENG_LABEL[e] for e in m.index], m.values,
                  color=[ENG_COLOR[e] for e in m.index])
    ax.set_yscale("log")
    ax.set_ylabel("VAD RTF (로그스케일, 낮을수록 빠름)")
    ax.set_title("① 엔진 실행비용 (VAD RTF)")
    for b, v in zip(bars, m.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=10)
    ratio = m["silero"] / m["webrtc"]
    ax.text(0.5, 0.92, f"WebRTC가 {ratio:.0f}배 빠름", transform=ax.transAxes,
            ha="center", fontsize=11, color="crimson")
    plt.tight_layout(); p = os.path.join(outdir, "fig1_cost.png"); plt.savefig(p, dpi=150); plt.close()
    return p


def fig_detection(df, outdir):
    set_korean_font()
    m = df[df.engine.isin(["silero", "webrtc"])].groupby("engine")[["f1", "precision", "recall"]].mean()
    m = m.reindex(["silero", "webrtc"])
    metrics, x = ["f1", "precision", "recall"], np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.35
    for i, e in enumerate(["silero", "webrtc"]):
        ax.bar(x + (i - 0.5) * w, [m.loc[e, mt] for mt in metrics], w,
               label=ENG_LABEL[e], color=ENG_COLOR[e])
    ax.set_xticks(x); ax.set_xticklabels(["F1", "Precision", "Recall"])
    ax.set_ylabel("점수"); ax.set_ylim(0, 1)
    ax.set_title("② 검출 정확도 (vs 수동자막 GT)")
    ax.legend()
    plt.tight_layout(); p = os.path.join(outdir, "fig2_detection.png"); plt.savefig(p, dpi=150); plt.close()
    return p


def fig_wer(df, outdir):
    set_korean_font()
    piv = df.pivot_table(index="grp", columns="engine", values="wer", aggfunc="mean")
    piv = piv.reindex(index=["high_silence", "low_silence"], columns=["reference", "silero", "webrtc"])
    x = np.arange(2); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, e in enumerate(["reference", "silero", "webrtc"]):
        ax.bar(x + (i - 1) * w, piv[e].values, w, label=ENG_LABEL[e], color=ENG_COLOR[e])
    ax.set_xticks(x); ax.set_xticklabels(["high_silence", "low_silence"])
    ax.set_ylabel("WER (낮을수록 좋음)")
    ax.set_title("③ Downstream WER — 엔진 × 무음군 (참조 대비)")
    ax.legend()
    plt.tight_layout(); p = os.path.join(outdir, "fig3_wer.png"); plt.savefig(p, dpi=150); plt.close()
    return p


def fig_tradeoff(df, outdir):
    set_korean_font()
    sub = df[df.engine.isin(["silero", "webrtc"])]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for e in ["silero", "webrtc"]:
        s = sub[sub.engine == e]
        ax.scatter(s["vad_rtf"], s["f1"], s=70, color=ENG_COLOR[e], label=ENG_LABEL[e], alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("VAD RTF (로그, 왼쪽일수록 쌈)")
    ax.set_ylabel("검출 F1 (위일수록 정확)")
    ax.set_title("④ 비용 vs 정확도 (파일별)")
    ax.legend()
    ax.text(0.02, 0.04, "← 싸고 정확 = 이상적 (좌상단)", transform=ax.transAxes, fontsize=9, color="gray")
    plt.tight_layout(); p = os.path.join(outdir, "fig4_tradeoff.png"); plt.savefig(p, dpi=150); plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/comparison.csv")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = _load(args.results)
    for f in (fig_cost, fig_detection, fig_wer, fig_tradeoff):
        print("생성:", f(df, args.outdir))


if __name__ == "__main__":
    main()
