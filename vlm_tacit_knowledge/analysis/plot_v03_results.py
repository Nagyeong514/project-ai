"""연구 ④ 1차 실험(V03) 시각화.

수치는 raw 결과 JSON(results/raw/V03_qwen.json)을 evaluation/score.py 채점
로직으로 직접 재채점해 산출한다(보고서 전사 X → 전사 오류 가능성 차단).
참고 집계값(2026-06-27 보고서):
    B_txt(말만)=0.738  B(프레임+말)=0.488  B_vid(영상만)=0.048  A(천장)=0.273
    파싱: 20 정상 / 2 복구 / 0 에러

출력: results/figures/fig1_ablation_recall.png, fig2_parse_reliability.png
실행: <anaconda>/python.exe -m analysis.plot_v03_results   (또는 직접 실행)
"""
from __future__ import annotations
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figures")

sys.path.insert(0, ROOT)
from evaluation.score import _flatten, _mean, score_one  # noqa: E402


def compute(video_id: str = "V03", backend: str = "qwen"):
    """raw JSON을 score.py 로직으로 재채점 → (RECALL, A천장, PARSE)."""
    data = json.load(open(os.path.join(ROOT, "results/raw", f"{video_id}_{backend}.json"),
                          encoding="utf-8"))
    gt = {s["seg_idx"]: s for s in
          json.load(open(os.path.join(ROOT, "data/ground_truth", f"{video_id}_answerkey.json"),
                         encoding="utf-8"))["segments"]}

    by_cond: dict[str, dict[int, float]] = {}
    a_ceiling = 0.0
    parse = {"정상": 0, "복구": 0, "에러": 0}
    for r in data["results"]:
        out = r["output"]
        if out.get("_error") or not out.get("knowledge_points"):
            parse["에러"] += 1
        elif out.get("_recovered"):
            parse["복구"] += 1
        else:
            parse["정상"] += 1

        text = _flatten(out)
        if r["condition"] == "A":
            all_kw = [c for s in gt.values() for c in s["keywords"]]
            a_ceiling = score_one(text, all_kw)["recall"]
            continue
        s = gt.get(r["seg_idx"])
        if not s:
            continue
        by_cond.setdefault(r["condition"], {})[r["seg_idx"]] = score_one(text, s["keywords"])["recall"]

    recall = {c: round(_mean([segs[i] for i in sorted(segs)]), 3) for c, segs in by_cond.items()}
    return recall, a_ceiling, parse


RECALL, A_CEILING, PARSE = compute()


def _set_korean_font() -> None:
    rcParams["axes.unicode_minus"] = False
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # 리눅스 서버
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",                              # Windows
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            return


def fig1_ablation() -> str:
    conds = ["B_txt", "B", "B_vid"]
    labels = ["말(STT)만", "프레임+말", "영상(프레임)만"]
    vals = [RECALL[c] for c in conds]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(labels, vals, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=1.2)

    # A 천장 참조선 (척도 달라 직접비교 불가 → 점선 + 주석)
    ax.axhline(A_CEILING, ls="--", lw=1.3, color="#888888", zorder=2)
    ax.text(2.42, A_CEILING + 0.012, f"A(전체영상 천장) {A_CEILING:.3f}",
            ha="right", va="bottom", fontsize=9, color="#666666")

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 0.85)
    ax.set_ylabel("개념 키워드 recall", fontsize=12)
    ax.set_title("연구 ④ V03 — 입력 조건별 암묵지 recall (n=1 파일럿)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # 핵심 발견 주석: 프레임을 더하면 오히려 떨어짐
    ax.annotate("", xy=(1, RECALL["B"] + 0.03), xytext=(0, RECALL["B_txt"] + 0.03),
                arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.6))
    ax.text(0.5, 0.80, "프레임 추가 -> 오히려 -0.250", ha="center",
            fontsize=10.5, color="#c1121f", fontweight="bold")

    fig.text(0.5, 0.015,
             "주의: B_vid=0.048은 '영상에 정보 없음'이 아니라 8GB 제약(8장·저화질)으로 모델이 못 읽은 것 · A는 채점 척도(분모)가 달라 직접비교 불가",
             ha="center", fontsize=8, color="#777777")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(FIG_DIR, "fig1_ablation_recall.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2_parse() -> str:
    labels = list(PARSE.keys())
    vals = list(PARSE.values())
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    # 0인 항목은 도넛에서 빠지므로 색만 맞춰 필터
    shown = [(l, v, c) for l, v, c in zip(labels, vals, colors) if v > 0]
    sl, sv, sc = zip(*shown)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    wedges, _ = ax.pie(sv, colors=sc, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    total = sum(vals)
    ax.text(0, 0.08, f"{total}", ha="center", va="center",
            fontsize=30, fontweight="bold")
    ax.text(0, -0.18, "출력", ha="center", va="center", fontsize=12, color="#666")

    legend_lbl = [f"{l}  {v}  ({v / total * 100:.0f}%)" for l, v in zip(labels, vals)]
    ax.legend(wedges, [legend_lbl[i] for i, v in enumerate(vals) if v > 0],
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False, fontsize=10, handlelength=1.1)

    ax.set_title("연구 ④ V03 — 출력 파싱 안정성\n(에러 0 → recall 신뢰 가능)",
                 fontsize=13, fontweight="bold", pad=10)
    fig.text(0.5, 0.02,
             "1차 디버깅 성과: 반복옵션發 JSON 키 변형·언어 드리프트로 19/22 파손 → 원인 분리·수정해 에러 0 복구",
             ha="center", fontsize=8, color="#777777")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(FIG_DIR, "fig2_parse_reliability.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    _set_korean_font()
    print("saved:", fig1_ablation())
    print("saved:", fig2_parse())


if __name__ == "__main__":
    main()
