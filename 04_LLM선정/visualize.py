#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 비교 실험 결과 시각화 (계획서 3.4/3.6/6).
입력: results/scores.csv  →  출력: results/viz_*.png
심판=Claude Opus(LLM 5항목) + 기계 형식점수. 가중 Final Score 기준.
"""
import csv, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 한글 폰트 (Noto Sans CJK KR) ──
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(NOTO)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=NOTO).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).parent
ITEMS = ["faithfulness", "accuracy", "usefulness", "codeswitch", "fluency", "format"]
LABELS = ["충실도", "정확성", "유용성", "한·영혼용", "한국어", "형식"]
WEIGHTS = {"faithfulness":0.25,"accuracy":0.20,"usefulness":0.20,
           "codeswitch":0.15,"fluency":0.10,"format":0.10}
MODELS = ["qwen14b", "llama"]
MNAME = {"qwen14b":"Qwen2.5-14B", "llama":"Llama-3.1-8B"}
COLOR = {"qwen14b":"#2563eb", "llama":"#f59e0b"}

# ── 데이터 로드 ──
rows = list(csv.DictReader(open(ROOT/"results/scores.csv")))
by = {m:{} for m in MODELS}
for r in rows:
    by[r["model"]][r["sample_id"]] = {k:float(r[k]) for k in ITEMS}
sids = sorted(by["qwen14b"])

def final(d):  # 가중 Final
    return sum(WEIGHTS[k]*d[k] for k in ITEMS)

item_mean = {m:{k:np.mean([by[m][s][k] for s in sids]) for k in ITEMS} for m in MODELS}
finals    = {m:np.array([final(by[m][s]) for s in sids]) for m in MODELS}

# ════════════════════════════════════════════════════════════════════
# 종합 4분할 대시보드
# ════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 11))
fig.suptitle("암묵지 추출 LLM 비교 — 결과 시각화  (심판=Claude Opus, n=30)",
             fontsize=16, fontweight="bold", y=0.98)

# (a) 항목별 평균 막대
ax = fig.add_subplot(2, 2, 1)
x = np.arange(len(ITEMS)); w = 0.38
for i, m in enumerate(MODELS):
    vals = [item_mean[m][k] for k in ITEMS]
    bars = ax.bar(x + (i-0.5)*w, vals, w, label=MNAME[m], color=COLOR[m])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylim(0, 5.6); ax.set_ylabel("평균 점수 (1~5)")
ax.set_title("(a) 6개 항목별 평균 점수", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)

# (b) 레이더 (6항목)
ax = fig.add_subplot(2, 2, 2, polar=True)
ang = np.linspace(0, 2*np.pi, len(ITEMS), endpoint=False).tolist(); ang += ang[:1]
for m in MODELS:
    vals = [item_mean[m][k] for k in ITEMS]; vals += vals[:1]
    ax.plot(ang, vals, "o-", lw=2, color=COLOR[m], label=MNAME[m])
    ax.fill(ang, vals, alpha=0.12, color=COLOR[m])
ax.set_xticks(ang[:-1]); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylim(0, 5); ax.set_yticks([1,2,3,4,5])
ax.set_title("(b) 항목별 프로파일 (레이더)", fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.12))

# (c) 샘플별 Final Score (대응)
ax = fig.add_subplot(2, 1, 2)
xs = np.arange(len(sids))
for m in MODELS:
    ax.plot(xs, finals[m], "o-", ms=4, color=COLOR[m], label=MNAME[m])
# 승자 음영
qw, ll = finals["qwen14b"], finals["llama"]
ax.fill_between(xs, qw, ll, where=qw>=ll, color=COLOR["qwen14b"], alpha=0.10)
ax.fill_between(xs, qw, ll, where=qw<ll,  color=COLOR["llama"],   alpha=0.18)
# llama가 이긴 샘플 표시
for i, s in enumerate(sids):
    if ll[i] > qw[i]:
        ax.annotate(s, (i, ll[i]), textcoords="offset points", xytext=(0,8),
                    ha="center", fontsize=8, color=COLOR["llama"], fontweight="bold")
ax.axhline(np.mean(qw), ls="--", lw=1, color=COLOR["qwen14b"], alpha=0.6)
ax.axhline(np.mean(ll), ls="--", lw=1, color=COLOR["llama"], alpha=0.6)
ax.set_xticks(xs); ax.set_xticklabels(sids, rotation=90, fontsize=7)
ax.set_ylabel("Final Score (가중합)"); ax.set_ylim(1, 5.2)
nwin = int(np.sum(qw>ll)); lwin = int(np.sum(ll>qw)); tie = int(np.sum(qw==ll))
ax.set_title(f"(c) 샘플별 Final Score 대응 비교 — Qwen 우세 {nwin} / Llama 우세 {lwin} / 동률 {tie}  "
             f"(주황 음영=Llama 우세 구간)", fontweight="bold")
ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT/"results/viz_dashboard.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("저장:", out)

# ════════════════════════════════════════════════════════════════════
# Final Score 분포 (박스+산점)
# ════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(7, 5.5))
data = [finals[m] for m in MODELS]
bp = ax.boxplot(data, patch_artist=True, widths=0.5, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
for patch, m in zip(bp["boxes"], MODELS):
    patch.set_facecolor(COLOR[m]); patch.set_alpha(0.45)
for i, m in enumerate(MODELS, 1):
    jitter = (np.random.RandomState(0).rand(len(sids))-0.5)*0.12
    ax.scatter(np.full(len(sids), i)+jitter, finals[m], color=COLOR[m], s=22, alpha=0.7, zorder=3)
ax.set_xticks([1,2]); ax.set_xticklabels([MNAME[m] for m in MODELS])
ax.set_ylabel("Final Score (가중합, 1~5)")
ax.set_title("Final Score 분포 (박스=사분위, ◇=평균, n=30)\npaired t-test: Δ=+0.91, p<0.0001",
             fontweight="bold")
ax.grid(axis="y", alpha=0.3); ax.set_ylim(1, 5.2)
fig2.tight_layout()
out2 = ROOT/"results/viz_distribution.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print("저장:", out2)
print("Qwen 평균 Final:", round(np.mean(finals['qwen14b']),3),
      "| Llama 평균 Final:", round(np.mean(finals['llama']),3))
