"""51624 정답지(필수개념 9개) recall — Qwen3-VL-8B vs InternVL2.5-8B 비교 그래프.
정답지 개념 ↔ 모델 출력 대조 결과를 시각화."""
import os
from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _set_korean_font():
    rcParams["axes.unicode_minus"] = False
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            return


_set_korean_font()

# evaluation/score.py 51624 채점 결과
conds = ["B\n(프레임+말)", "B_txt\n(말만)", "A\n(전체 천장)", "B_vid\n(영상만)"]
qwen = [0.667, 0.834, 0.667, 0.167]
intern = [0.167, 0.250, 0.444, 0.000]

x = range(len(conds))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar([i - w / 2 for i in x], qwen, w, label="Qwen3-VL-8B", color="#2563eb")
b2 = ax.bar([i + w / 2 for i in x], intern, w, label="InternVL2.5-8B", color="#cbd5e1")

for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("개념 recall (정답지 9개 중 적중 비율)", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_xticks(list(x))
ax.set_xticklabels(conds, fontsize=10)
ax.set_title("정답지(필수개념 9개) ↔ 모델 출력 대조 — recall 비교\n51624.mp4 (PC 조립), 1차 정답지·검수 전",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.annotate("⚠ B_vid(영상만)는 정답지가 전사 기반이라 양쪽 낮음 — 모델 비교 결론에만 사용",
            xy=(0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=9, color="#b45309")
fig.tight_layout()
out = os.path.join(ROOT, "results", "viz_answerkey_recall.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
