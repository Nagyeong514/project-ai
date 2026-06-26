"""채점 — 개념 키워드 방식 + ablation 비교.

정답지 keywords = [[동의어,...], ...] (각 안쪽 리스트 = 1개 '필수 개념').
개념 적중 = 그 동의어 중 하나라도 출력 텍스트에 등장.
구간 recall = 적중 개념 / 전체 개념.

ablation:
  B     = 프레임+STT,  B_vid = 프레임만,  B_txt = STT만
  B_txt ≈ B 이고 B_vid 가 낮으면 → 출력이 사실상 STT에서 나옴(영상 기여 작음).
  B > B_txt 이고 B_vid 가 의미있으면 → 영상이 실제로 기여.

주의(키워드 방식 한계): 동의어 누락=가짜 0점, 부분문자열=가짜 적중 가능.
의미 판단이 필요하면 LLM-judge로 교체(match 함수만 교체 가능 구조).
"""
from __future__ import annotations
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flatten(out: dict) -> str:
    if not isinstance(out, dict):
        return str(out)
    parts = []
    for kp in out.get("knowledge_points", []):
        parts += [str(kp.get("action", "")), str(kp.get("tacit", "")), str(kp.get("evidence", ""))]
    if "_raw" in out:
        parts.append(str(out["_raw"]))
    return " ".join(parts)


def concept_hits(text: str, kw_sets: list[list[str]]) -> list[bool]:
    t = text.replace(" ", "").lower()
    return [any(s.replace(" ", "").lower() in t for s in concept) for concept in kw_sets]


def score_one(text, kw_sets):
    hits = concept_hits(text, kw_sets)
    n = len(kw_sets)
    return {"n": n, "hit": sum(hits), "recall": round(sum(hits) / n, 3) if n else 0.0,
            "missed": [i for i, h in enumerate(hits) if not h]}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _wil(b, c):
    try:
        from scipy.stats import wilcoxon
        if len(b) > 1 and any(x != y for x, y in zip(b, c)):
            return f"  (Wilcoxon p={wilcoxon(b, c)[1]:.4f})"
    except Exception:
        pass
    return ""


def run(video_id: str, backend: str = "qwen"):
    data = json.load(open(os.path.join(ROOT, "results/raw", f"{video_id}_{backend}.json"), encoding="utf-8"))
    gt = {s["seg_idx"]: s for s in
          json.load(open(os.path.join(ROOT, "data/ground_truth", f"{video_id}_answerkey.json"), encoding="utf-8"))["segments"]}

    by_cond = {}   # cond -> {seg_idx: recall}
    a_row = None
    for r in data["results"]:
        text = _flatten(r["output"])
        if r["condition"] == "A":
            all_kw = [c for s in gt.values() for c in s["keywords"]]
            a_row = score_one(text, all_kw)
            continue
        s = gt.get(r["seg_idx"])
        if not s:
            continue
        by_cond.setdefault(r["condition"], {})[r["seg_idx"]] = score_one(text, s["keywords"])

    print(f"=== 채점: {video_id} (backend={backend}) ===\n")
    order = ["B", "B_vid", "B_txt", "C"]
    label = {"B": "프레임+STT", "B_vid": "영상만", "B_txt": "STT만", "C": "랜덤창"}
    means = {}
    for cond in order:
        if cond not in by_cond:
            continue
        segs = by_cond[cond]
        rec = [segs[i]["recall"] for i in sorted(segs)]
        means[cond] = _mean(rec)
        print(f"[{cond:5}] {label[cond]:<10} 구간평균 recall={means[cond]:.3f}  "
              f"(구간별 {[segs[i]['recall'] for i in sorted(segs)]})")
    if a_row:
        print(f"\n[A    ] 천장(전체 {a_row['n']}개념)   recall={a_row['recall']:.3f}")

    # ── ablation 해석 (STT 혼입 분리) ──
    if {"B", "B_vid", "B_txt"} <= means.keys():
        print("\n── 영상 vs 발화 기여 (STT 혼입 분리) ──")
        print(f"  STT만(B_txt)={means['B_txt']:.3f}  영상만(B_vid)={means['B_vid']:.3f}  둘다(B)={means['B']:.3f}")
        if means["B"] - means["B_txt"] < 0.05:
            print("  → B ≈ B_txt: 출력이 사실상 STT에서 나옴. 영상 기여 미미(혼입 경고).")
        else:
            print(f"  → 영상이 STT 위에 +{means['B']-means['B_txt']:.3f} 기여.")

    # ── B vs C (C 활성 시) ──
    if "C" in by_cond and "B" in by_cond:
        common = sorted(set(by_cond["B"]) & set(by_cond["C"]))
        b = [by_cond["B"][i]["recall"] for i in common]
        c = [by_cond["C"][i]["recall"] for i in common]
        print(f"\n[B vs C] B={_mean(b):.3f} C={_mean(c):.3f} (n={len(common)}){_wil(b, c)}")

    if data["meta"]["backend"] == "stub":
        print("\n[주의] stub — 더미 출력.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "V03",
        sys.argv[2] if len(sys.argv) > 2 else "qwen")
