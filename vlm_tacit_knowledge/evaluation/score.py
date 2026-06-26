"""채점 — 개념 키워드 방식 (concept_keyword).

정답지(data/ground_truth/<vid>_answerkey.json)의 구간별 keywords 와
VLM 출력 텍스트를 대조한다.

  - keywords = [[동의어,...], [동의어,...], ...]  ← 각 안쪽 리스트 = 1개 '필수 개념'
  - 한 개념은 그 동의어 중 하나라도 출력에 나오면 '적중'.
  - 구간 점수(recall) = 적중 개념 수 / 전체 개념 수.
  - 환각(hallucination)은 키워드 방식으론 약하게만 잡힘(정답지에 없는 새 주장 수는
    의미판단 필요 → LLM-judge 몫). 여기선 recall 중심.

A는 천장 참고치: 영상의 '모든' 개념 대비 적중률(영상 단위)로 따로 본다.
"""
from __future__ import annotations
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _output_text(out: dict) -> str:
    """VLM 출력(dict)에서 채점 대상 텍스트를 평탄화."""
    if not isinstance(out, dict):
        return str(out)
    parts = []
    for kp in out.get("knowledge_points", []):
        parts += [str(kp.get("action", "")), str(kp.get("tacit", "")), str(kp.get("evidence", ""))]
    if "_raw" in out:
        parts.append(str(out["_raw"]))
    return " ".join(parts)


def concept_hits(text: str, keyword_sets: list[list[str]]) -> list[bool]:
    """각 개념(동의어 묶음)이 text에 하나라도 있으면 True."""
    t = text.replace(" ", "")
    return [any(syn.replace(" ", "") in t for syn in concept) for concept in keyword_sets]


def score_segment(text: str, kw_sets: list[list[str]]) -> dict:
    hits = concept_hits(text, kw_sets)
    n = len(kw_sets)
    return {
        "n_concepts": n, "hit": sum(hits),
        "recall": round(sum(hits) / n, 3) if n else 0.0,
        "missed": [i for i, h in enumerate(hits) if not h],
    }


def run(video_id: str, backend: str = "stub"):
    res_path = os.path.join(ROOT, "results", "raw", f"{video_id}_{backend}.json")
    gt_path = os.path.join(ROOT, "data", "ground_truth", f"{video_id}_answerkey.json")
    with open(res_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(gt_path, encoding="utf-8") as f:
        gt = {s["seg_idx"]: s for s in json.load(f)["segments"]}

    seg_rows, a_row = [], None
    for r in data["results"]:
        text = _output_text(r["output"])
        if r["condition"] == "A" or r["seg_idx"] is None:
            # A: 전체 개념 대비 적중 (천장 참고치)
            all_kw = [c for s in gt.values() for c in s["keywords"]]
            a_row = {"condition": "A", **score_segment(text, all_kw)}
            continue
        s = gt.get(r["seg_idx"])
        if not s:
            continue
        m = score_segment(text, s["keywords"])
        seg_rows.append({"seg_idx": r["seg_idx"], "condition": r["condition"],
                         "task": s["task"], **m})

    print(f"=== 채점: {video_id} (backend={backend}, 개념 키워드) ===\n")
    # B 구간별
    bs = sorted([r for r in seg_rows if r["condition"] == "B"], key=lambda x: x["seg_idx"])
    if bs:
        print("[B] 발화 구간별 개념 적중:")
        for r in bs:
            print(f"  구간{r['seg_idx']} {r['task'][:18]:<18} "
                  f"{r['hit']}/{r['n_concepts']} (recall {r['recall']})")
        print(f"  → B 평균 recall: {_mean([r['recall'] for r in bs]):.3f}")
    if a_row:
        print(f"\n[A] 천장 참고치(전체 {a_row['n_concepts']}개념): "
              f"{a_row['hit']}/{a_row['n_concepts']} (recall {a_row['recall']})")

    # B vs C (C 활성화 시)
    cs = {r["seg_idx"]: r for r in seg_rows if r["condition"] == "C"}
    if cs:
        pairs = [(r, cs[r["seg_idx"]]) for r in bs if r["seg_idx"] in cs]
        b = [p[0]["recall"] for p in pairs]
        c = [p[1]["recall"] for p in pairs]
        print(f"\n[B vs C] recall  B={_mean(b):.3f}  C={_mean(c):.3f}  (n={len(pairs)})"
              + _wilcoxon(b, c))

    if data["meta"]["backend"] == "stub":
        print("\n[주의] backend=stub — 더미 출력이라 적중 0 정상. 가중치 후 backend=qwen 재실행.")
    return seg_rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _wilcoxon(b, c):
    try:
        from scipy.stats import wilcoxon
        if any(x != y for x, y in zip(b, c)):
            _, p = wilcoxon(b, c)
            return f"   Wilcoxon p={p:.4f}"
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "V03"
    bk = sys.argv[2] if len(sys.argv) > 2 else "stub"
    run(vid, bk)
