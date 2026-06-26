"""채점 — 1단계는 '내가 쓴 정답지' 대조.

정답지(data/ground_truth/<vid>_answerkey.json)와 VLM 출력을 비교해
구간별 점수를 매기고, B vs C 페어를 만들어 Wilcoxon까지 돌린다.

매칭(추출 항목 ↔ 정답 항목)은 의미 비교가 필요해 LLM-judge가 이상적이다.
지금은 plug 가능한 matcher 구조만 두고, 기본은 어휘 겹침(lexical)으로 둔다.
LLM-judge 도착 시 match_fn 만 갈아끼우면 된다.
"""
from __future__ import annotations
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(s: str) -> set:
    return set(re.findall(r"[가-힣a-zA-Z0-9]+", (s or "").lower()))


def lexical_match(extracted: dict, gt_point: dict, thr: float = 0.25) -> bool:
    """기본 matcher: action+tacit 토큰 자카드 유사도 thr 이상이면 매치."""
    e = _norm(extracted.get("action", "") + " " + extracted.get("tacit", ""))
    g = _norm(gt_point.get("action", "") + " " + gt_point.get("tacit", ""))
    if not e or not g:
        return False
    return len(e & g) / len(e | g) >= thr


def score_segment(extracted_points: list[dict], gt_points: list[dict], match_fn=lexical_match) -> dict:
    """한 구간 채점: 정답 대비 hit(완결성), 추출 중 무근거(환각)."""
    matched_gt = set()
    hallucinated = 0
    for ep in extracted_points:
        hit = next((j for j, gp in enumerate(gt_points)
                    if j not in matched_gt and match_fn(ep, gp)), None)
        if hit is None:
            hallucinated += 1          # 정답에 없는 항목 = 환각 후보
        else:
            matched_gt.add(hit)
    n_gt = len(gt_points)
    recall = len(matched_gt) / n_gt if n_gt else 0.0          # 완결성(누락 역)
    precision = (len(matched_gt) / len(extracted_points)) if extracted_points else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_extracted": len(extracted_points), "n_gt": n_gt,
        "matched": len(matched_gt), "hallucinated": hallucinated,
        "recall": round(recall, 3), "precision": round(precision, 3), "f1": round(f1, 3),
    }


def run(video_id: str, backend: str = "stub", match_fn=lexical_match):
    res_path = os.path.join(ROOT, "results", "raw", f"{video_id}_{backend}.json")
    gt_path = os.path.join(ROOT, "data", "ground_truth", f"{video_id}_answerkey.json")
    with open(res_path, encoding="utf-8") as f:
        data = json.load(f)
    if not os.path.exists(gt_path):
        print(f"[!] 정답지 없음: {gt_path}\n    먼저 정답지를 작성하세요 (scripts/make_answerkey_template.py).")
        return
    with open(gt_path, encoding="utf-8") as f:
        gt = {s["seg_idx"]: s.get("knowledge_points", []) for s in json.load(f)["segments"]}

    rows = []  # 구간별 (seg_idx, condition, metrics)
    for r in data["results"]:
        if r["condition"] == "A" or r["seg_idx"] is None:
            continue  # A는 천장 참고치 — 구간 통계 미포함
        pts = r["output"].get("knowledge_points", [])
        m = score_segment(pts, gt.get(r["seg_idx"], []), match_fn)
        rows.append({"seg_idx": r["seg_idx"], "condition": r["condition"], **m})

    # B vs C 페어 (메인 비교)
    by = {}
    for row in rows:
        by.setdefault(row["seg_idx"], {})[row["condition"]] = row
    pairs = [(by[i]["B"], by[i]["C"]) for i in sorted(by)
             if "B" in by[i] and "C" in by[i]]

    print(f"=== 채점 결과: {video_id} (backend={backend}) ===")
    for metric in ("f1", "recall", "precision", "hallucinated"):
        b = [p[0][metric] for p in pairs]
        c = [p[1][metric] for p in pairs]
        line = f"{metric:>12}:  B={_mean(b):.3f}  C={_mean(c):.3f}  (n={len(pairs)})"
        line += _wilcoxon(b, c)
        print(line)
    if data["meta"]["backend"] == "stub":
        print("\n[주의] backend=stub — 숫자는 흐름 검증용 더미. 가중치 도착 후 backend=qwen 로 재실행.")
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _wilcoxon(b, c):
    try:
        from scipy.stats import wilcoxon
        if len(b) >= 1 and any(x != y for x, y in zip(b, c)):
            stat, p = wilcoxon(b, c)
            return f"   Wilcoxon p={p:.4f}"
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "V01"
    bk = sys.argv[2] if len(sys.argv) > 2 else "stub"
    run(vid, bk)
