"""정답지 채점기 — output/tacit_json/*.tacit.json 을 정답지 9건 대비 회수율로 채점.

사용: python eval/score.py [--config config.yaml] [--key eval/answer_key.yaml]

판정은 조잡한 키워드 매칭이다(그룹 안 키워드 전부 포함 → 회수). 정밀 판별이 아니라
"파이프라인 개선이 정답 회수를 늘렸나 줄였나"를 실행마다 같은 잣대로 비교하는 회귀 지표다.
★ 이 스크립트/정답지는 평가 전용 — 프롬프트에 절대 주입 금지(정답 leak).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from core.config import load_cfg  # noqa: E402
from core.paths import Contract, clip_id  # noqa: E402


def candidate_text(cand: dict) -> str:
    k = cand.get("knowledge", {})
    parts = [k.get("situation", ""), k.get("tacit_insight", ""), k.get("reasoning", "")]
    for s in k.get("diagnostic_steps", []) or []:
        parts += [s.get("action", ""), s.get("source_utterance") or ""]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--key", default="eval/answer_key.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    key = yaml.safe_load(Path(args.key).read_text(encoding="utf-8"))
    contract = Contract(cfg.paths.output_dir)

    # clip 라벨(CLIP1 등) → tacit 파일 매핑 (부분 일치)
    docs = {}
    for v in cfg.paths.videos:
        cid = clip_id(v)
        p = contract.tacit(cid)
        if p.exists():
            docs[cid] = json.loads(p.read_text(encoding="utf-8"))

    if not docs:
        print("[eval] tacit_json 산출물 없음 — 파이프라인 먼저 실행")
        sys.exit(1)

    def docs_for(label: str):
        return [(cid, d) for cid, d in docs.items() if label in cid]

    hit, miss, silent_hit, silent_total = 0, 0, 0, 0
    print(f"{'ID':4s} {'클립':7s} {'판정':6s} 항목")
    print("-" * 78)
    for item in key["items"]:
        matched_by = None
        for cid, doc in docs_for(item["clip"]):
            for cand in doc.get("candidates", []):
                text = candidate_text(cand)
                for group in item["any_of"]:
                    if all(kw in text for kw in group):
                        matched_by = (cand.get("id", "?"), group)
                        break
                if matched_by:
                    break
            if matched_by:
                break
        is_silent = item.get("silent", False)
        if is_silent:
            silent_total += 1
        if matched_by:
            hit += 1
            silent_hit += 1 if is_silent else 0
            tag = "회수✓"
            extra = f" ← {matched_by[0]} (키워드 {matched_by[1]})"
        else:
            miss += 1
            tag = "누락✗"
            extra = "  ← 침묵 암묵지(action_only 경로 점검)" if is_silent else ""
        print(f"{item['id']:4s} {item['clip']:7s} {tag:6s} {item['desc'][:44]}{extra}")

    print("-" * 78)
    total = hit + miss
    print(f"[eval] 회수율: {hit}/{total}  (침묵 항목: {silent_hit}/{silent_total})")

    # 특수 검사: CLIP2 conflict
    for sp in key.get("special", []):
        if sp.get("check") == "conflict_true":
            found = any(
                c.get("knowledge", {}).get("conflict") is True
                for cid, d in docs_for(sp["clip"]) for c in d.get("candidates", [])
            )
            state = "conflict=true 존재 ✓" if found else "conflict=true 없음 (촬영본이 대본대로인지 확인)"
            print(f"[eval] {sp['id']} {sp['clip']}: {state}")

    print("\n[eval] 미회수 항목은 (1) VLM 관찰에 그 장면이 있는지"
          " (output/vlm_observations) → (2) 윈도우에 들어갔는지 → (3) LLM 이 버렸는지"
          " 순서로 역추적하면 어느 스테이지 문제인지 갈린다.")


if __name__ == "__main__":
    main()
