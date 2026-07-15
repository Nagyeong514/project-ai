# -*- coding: utf-8 -*-
"""
STEP6 → STEP7 스키마 어댑터 (2026-07-14).

STEP6 출력(`*_verified.json`)의 verification 블록을 STEP7(build_db)이 읽는 형식으로 변환한다.
STEP5→STEP6 의 step6_adapter.py 와 같은 "생산 STEP이 다음 STEP 입력을 맞춰준다" 패턴.

왜 필요한가 (스키마 불일치):
  STEP6 verification = {decision:"accept|hold|reject", confidence:<float>, gate_a_manual_comparison,
                        scores:{reasoning_grounding, step_grounding_ratio, action_reason_consistency,
                        utterance_signal}, ...}
  STEP7 로더가 verification에서 실제로 읽는 것 = routing(str), confidence.score(float) 딱 2개.
  → 그대로 물리면 routing=None(+ confidence가 float라 .get 에러). 이 어댑터가 아래로 매핑:
    decision            → verification.routing
    confidence(float)   → verification.confidence.score
  (id/knowledge/metadata 는 그대로 통과. gate_a/b/c 는 STEP7이 안 읽으므로 참고용으로만 채운다.)

표준 라이브러리만 사용 → 어느 venv로도 실행 가능.

사용:
    python3 step7_adapter.py --input <STEP6 output dir> --output <STEP7 입력 dir> [--accept-only]
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _to_step7(verified: dict) -> dict:
    v = verified.get("verification", {}) or {}
    sc = v.get("scores", {}) or {}
    ga = v.get("gate_a_manual_comparison", {}) or {}
    decision = v.get("decision")
    conf = v.get("confidence")

    out = {
        "id": verified.get("id"),
        "schema_version": verified.get("schema_version"),
        "metadata": verified.get("metadata", {}),
        "knowledge": verified.get("knowledge", {}),
        "verification": {
            "pipeline": "step6->step7-adapter-v1",
            # STEP7이 읽는 필수 2필드
            "routing": decision,
            "confidence": {"score": conf},
            # 참고용(STEP7 로더는 안 읽음): STEP6 원본 점수를 gold_records 형식에 가깝게 보존
            "gate_a": {
                "relation": ga.get("relation"),
                "top1_similarity": ga.get("manual_top_score"),
            },
            "gate_b": {"action_reason_consistency": sc.get("action_reason_consistency")},
            "gate_c": {
                "reasoning_grounding": sc.get("reasoning_grounding"),
                "step_grounding_ratio": sc.get("step_grounding_ratio"),
                "utterance_signal": sc.get("utterance_signal"),
            },
            "weight_track": v.get("weight_track"),
        },
    }
    return out


def run(input_dir: str, output_dir: str, accept_only: bool = False) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*_verified.json")))
    if not files:
        raise FileNotFoundError(f"STEP6 산출물(*_verified.json) 없음: {input_dir}")
    n_in = n_out = n_skip = 0
    for f in files:
        n_in += 1
        with open(f, "r", encoding="utf-8") as fh:
            verified = json.load(fh)
        decision = (verified.get("verification", {}) or {}).get("decision")
        if accept_only and decision != "accept":
            n_skip += 1
            continue
        rec = _to_step7(verified)
        out_path = os.path.join(output_dir, f"{rec['id']}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        n_out += 1
    stats = {"input": n_in, "written": n_out, "skipped": n_skip}
    print(f"[step7_adapter] {stats} → {output_dir}"
          f"{' (accept-only)' if accept_only else ' (routing 태그 보존, 전건)'}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="STEP6 판정 결과를 STEP7(build_db) 입력 형식으로 변환")
    ap.add_argument("--input", required=True, help="STEP6 output 폴더(*_verified.json)")
    ap.add_argument("--output", required=True, help="STEP7 --input_dir 로 넘길 폴더")
    ap.add_argument("--accept-only", action="store_true",
                    help="accept만 적재(기본: 전건 변환 + routing 태그 보존 — STEP7이 검색 시 필터)")
    args = ap.parse_args()
    run(args.input, args.output, accept_only=args.accept_only)
