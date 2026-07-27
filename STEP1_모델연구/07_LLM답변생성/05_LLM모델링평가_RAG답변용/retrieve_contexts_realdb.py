# -*- coding: utf-8 -*-
"""
실DB 벤치마크 1단계 — 실데이터(모션 run6 accept 6건)만 격리 Qdrant에 적재하고,
증상형 질의 7개(무관 1개 포함)로 검색해 LLM 벤치마크용 컨텍스트 JSON을 만든다.

실행(로그인 노드, CPU — GPU 불필요):
    /home/ai_user/team_a2/members/안나경/project-ai/STEP7_DB/.venv/bin/python3 retrieve_contexts_realdb.py

설계 결정(2026-07-11, 실DB 벤치마크 v1):
- 공유 qdrant_db(골든 9 + 실데이터 6 혼재)는 건드리지 않는다 — 이 폴더 밑
  qdrant_realonly/ 에 실데이터 6건만 별도 적재(격리 경로라 STEP8 서비스와 무충돌).
- STEP7의 build_or_load_vectorstore()는 로드 시마다 input_dir을 재업서트하므로
  공유 DB를 열면 그 자체가 쓰기 행위가 된다 — 그래서 격리 경로가 필수다.
- 검색 파라미터는 STEP7 확정값 그대로: top_k=3, min_similarity=0.40.
- 컨텍스트는 search_tacit의 [핵심/이유/절차/출처] 조립 결과를 그대로 이어붙인다
  (STEP8 서비스가 LLM에 주는 형태와 동일 계열). 무관 질의는 NOT_FOUND 메시지가
  컨텍스트로 들어가는 게 정상 동작이다(환각 방지 케이스 — 답변 정성평가용).
"""
from __future__ import annotations

import json
import os
import sys

STEP7_DIR = "/home/ai_user/team_a2/members/안나경/project-ai/STEP7_DB"
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

REAL_INPUT_DIR = os.path.join(STEP7_DIR, "input_motion_obs_run6_accept")
QDRANT_PATH = os.path.join(EVAL_DIR, "qdrant_realonly")  # 격리 경로 — 공유 DB 무접촉
QDRANT_COLLECTION = "tacit_realonly"
OUTPUT_FILE = os.path.join(EVAL_DIR, "realdb_contexts.json")

TOP_K = 3
MIN_SIMILARITY = 0.40  # STEP7/STEP8 확정값(2026-07-04)

# 실데이터 6건의 situation에 대응하는 신입 증상형 질의 + 무관 질의 1개.
# expected_record는 검색 품질 확인용 참고 정보일 뿐, 채점에 쓰지 않는다.
QUERIES = [
    {"query_id": "q1_ram_latch_confirm",
     "query": "램을 꽂았는데 제대로 잠겼는지 어떻게 확인하나요?",
     "expected_record": "tk_CLIP1_정상조립과정_004"},
    {"query_id": "q2_gpu_fix_check",
     "query": "그래픽카드가 제대로 고정됐는지 어떻게 점검하나요?",
     "expected_record": "tk_CLIP2_003"},
    {"query_id": "q3_ram_removal",
     "query": "램을 뺄 때 주의할 점이 있나요?",
     "expected_record": "tk_CLIP2_004"},
    {"query_id": "q4_ram_terminal_clean",
     "query": "램 단자가 오염된 것 같은데 어떻게 세척하나요?",
     "expected_record": "tk_CLIP3_재부팅시도 전까지_002"},
    {"query_id": "q5_capacity_check",
     "query": "부팅했는데 메모리 용량이 이상하게 나와요. 뭘 확인해야 하나요?",
     "expected_record": "tk_CLIP4_재부팅및BIOS_006"},
    {"query_id": "q6_channel_check",
     "query": "메모리 채널이 정상인지 어떻게 확인하나요?",
     "expected_record": "tk_CLIP4_재부팅및BIOS_007"},
    {"query_id": "q7_irrelevant",
     "query": "오늘 점심 메뉴 추천해줘",
     "expected_record": None},  # NOT_FOUND가 정상(환각 방지 케이스)
]


def main() -> None:
    sys.path.insert(0, STEP7_DIR)

    from preflight import run_preflight
    run_preflight()

    from config import Config
    import vectordb_utils as vdb
    import search_tacit as st

    cfg = Config(
        input_dir=REAL_INPUT_DIR,
        qdrant_path=QDRANT_PATH,
        qdrant_collection=QDRANT_COLLECTION,
        # 2026-07-12: STEP7 기본 백엔드가 chroma로 전환됨 — 이 스크립트는 실DB 벤치(07-11)
        # 당시 실험 조건(qdrant embedded) 재현용이므로 qdrant로 고정.
        vector_backend="qdrant",
    )

    print(f"[1/3] 임베딩 모델 로딩: {cfg.embedding_model_name} (device={cfg.device})")
    embeddings = vdb.build_embeddings(cfg)

    print(f"[2/3] 실데이터 격리 적재: '{REAL_INPUT_DIR}' -> {QDRANT_PATH}::{QDRANT_COLLECTION}")
    vectorstore, client, n_docs = vdb.build_or_load_vectorstore(cfg, embeddings)
    assert n_docs == 6, f"실데이터는 6건이어야 함 (적재됨: {n_docs}건)"
    print(f"       적재 완료: {n_docs}건")

    print(f"[3/3] 질의 {len(QUERIES)}건 검색 (top_k={TOP_K}, min_similarity={MIN_SIMILARITY})")
    entries = []
    for q in QUERIES:
        results = st.search_tacit(
            vectorstore, q["query"], top_k=TOP_K, min_similarity=MIN_SIMILARITY,
        )
        context_text = "\n\n".join(r["answer"] for r in results)
        retrieved = [{"id": r["id"], "score": round(r["score"], 4)} for r in results]
        print(f"    - {q['query_id']}: hit={[r['id'] for r in retrieved]}")
        entries.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "expected_record": q["expected_record"],
            "retrieved": retrieved,
            "context_text": context_text,
        })

    # 스모크 검증: exit 0 + 빈 결과 방지(체크리스트 5단계)
    assert len(entries) == len(QUERIES), "질의 수와 결과 수 불일치"
    assert all(e["context_text"].strip() for e in entries), "빈 컨텍스트 존재"
    hit_any = [e for e in entries if e["expected_record"] and any(r["id"] for r in e["retrieved"])]
    assert hit_any, "관련 질의 6건이 전부 NOT_FOUND — 임베딩/적재 이상 의심"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {OUTPUT_FILE} ({len(entries)}건)")


if __name__ == "__main__":
    main()
