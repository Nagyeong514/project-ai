"""
STEP7_DB 검색 진입점 — 신입 작업자의 증상 질의로 암묵지 Q&A를 테스트한다.

사용:
    .venv/bin/python3 query_tacit.py "화면이 안 떠요"
    .venv/bin/python3 query_tacit.py "래미 안 잡혀요" --top_k 3
    .venv/bin/python3 query_tacit.py "래치 어떻게 눌러요" --accept_only
"""
from __future__ import annotations

import argparse

from preflight import run_preflight

# 무거운 임베딩 모델/벡터DB 클라이언트는 프리플라이트 통과 후에만 import한다.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="신입 작업자의 증상 질의(자연어)")
    ap.add_argument("--top_k", type=int, default=None, help="기본: config.py의 top_k_default")
    ap.add_argument("--accept_only", action="store_true",
                     help="routing==accept만 검색(D5 훅). 기본은 꺼짐(config.accept_only_default).")
    args = ap.parse_args()

    run_preflight()

    from config import CONFIG
    import vectordb_utils as vdb
    import search_tacit as st

    top_k = args.top_k if args.top_k is not None else CONFIG.top_k_default
    accept_only = args.accept_only or CONFIG.accept_only_default

    print(f"[준비] 임베딩 모델 로딩: {CONFIG.embedding_model_name} (device={CONFIG.device})")
    embeddings = vdb.build_embeddings(CONFIG)
    vectorstore, client, n_docs = vdb.build_or_load_vectorstore(CONFIG, embeddings)
    print(f"       컬렉션 '{CONFIG.qdrant_collection}' 준비 완료 (문서 {n_docs}건)")

    print(f"\n[질의] {args.query!r}  (top_k={top_k}, accept_only={accept_only}, "
          f"min_similarity={CONFIG.min_similarity_threshold})")
    results = st.search_tacit(
        vectorstore, args.query, top_k=top_k,
        accept_only=accept_only, min_similarity=CONFIG.min_similarity_threshold,
    )

    for i, r in enumerate(results, start=1):
        print(f"\n--- 결과 {i} (id={r['id']}, routing={r['routing']}, confidence={r['confidence']}) ---")
        print(r["answer"])
        if r["timestamp_warnings"]:
            print(f"  (timestamp_validity 경고 {len(r['timestamp_warnings'])}건 — 위 로그 참고)")


if __name__ == "__main__":
    main()
