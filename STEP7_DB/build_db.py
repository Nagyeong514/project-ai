"""
STEP7_DB 진입점 — 확정된 암묵지 최종 JSON -> Embedding -> Vector DB(Qdrant, 로컬 디스크).

사용:
    .venv/bin/python3 build_db.py                    # config.py의 input_dir 사용
    .venv/bin/python3 build_db.py --input_dir <경로>  # 다른 폴더 지정
    .venv/bin/python3 build_db.py --query "래치 두 개 동시에"   # 적재 후 유사도 검색 스모크 테스트
"""
from __future__ import annotations

import argparse

from preflight import run_preflight

# 무거운 임베딩 모델/벡터DB 클라이언트는 프리플라이트 통과 후에만 import한다
# (STEP3/STEP6과 동일 원칙 — docs/실행전_방어_체크리스트.md).


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=None, help="암묵지 JSON 폴더(기본: config.py의 gold_records)")
    ap.add_argument("--query", default="래치 두 개를 동시에 눌러야 안전하다", help="적재 후 유사도 검색 스모크 테스트 쿼리")
    args = ap.parse_args()

    run_preflight()

    from config import CONFIG
    import vectordb_utils as vdb

    if args.input_dir:
        CONFIG.input_dir = args.input_dir

    print(f"[1/3] 임베딩 모델 로딩: {CONFIG.embedding_model_name} (device={CONFIG.device}, mock={CONFIG.mock_mode})")
    embeddings = vdb.build_embeddings(CONFIG)

    print(f"[2/3] Vector DB 적재: '{CONFIG.input_dir}' -> {CONFIG.qdrant_path}::{CONFIG.qdrant_collection}")
    vectorstore, client, n_docs = vdb.build_or_load_vectorstore(CONFIG, embeddings)

    # 스모크 테스트: "에러 없이 끝남"만으로 통과 처리하지 않는다 — 실제 개수/결과까지 확인한다
    # (docs/실행전_방어_체크리스트.md 5번 원칙).
    count = client.count(collection_name=CONFIG.qdrant_collection, exact=True).count
    print(f"      적재 완료: 이번 실행 문서 {n_docs}건 / 컬렉션 전체 포인트 {count}건")
    assert count > 0, "적재 후 컬렉션이 비어있음 — 적재 실패"

    print(f"[3/3] 유사도 검색 스모크 테스트: 쿼리='{args.query}'")
    hits = vdb.search_similar(vectorstore, args.query, top_k=3)
    assert hits, "검색 결과가 0건 — 적재/인덱싱이 실제로는 실패했을 가능성"
    for text, score, meta in hits:
        print(f"      - score={score:.3f} id={meta.get('id')} routing={meta.get('routing')} : {text}")

    print(f"\n[완료] Vector DB 위치: {CONFIG.qdrant_path} (컬렉션: {CONFIG.qdrant_collection}, 총 {count}건)")


if __name__ == "__main__":
    main()
