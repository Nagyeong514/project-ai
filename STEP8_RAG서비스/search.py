# -*- coding: utf-8 -*-
"""
RAG 검색 모듈 — 서시은(voice-rag-upload) app.py의 검색 로직
(①질문 임베딩 ②Qdrant 검색 ③유사도 임계값 필터)을 app.py/evaluate.py가
공유해서 쓸 수 있도록 함수로 분리했다.

이식: STEP7_DB의 --accept_only 필터 훅(routing==accept만 검색).
환각 방지 원칙: 임계값 미만인 결과는 여기서 걸러내고, 전부 걸러지면 빈 리스트를
반환한다 — 호출부(app.py)는 빈 리스트를 받으면 LLM을 아예 호출하지 않는다.

단독 실행(스모크 테스트):
    .venv/bin/python3 search.py "질문" [--top_k 3] [--accept_only]
"""
from __future__ import annotations

import json

from preflight import run_preflight

# 무거운 임베딩 모델/DB 클라이언트는 프리플라이트 통과 후에만 import한다.


def load_backend(config):
    """임베딩 모델 + Vector DB 클라이언트 로드(app.py 기동 시 한 번만 호출).

    2026-07-12: config.vector_backend 분기(chroma 기본 / qdrant 롤백)."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.embedding_model_name, device=config.device)
    if config.vector_backend == "chroma":
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=config.chroma_path, settings=Settings(anonymized_telemetry=False))
    else:
        from qdrant_client import QdrantClient
        client = QdrantClient(path=config.qdrant_path)
    return embedder, client


def _accept_filter():
    """STEP7_DB search_tacit.py의 _build_accept_filter()와 동일 개념.
    이 폴더는 payload가 플랫 구조(langchain_qdrant의 metadata.* 중첩이 아님)이므로
    key는 'routing' 그대로."""
    from qdrant_client.http import models as qmodels

    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="routing", match=qmodels.MatchValue(value="accept"))]
    )


def search(embedder, client, config, query: str, top_k: int | None = None,
           accept_only: bool | None = None) -> list[dict]:
    """신입 질의 -> bge-m3 인코딩 -> Qdrant 검색 -> 임계값 필터.

    반환: [{"id","similarity","routing","confidence","entry"}], 유사도 높은 순.
    전부 임계값 미만이면 빈 리스트(환각 방지 — LLM 호출 여부는 호출부가 결정).
    """
    top_k = config.top_k_default if top_k is None else top_k
    accept_only = config.accept_only_default if accept_only is None else accept_only

    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    # 2026-07-12 백엔드 분기. 두 경로 모두 (payload dict, cosine 유사도)로 정규화해
    # 아래 공통 조립부를 태운다 — 반환 형태는 전환 전과 바이트 수준 동일.
    if config.vector_backend == "chroma":
        collection = client.get_collection(config.qdrant_collection)
        res = collection.query(
            query_embeddings=[q_emb], n_results=top_k,
            where={"routing": "accept"} if accept_only else None,
            include=["metadatas", "distances"],
        )
        # chroma cosine distance = 1 - cos_sim → 유사도 복원(STEP7 어댑터와 동일 규칙)
        hits = [(meta, 1.0 - float(dist))
                for meta, dist in zip(res["metadatas"][0], res["distances"][0])]
    else:
        kwargs = {"limit": top_k}
        if accept_only:
            kwargs["query_filter"] = _accept_filter()
        points = client.query_points(config.qdrant_collection, query=q_emb, **kwargs).points
        hits = [(p.payload, p.score) for p in points]

    retrieved = []
    for payload, score in hits:
        if score < config.min_similarity_threshold:
            continue
        retrieved.append(
            {
                "id": payload["doc_id"],
                "similarity": round(score, 3),
                "routing": payload.get("routing"),
                "confidence": payload.get("confidence"),
                "entry": json.loads(payload["raw_json"]),
            }
        )
    return retrieved


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--accept_only", action="store_true", default=None)
    args = ap.parse_args()

    run_preflight()

    from config import CONFIG

    embedder, client = load_backend(CONFIG)
    print(f"[검색] collection={CONFIG.qdrant_collection} top_k={args.top_k or CONFIG.top_k_default} "
          f"threshold={CONFIG.min_similarity_threshold} accept_only={args.accept_only or CONFIG.accept_only_default}")

    results = search(embedder, client, CONFIG, args.query, top_k=args.top_k, accept_only=args.accept_only)
    if not results:
        print("관련 암묵지를 찾지 못했어요. (임계값 미만이거나 검색 결과 0건)")
    for r in results:
        print(f"\n- sim={r['similarity']}  id={r['id']}  routing={r['routing']}  confidence={r['confidence']}")
        print(f"  task: {r['entry']['metadata']['task']}")
        print(f"  insight: {r['entry']['knowledge']['tacit_insight']}")
