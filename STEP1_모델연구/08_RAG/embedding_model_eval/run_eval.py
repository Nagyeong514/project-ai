# -*- coding: utf-8 -*-
"""
RAG embedding model 비교 평가 실행 스크립트.

방식: in-memory cosine similarity (Qdrant 미사용).
  - chunk 9건 규모라 벡터DB 서버/컬렉션 없이 numpy로 전량 cosine을 바로 계산하는 게
    더 단순하고 재현 가능하다. Qdrant를 쓰더라도 결과(순위)는 동일하다 — Qdrant의
    Distance.COSINE도 결국 정규화된 벡터의 내적이기 때문. 모델별 벡터를 섞지 않도록
    모델마다 별도로 전체 chunk를 인코딩하고 그 모델의 벡터로만 검색한다.
  - 같은 이유로 Qdrant collection을 만든다면 모델별로 분리해야 한다는 요구사항도
    이 방식에서는 "모델별로 별도 in-memory 행렬을 쓴다"로 동일하게 지켜진다.

모델별 필수 사항:
  - 문서 chunk 임베딩과 query 임베딩은 반드시 같은 모델로 인코딩한다(모델 간 벡터 혼합 금지).
  - intfloat/multilingual-e5-base는 공식 사용법대로 "query: "/"passage: " 프리픽스를 붙인다
    (프리픽스 없이 쓰면 성능이 저하되는 것으로 알려진 E5 계열의 공식 요구사항).
  - BAAI/bge-m3, paraphrase-multilingual-MiniLM-L12-v2는 프리픽스 불필요.

실행 (project-ai/STEP8_RAG서비스/.venv 재사용, GPU 노드에서):
    ../../../STEP8_RAG서비스/.venv/bin/python3 run_eval.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from eval_data import QUERIES, load_chunks

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(OUT_DIR, "results_raw.json")

MODELS = [
    {
        "key": "bge_m3",
        "name": "BAAI/bge-m3",
        "query_prefix": "",
        "doc_prefix": "",
    },
    {
        "key": "e5_base",
        "name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    {
        "key": "minilm",
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "doc_prefix": "",
    },
]

TOP_K_MAX = 5


def cosine_topk(query_vec: np.ndarray, doc_matrix: np.ndarray, k: int) -> list[int]:
    """정규화된 벡터 기준 내적 = cosine similarity. 인덱스를 유사도 내림차순으로 반환."""
    scores = doc_matrix @ query_vec
    order = np.argsort(-scores)[:k]
    return order.tolist(), scores


def dcg_at_k(rank: int | None, k: int) -> float:
    """정답이 1개뿐인 binary relevance 기준 DCG@k. rank는 1-based, 없으면 None."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_model(model_info: dict, chunks: list[dict], queries: list[tuple[str, str, str]]) -> dict:
    print(f"\n=== {model_info['name']} 로딩 ===")
    t_load_start = time.time()
    model = SentenceTransformer(model_info["name"], device="cpu")
    t_load = time.time() - t_load_start
    dim = model.get_sentence_embedding_dimension()
    print(f"    로딩 완료 ({t_load:.1f}s), embedding dimension={dim}")

    doc_texts = [model_info["doc_prefix"] + c["text"] for c in chunks]
    doc_ids = [c["id"] for c in chunks]
    doc_matrix = model.encode(doc_texts, normalize_embeddings=True, convert_to_numpy=True)

    per_query = []
    latencies = []
    for query, gold_id, qtype in queries:
        q_text = model_info["query_prefix"] + query
        t0 = time.time()
        q_vec = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        order, scores = cosine_topk(q_vec, doc_matrix, TOP_K_MAX)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        ranked_ids = [doc_ids[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]
        gold_rank = ranked_ids.index(gold_id) + 1 if gold_id in ranked_ids else None
        if gold_rank is None:
            # top-5 밖이면 전체 순위를 다시 계산(9건 전부이므로 항상 찾을 수 있음)
            full_order = np.argsort(-scores)
            full_ranked_ids = [doc_ids[i] for i in full_order]
            gold_rank = full_ranked_ids.index(gold_id) + 1

        per_query.append(
            {
                "query": query,
                "qtype": qtype,
                "gold_id": gold_id,
                "gold_rank": gold_rank,
                "top5_ids": ranked_ids,
                "top5_scores": ranked_scores,
                "latency_sec": elapsed,
            }
        )

    n = len(queries)
    hit_at = {k: sum(1 for r in per_query if r["gold_rank"] <= k) / n for k in (1, 3, 5)}
    # 정답 chunk가 query당 1개뿐이므로 Recall@k == Hit@k (분모=관련문서 총수=1)
    recall_at = {k: hit_at[k] for k in (3, 5)}
    mrr = sum(1.0 / r["gold_rank"] for r in per_query) / n
    ndcg_at = {k: sum(dcg_at_k(r["gold_rank"], k) for r in per_query) / n for k in (3, 5)}
    avg_latency_ms = (sum(latencies) / n) * 1000

    return {
        "model": model_info["name"],
        "dim": dim,
        "load_time_sec": round(t_load, 2),
        "hit_at_1": hit_at[1],
        "hit_at_3": hit_at[3],
        "hit_at_5": hit_at[5],
        "recall_at_3": recall_at[3],
        "recall_at_5": recall_at[5],
        "mrr": mrr,
        "ndcg_at_3": ndcg_at[3],
        "ndcg_at_5": ndcg_at[5],
        "avg_latency_ms": round(avg_latency_ms, 2),
        "per_query": per_query,
    }


def main() -> None:
    chunks = load_chunks()
    print(f"chunk {len(chunks)}건, query {len(QUERIES)}건 로드 완료")

    all_results = {}
    for model_info in MODELS:
        try:
            result = evaluate_model(model_info, chunks, QUERIES)
            all_results[model_info["key"]] = result
            print(
                f"    Hit@1={result['hit_at_1']:.2f} Hit@3={result['hit_at_3']:.2f} "
                f"Hit@5={result['hit_at_5']:.2f} MRR={result['mrr']:.3f} "
                f"nDCG@3={result['ndcg_at_3']:.3f} latency={result['avg_latency_ms']}ms"
            )
        except Exception as e:
            print(f"    [FAIL] {model_info['name']}: {e}")
            all_results[model_info["key"]] = {"model": model_info["name"], "error": str(e)}

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
