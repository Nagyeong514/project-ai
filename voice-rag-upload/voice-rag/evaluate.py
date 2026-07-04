# -*- coding: utf-8 -*-
"""
하이퍼파라미터 비교 실험용 평가 스크립트.
dummy 질문 → 정답 문서 ID를 정의해두고, 검색이 top-k 안에 정답을 넣는 비율(Recall@k)을 측정.

사용법:
    python evaluate.py --collection tacit_knowledge_full --top-k 3
    python evaluate.py --collection tacit_knowledge_insight --top-k 3
    → 결과 비교해서 보고서 표에 기입
"""
import argparse

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 평가 세트: (질문, 정답 문서 id) — 팀에서 자유롭게 추가
EVAL_SET = [
    ("컴퓨터가 부팅이 안 돼요 뭐부터 확인해요?", "tk_dell7920_mem_boot_001"),
    ("전원은 들어오는데 화면이 안 나와요", "tk_dell7920_mem_boot_001"),
    ("LED가 깜빡거리는데 무슨 뜻이에요?", "tk_dell7920_mem_boot_001"),
    ("CPU 꽂을 때 방향 어떻게 맞춰요?", "tk_dell7920_cpu_seat_002"),
    ("CPU가 잘 안 들어가는데 눌러도 돼요?", "tk_dell7920_cpu_seat_002"),
    ("히트싱크 나사 조이는 순서 있어요?", "tk_dell7920_heatsink_003"),
    ("쿨러 달았는데 온도가 너무 높아요", "tk_dell7920_heatsink_003"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="tacit_knowledge_full")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--db-path", default="qdrant_db")
    args = ap.parse_args()

    embedder = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(path=args.db_path)

    hits = 0
    for question, gold_id in EVAL_SET:
        emb = embedder.encode([question], normalize_embeddings=True)[0].tolist()
        points = client.query_points(args.collection, query=emb, limit=args.top_k).points
        ids = [p.payload["doc_id"] for p in points]
        ok = gold_id in ids
        hits += ok
        top_sim = round(points[0].score, 3)
        print(f"{'O' if ok else 'X'}  top1_sim={top_sim}  Q: {question}")

    print(f"\nRecall@{args.top_k} = {hits}/{len(EVAL_SET)} = {hits/len(EVAL_SET):.2f}  (collection={args.collection})")


if __name__ == "__main__":
    main()
