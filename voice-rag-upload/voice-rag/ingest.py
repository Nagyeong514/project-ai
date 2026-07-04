# -*- coding: utf-8 -*-
"""
6단계 산출물(암묵지 JSON, 스키마 v1.3)을 Vector DB(ChromaDB)에 적재하는 스크립트.

사용법:
    python ingest.py --input data/sample_knowledge.json
    python ingest.py --input data/ --embed-mode full   # 폴더 통째로도 가능

하이퍼파라미터 비교 실험:
    --embed-mode insight  : tacit_insight만 임베딩
    --embed-mode full     : situation + insight + reasoning + steps 전체 임베딩 (기본)
    → 두 컬렉션이 따로 저장되므로 app.py에서 --collection으로 바꿔가며 Recall@3 비교
"""
import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"  # 한국어 강함, 로컬 무료. 비교군: "jhgan/ko-sroberta-multitask"


def build_document(entry: dict, mode: str) -> str:
    """임베딩 대상 텍스트 구성. mode에 따라 실험 비교."""
    k = entry["knowledge"]
    m = entry["metadata"]
    if mode == "insight":
        return k["tacit_insight"]
    steps = " / ".join(s["action"] for s in k.get("diagnostic_steps", []))
    return (
        f"작업: {m['task']}. 상황: {k['situation']}. "
        f"노하우: {k['tacit_insight']} 이유: {k.get('reasoning','')} "
        f"절차: {steps}. 키워드: {', '.join(m.get('keywords', []))}"
    )


def load_entries(input_path: Path) -> list[dict]:
    files = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    entries = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        entries.extend(data if isinstance(data, list) else [data])
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sample_knowledge.json")
    ap.add_argument("--db-path", default="qdrant_db")
    ap.add_argument("--embed-mode", choices=["full", "insight"], default="full")
    args = ap.parse_args()

    entries = load_entries(Path(args.input))
    print(f"[1/3] 암묵지 {len(entries)}건 로드")

    print(f"[2/3] 임베딩 모델 로드: {EMBED_MODEL} (최초 1회 다운로드, 수 분 소요)")
    model = SentenceTransformer(EMBED_MODEL)
    docs = [build_document(e, args.embed_mode) for e in entries]
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)

    client = QdrantClient(path=args.db_path)
    coll_name = f"tacit_knowledge_{args.embed_mode}"
    if client.collection_exists(coll_name):
        client.delete_collection(coll_name)
    client.create_collection(
        coll_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    client.upsert(
        collection_name=coll_name,
        points=[
            PointStruct(
                # Qdrant는 문자열 ID 불가(정수/UUID만) → 원본 id에서 UUID 파생, 원본은 payload에 보존
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, e["id"])),
                vector=embeddings[i].tolist(),
                payload={
                    "doc_id": e["id"],
                    "document": docs[i],
                    "task": e["metadata"]["task"],
                    "video_id": e["metadata"]["source"]["video_id"],
                    "clip_start": e["metadata"]["source"]["clip_start"],
                    "clip_end": e["metadata"]["source"]["clip_end"],
                    "raw_json": json.dumps(e, ensure_ascii=False),  # 원본 전체 보존 → LLM 컨텍스트용
                },
            )
            for i, e in enumerate(entries)
        ],
    )
    print(f"[3/3] 적재 완료 → 컬렉션 '{coll_name}' ({args.db_path}/)")


if __name__ == "__main__":
    main()
