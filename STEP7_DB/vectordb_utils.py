# -*- coding: utf-8 -*-
"""
확정된 암묵지 최종 JSON(schema_version 1.3) -> Embedding -> Vector DB(Qdrant, 로컬 디스크).
STEP6_품질검증/embedding_utils.py와 동일 스택(BAAI/bge-m3 + Qdrant + LangChain)을 재사용.

임베딩 대상(2026-07-04, D1): situation + tacit_insight + reasoning 합본.
  "[상황] ... \n[노하우] ... \n[이유] ..." 형태로 합쳐서 임베딩한다 — situation이 신입의
  증상 질의("화면이 안 떠요")와 붙는 접점 역할을 하고, tacit_insight+reasoning이 의미를
  준다. tacit_insight 단독 임베딩이었을 때는 증상 질의 유사도가 낮았음(실측 0.27대).
  diagnostic_steps.action은 절차 나열이라 노이즈가 커서 임베딩엔 안 넣고 payload로만 보관.
"""
from __future__ import annotations

import glob
import json
import os
import uuid
from typing import Any, Dict, List, Tuple

from langchain_core.embeddings import Embeddings

# 재적재해도 같은 id면 같은 포인트를 덮어쓰게(중복 방지) 고정 네임스페이스로 UUID5 생성.
_QDRANT_ID_NAMESPACE = uuid.UUID("7b1b6e2e-6f6b-4a0b-9c1d-3a9d1c0f7e10")


class MockEmbeddings(Embeddings):
    """오프라인/구조 테스트용 더미 임베딩. 실제 적재에는 쓰지 말 것."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [b / 255.0 for b in h[: self.dim]]
        while len(vec) < self.dim:
            vec.append(0.0)
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def build_embeddings(config):
    """BGE-M3 임베딩 로더. mock_mode면 더미로 대체."""
    if config.mock_mode:
        return MockEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": config.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _point_id(tacit_id: str) -> str:
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, tacit_id))


def load_candidate_documents(input_dir: str):
    """input_dir 안의 암묵지 최종 JSON을 전부 읽어 (Document, point_id) 리스트로 변환.

    스키마와 안 맞는 파일(필수 필드 누락 등)은 건너뛰고 경고만 출력한다 — 적재 중 하나
    잘못됐다고 나머지까지 막히면 안 되므로(STEP3/STEP6과 동일한 "관용적 실패" 원칙).
    """
    from langchain_core.documents import Document

    docs = []
    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            tacit_id = data["id"]
            knowledge = data["knowledge"]
            tacit_insight = knowledge["tacit_insight"]
            situation = knowledge.get("situation", "")
            reasoning = knowledge.get("reasoning", "")
            metadata_block = data.get("metadata", {})
            source = metadata_block.get("source", {})
            verification = data.get("verification", {})

            payload: Dict[str, Any] = {
                "id": tacit_id,
                "schema_version": data.get("schema_version"),
                "source_file": os.path.basename(path),
                "equipment": metadata_block.get("equipment"),
                "task": metadata_block.get("task"),
                "keywords": metadata_block.get("keywords", []),
                "scenario_id": metadata_block.get("scenario_id"),
                "scenario_title": metadata_block.get("scenario_title"),
                "video_id": source.get("video_id"),
                "clip_start": source.get("clip_start"),
                "clip_end": source.get("clip_end"),
                "situation": situation,
                "tacit_insight": tacit_insight,
                "reasoning": reasoning,
                "reasoning_origin": knowledge.get("reasoning_origin"),
                "diagnostic_steps": knowledge.get("diagnostic_steps", []),
                # STEP6 routing/confidence는 있으면 그대로 저장(지금은 필터링에 안 씀 — config.py 주석 참고).
                "routing": verification.get("routing"),
                "confidence": (verification.get("confidence") or {}).get("score"),
            }
            # 2026-07-04(D1): tacit_insight 단독 임베딩은 신입의 "증상 언어" 질의와 유사도가
            # 낮게 나왔다(예: 0.27). situation(증상 질의와 붙는 접점) + tacit_insight + reasoning
            # 합본으로 바꿔 재적재한다. diagnostic_steps.action은 절차 나열이라 노이즈가 커서
            # 임베딩엔 안 넣고 payload로만 보관(D3 답변 조립용).
            embedding_text = f"[상황] {situation}\n[노하우] {tacit_insight}\n[이유] {reasoning}"
            docs.append((Document(page_content=embedding_text, metadata=payload), _point_id(tacit_id)))
        except (KeyError, json.JSONDecodeError) as e:
            print(f"      [WARN] '{path}' 건너뜀 — 스키마 불일치: {e}")
    return docs


def build_or_load_vectorstore(config, embeddings):
    """documents를 chunk 없이(tacit_insight는 짧은 한 문장) 그대로 Qdrant에 upsert."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    docs_and_ids = load_candidate_documents(config.input_dir)
    if not docs_and_ids:
        raise FileNotFoundError(
            f"'{config.input_dir}'에서 유효한 암묵지 JSON을 하나도 못 읽었습니다."
        )
    docs = [d for d, _ in docs_and_ids]
    ids = [i for _, i in docs_and_ids]

    # path=로 로컬 디스크 영구 저장(서버 프로세스 불필요, sqlite 유사 — 동시에 한 프로세스만 접근 가능)
    client = QdrantClient(path=config.qdrant_path)
    sample_vec = embeddings.embed_query("dimension_probe")
    dim = len(sample_vec)
    existing = [c.name for c in client.get_collections().collections]
    if config.qdrant_collection not in existing:
        client.create_collection(
            collection_name=config.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.qdrant_collection,
        embedding=embeddings,
    )
    vectorstore.add_documents(docs, ids=ids)
    return vectorstore, client, len(docs)


def search_similar(vectorstore, query: str, top_k: int = 3) -> List[Tuple[str, float, dict]]:
    """(tacit_insight, similarity, payload) 리스트. score는 cosine 유사도(높을수록 유사)."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, score in results:
        similarity = score if score <= 1.0 else 1.0 / (1.0 + score)
        out.append((doc.page_content, float(similarity), doc.metadata))
    return out
