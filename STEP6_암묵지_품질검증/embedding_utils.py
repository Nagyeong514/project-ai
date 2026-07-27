# -*- coding: utf-8 -*-
"""
[게이트 A] Manual Comparison 용 RAG.
- 임베딩: BAAI/bge-m3 (오픈소스, langchain_huggingface)
- 벡터DB: ChromaDB (in-memory/Ephemeral, 매 실행 재색인) — 2026-07-14 Qdrant에서 전환.
          롤백용 qdrant 백엔드 코드 보존(config.vector_backend).
- 체인 구성: LangChain
"""
import glob
import os
from typing import List, Tuple


from langchain_core.embeddings import Embeddings


class MockEmbeddings(Embeddings):
    """오프라인/테스트용 더미 임베딩 (mock_mode=True 일 때만 사용). 실제 검증에는 사용하지 말 것."""

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
    """BGE-M3 임베딩 로더. mock_mode 면 더미로 대체."""
    if config.mock_mode:
        return MockEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": config.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_manual_documents(manual_dir: str):
    """manual_dir 안의 .txt / .md 파일들을 읽어 LangChain Document 리스트로 반환."""
    from langchain_core.documents import Document

    docs = []
    for path in sorted(glob.glob(os.path.join(manual_dir, "*.txt"))) + sorted(
        glob.glob(os.path.join(manual_dir, "*.md"))
    ):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))
    return docs


class ChromaManualAdapter:
    """Chroma 컬렉션을 기존 호출부(search_manual)가 기대하는
    `similarity_search_with_score(query, k)` 인터페이스로 감싼다.

    2026-07-14 전환(STEP7 ChromaVectorStoreAdapter와 동일 패턴):
    - score: Chroma cosine `distance = 1 - cos_sim` → `1 - distance`로 유사도 복원
      (bge-m3 정규화 벡터라 Qdrant cosine 점수와 동일값 — manual_similarity_threshold 그대로).
    - 매뉴얼 chunk의 metadata(source 파일명)는 스칼라라 그대로 저장/복원한다.
    """

    def __init__(self, collection, embeddings):
        self.collection = collection
        self.embeddings = embeddings

    def similarity_search_with_score(self, query: str, k: int = 3):
        from langchain_core.documents import Document

        qvec = self.embeddings.embed_query(query)
        res = self.collection.query(
            query_embeddings=[qvec], n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append((Document(page_content=text, metadata=dict(meta or {})), 1.0 - float(dist)))
        return out


def _chunk_manual(config):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    raw_docs = load_manual_documents(config.manual_dir)
    if not raw_docs:
        raise FileNotFoundError(
            f"'{config.manual_dir}' 에 매뉴얼 문서(.txt/.md)가 없습니다. "
            f"Gate A(Manual Comparison)를 실행하려면 최소 1개 이상의 매뉴얼 파일이 필요합니다."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.manual_chunk_size,
        chunk_overlap=config.manual_chunk_overlap,
    )
    return splitter.split_documents(raw_docs)


def _build_chroma_manual(config, embeddings, chunks):
    import chromadb
    from chromadb.config import Settings

    # 매뉴얼은 소규모 + 매 실행 재색인이라 in-memory(Ephemeral)로 둔다.
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        config.chroma_collection, metadata={"hnsw:space": "cosine"})
    vecs = embeddings.embed_documents([c.page_content for c in chunks])
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=vecs,
        documents=[c.page_content for c in chunks],
        metadatas=[{"source": c.metadata.get("source", "")} for c in chunks],
    )
    return ChromaManualAdapter(collection, embeddings)


def _build_qdrant_manual(config, embeddings, chunks):
    """롤백용(vector_backend='qdrant'). 2026-07-14 Chroma 전환 전 원본 코드 보존."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    client = QdrantClient(location=config.qdrant_location)
    dim = len(embeddings.embed_query("dimension_probe"))
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
    vectorstore.add_documents(chunks)
    return vectorstore


def build_manual_vectorstore(config, embeddings):
    """매뉴얼 문서를 chunk -> embedding -> 벡터DB(config.vector_backend)로 적재.

    2026-07-14: 기본 chroma. 롤백은 config.vector_backend='qdrant'(코드 보존)."""
    chunks = _chunk_manual(config)
    backend = getattr(config, "vector_backend", "chroma")
    if backend == "qdrant":
        return _build_qdrant_manual(config, embeddings, chunks)
    return _build_chroma_manual(config, embeddings, chunks)


def search_manual(vectorstore, query: str, top_k: int) -> List[Tuple[str, float, dict]]:
    """(text, similarity_score, metadata) 리스트. score는 cosine 유사도(높을수록 유사).

    Chroma 어댑터는 (1 - distance)로 이미 유사도를 반환한다. 아래 방어식은 혹시
    백엔드가 거리(>1.0)를 그대로 주는 경우를 대비한 보정(정상 경로는 그대로 통과)."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, score in results:
        similarity = score if score <= 1.0 else 1.0 / (1.0 + score)
        out.append((doc.page_content, float(similarity), doc.metadata))
    return out
