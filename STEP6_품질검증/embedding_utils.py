# -*- coding: utf-8 -*-
"""
[게이트 A] Manual Comparison 용 RAG.
- 임베딩: BAAI/bge-m3 (오픈소스, langchain_huggingface)
- 벡터DB: Qdrant (기본 in-memory, qdrant_location 설정으로 로컬/서버 전환 가능)
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


def build_manual_vectorstore(config, embeddings):
    """매뉴얼 문서를 chunk -> embedding -> Qdrant 컬렉션으로 적재."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

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
    chunks = splitter.split_documents(raw_docs)

    client = QdrantClient(location=config.qdrant_location)
    # 임베딩 차원 확인 후 컬렉션 생성 (이미 있으면 재사용)
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
    vectorstore.add_documents(chunks)
    return vectorstore


def search_manual(vectorstore, query: str, top_k: int) -> List[Tuple[str, float, dict]]:
    """(text, similarity_score, metadata) 리스트. score는 cosine 유사도(높을수록 유사)로 정규화."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, score in results:
        # Qdrant cosine distance 백엔드에 따라 score가 거리/유사도 어느 쪽으로 올 수 있어 방어적으로 처리
        similarity = score if score <= 1.0 else 1.0 / (1.0 + score)
        out.append((doc.page_content, float(similarity), doc.metadata))
    return out
