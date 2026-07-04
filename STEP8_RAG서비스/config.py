# -*- coding: utf-8 -*-
"""
STEP8_RAG서비스 설정.

서시은(voice-rag-upload)의 "환경변수로 하이퍼파라미터 노출" 방식은 그대로 유지하되,
안나경(STEP7_DB)의 원칙대로 pydantic BaseModel로 감싸서 값이 이상하면(오타·범위 밖)
config를 import하는 시점에 즉시 예외로 죽는다(docs/실행전_방어_체크리스트.md 참고).

입력 데이터는 이 폴더 안에 복제하지 않고 STEP7_DB/gold_records를 그대로 참조한다
(요청사항: "다른 파일럿 데이터는 안 쓰고 스텝 7의 JSON을 기준으로").
"""
from __future__ import annotations

import os

from pydantic import BaseModel, field_validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(BaseModel):
    # ---------------- 입력 (STEP7_DB의 예시 JSON을 그대로 참조) ----------------
    input_dir: str = os.path.join(BASE_DIR, "..", "STEP7_DB", "gold_records")

    # ---------------- 임베딩 ----------------
    embedding_model_name: str = "BAAI/bge-m3"
    device: str = "cpu"  # 문서 9건 임베딩엔 GPU 불필요
    # 서시은의 embed-mode 실험 스위치 유지: full(상황+노하우+이유+절차+키워드) vs insight(단독)
    embed_mode: str = os.getenv("EMBED_MODE", "full")

    # ---------------- Vector DB (Qdrant, 로컬 디스크) ----------------
    qdrant_path: str = os.path.join(BASE_DIR, "qdrant_db")
    # 서시은 방식대로 embed_mode별 컬렉션 분리(비교 실험용) — COLLECTION env로 강제 지정도 가능
    qdrant_collection: str = os.getenv("COLLECTION", "")  # ""면 ingest.py가 embed_mode로 자동 결정

    # ---------------- 검색(RAG) — 서시은의 환경변수 스위치 유지 ----------------
    top_k_default: int = int(os.getenv("TOP_K", "3"))
    # 2026-07-04: 서시은 원본 기본값(0.35)은 애매한 질의("오늘 점심 뭐 먹지", top1=0.372)를
    # 못 걸러냄(방어선2=시스템 프롬프트가 겨우 막음). STEP7_DB가 실측으로 검증한 값(무관 질의
    # baseline 0.30~0.36 vs 관련 질의 0.44+)으로 올려 방어선1 자체를 더 확실하게 만듦.
    min_similarity_threshold: float = float(os.getenv("SIM_THRESHOLD", "0.40"))
    # STEP7_DB의 --accept_only 훅과 동일 개념. 기본 꺼짐(진짜 STEP3 데이터 나오기 전까지).
    accept_only_default: bool = os.getenv("ACCEPT_ONLY", "false").lower() == "true"

    # ---------------- LLM (Ollama) ----------------
    # 2026-07-04: 이 환경(team_a2 계정)에 qwen2.5:3b-instruct는 없고 14b-instruct가 이미
    # pull되어 있어(~/.local/ollama/data) 기본값을 14b로 맞춤 — 3b를 쓰려면 LLM_MODEL 환경변수로.
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:14b-instruct")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError(f"device는 'cuda' 또는 'cpu'여야 함 (받은 값: {v!r})")
        return v

    @field_validator("embed_mode")
    @classmethod
    def _validate_embed_mode(cls, v: str) -> str:
        if v not in ("full", "insight"):
            raise ValueError(f"embed_mode는 'full' 또는 'insight'여야 함 (받은 값: {v!r})")
        return v

    @field_validator("min_similarity_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_similarity_threshold는 0.0~1.0 범위여야 함 (받은 값: {v!r})")
        return v

    @field_validator("top_k_default")
    @classmethod
    def _validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"top_k_default는 1 이상이어야 함 (받은 값: {v!r})")
        return v


CONFIG = Config()

# COLLECTION env를 안 줬으면 embed_mode로 컬렉션명 자동 결정(서시은 ingest.py와 동일 규칙)
if not CONFIG.qdrant_collection:
    CONFIG.qdrant_collection = f"tacit_knowledge_{CONFIG.embed_mode}"
