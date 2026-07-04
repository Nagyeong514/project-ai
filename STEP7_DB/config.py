# -*- coding: utf-8 -*-
"""
STEP7_DB 설정. 확정된 암묵지 최종 JSON(schema_version 1.3)을 읽어
Embedding + Vector DB(Qdrant, 로컬 디스크 영구 저장)를 만드는 단계.

pydantic BaseModel — 값이 이상하면(예: device 오타) import 시점에 즉시 예외로 죽는다
(STEP3/STEP6과 동일 원칙, docs/실행전_방어_체크리스트.md 참고).
"""
from __future__ import annotations

import os

from pydantic import BaseModel, field_validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(BaseModel):
    # ---------------- 입력 ----------------
    # 2026-07-04: 지금은 STEP6 routing으로 거르지 않고 input_dir 안의 JSON을 전부 적재한다
    # (VLM이 아직 막혀 있어 gold_records 예시로 먼저 벡터 DB 파이프라인 자체를 만드는 단계 —
    # accept/hold/reject 필터링은 나중 과제로 미룸). 각 JSON의 verification.routing 값은
    # 있으면 payload에 그대로 저장해 나중에 필터링 기준으로 쓸 수 있게만 해둔다.
    input_dir: str = os.path.join(BASE_DIR, "gold_records")

    # ---------------- 임베딩 (STEP6과 동일 스택 재사용) ----------------
    embedding_model_name: str = "BAAI/bge-m3"
    device: str = "cpu"  # 문서 9건 임베딩엔 GPU 불필요 — 로그인 노드에서 srun 없이 바로 실행 가능
    mock_mode: bool = False  # True: 더미 임베딩(오프라인 구조 테스트용)

    # ---------------- Vector DB (Qdrant, 로컬 디스크 영구 저장) ----------------
    qdrant_path: str = os.path.join(BASE_DIR, "qdrant_db")
    qdrant_collection: str = "tacit_knowledge"

    # ---------------- 검색(RAG) ----------------
    top_k_default: int = 3
    # 2026-07-04 실측 조정: 처음엔 0.30으로 뒀는데 완전 무관한 질의("오늘 점심 뭐 먹지" 등)도
    # bge-m3 특성상 baseline 유사도가 0.30~0.36대로 나와서 걸러지지 않았다. 실측 결과 무관한
    # 질의(0.30~0.36)와 실제 관련 질의(0.44+) 사이에 뚜렷한 갭이 있어 0.40으로 올림.
    # 이 아래면 "관련 노하우를 찾지 못했다"로 응답(환각 대신 정직한 실패).
    min_similarity_threshold: float = 0.40
    # STEP3 VLM 버그 수정 후 켤 자리(D5) — 기본 False로 지금은 전량 검색 대상.
    accept_only_default: bool = False

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError(f"device는 'cuda' 또는 'cpu'여야 함 (받은 값: {v!r})")
        return v


CONFIG = Config()
