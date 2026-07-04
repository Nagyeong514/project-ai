# -*- coding: utf-8 -*-
"""
6단계 품질 검증 파이프라인 설정
문서 6-5, 6-6 (트랙 A: 라벨 0~20건, 고정 가중치 / 휴리스틱 threshold) 기준 기본값 사용.
라벨이 50건 이상 쌓이면 track="B"로 바꾸고 weight_b_*, t_high/t_low를 재산정한 값으로 교체하면 됨.

dataclass -> pydantic BaseModel 전환(2026-07-03): 필드 이름/메서드는 전부 그대로라
기존 사용처(llm_utils.py, embedding_utils.py, main.py, graph.py, rules.py)는 무수정.
바뀐 건 "config 값이 이상하면(예: track이 'A'/'B'가 아니거나 device 오타) 이 파일을
import하는 순간 즉시 예외로 죽는다"는 것 — 예전엔 오타가 나도 조용히 통과하고
한참 뒤 diagnostic_steps 계산 단계에서야 이상한 값으로 죽거나 조용히 틀린 결과를 냈음.
"""
from __future__ import annotations

import os
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(BaseModel):
    # ---------------- 경로 ----------------
    input_dir: str = os.path.join(BASE_DIR, "input")
    output_dir: str = os.path.join(BASE_DIR, "output")
    manual_dir: str = os.path.join(BASE_DIR, "manuals")

    # ---------------- 모델 (전부 오픈소스, API 미사용) ----------------
    llm_model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    embedding_model_name: str = "BAAI/bge-m3"
    device: str = "cuda"  # GPU 없으면 "cpu" 로 변경
    llm_max_new_tokens: int = 800
    llm_temperature: float = 0.0  # judge 용이므로 결정론적으로

    # ---------------- Qdrant ----------------
    qdrant_location: str = ":memory:"  # 로컬 디스크 사용시 예: os.path.join(BASE_DIR, "qdrant_db")
    qdrant_collection: str = "manual_chunks"

    # ---------------- Gate A: Manual RAG ----------------
    manual_chunk_size: int = 300
    manual_chunk_overlap: int = 50
    manual_top_k: int = 3
    manual_similarity_threshold: float = 0.45  # 미만이면 키워드 재구성 후 1회 재검색 (문서 6-3, 6-4 [1단계])

    # ---------------- Gate B/C 가중치 (트랙 A: 우선순위 기반 수동 배정, 문서 6-5) ----------------
    track: str = "A"  # "A" (라벨 0~20건, 고정 가중치) / "B" (라벨 50건+, 로지스틱 회귀 가중치)

    weight_a: Dict[str, float] = Field(default_factory=lambda: {
        "reasoning_grounding": 0.35,
        "step_grounding_ratio": 0.30,
        "action_reason_consistency": 0.20,
        "utterance_signal": 0.15,
    })
    # 트랙 B 전환 시 과거 라벨로 로지스틱 회귀를 새로 학습해 이 값을 덮어쓸 것
    # (문서 6-5 더미 시뮬레이션 예시 값)
    weight_b: Dict[str, float] = Field(default_factory=lambda: {
        "reasoning_grounding": 0.311,
        "action_reason_consistency": 0.255,
        "step_grounding_ratio": 0.248,
        "utterance_signal": 0.186,
    })

    # ---------------- Threshold (문서 6-6) ----------------
    # 트랙 A: 휴리스틱 (accept/hold/reject ≈ 20% / 60% / 20% 가 되도록 넓게 hold 설정)
    t_high_a: float = 0.70
    t_low_a: float = 0.40
    # 트랙 B: 더미 시뮬레이션(precision>=0.97) 예시 값 — 실제 전환 시 PR-curve로 재산출
    t_high_b: float = 0.725
    t_low_b: float = 0.391

    # ---------------- 근거성 발화 마커 (utterance_signal 규칙 기반 부분, 문서 3-2-3) ----------------
    causal_markers: List[str] = Field(default_factory=lambda: ["때문에", "하면 ", "안 그러면", "그래야"])
    deviation_markers: List[str] = Field(default_factory=lambda: ["원래는", "보통은", "사실"])
    caution_markers: List[str] = Field(default_factory=lambda: ["꼭", "절대", "조심"])
    negation_markers: List[str] = Field(default_factory=lambda: ["하면 안", "안 돼", "안됩니다", "안 됩니다"])

    # ---------------- 기타 ----------------
    mock_mode: bool = False  # True: LLM/임베딩을 더미로 대체 (오프라인 동작 확인/테스트용)

    @field_validator("track")
    @classmethod
    def _validate_track(cls, v: str) -> str:
        if v not in ("A", "B"):
            raise ValueError(f"track은 'A' 또는 'B'여야 함 (받은 값: {v!r})")
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError(f"device는 'cuda' 또는 'cpu'여야 함 (받은 값: {v!r})")
        return v

    def weights(self) -> Dict[str, float]:
        return self.weight_b if self.track == "B" else self.weight_a

    def thresholds(self) -> tuple[float, float]:
        if self.track == "B":
            return self.t_high_b, self.t_low_b
        return self.t_high_a, self.t_low_a


CONFIG = Config()
