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
    # 2026-07-12: Qwen2.5-14B-Instruct → Qwen3-14B 교체(지시 — STEP5 융합·STEP8 답변과
    # 모델 계열 통일). thinking은 llm_utils.QwenLLM이 자동으로 끈다(qwen3 계열 감지).
    # ⚠️ Judge 교체는 판정 기준 자체의 변화 — 기존 판정(motion_obs_run6 등)은 Qwen2.5
    # 기준 산출물이므로 새 실행과 직접 비교하지 말 것. 롤백값: "Qwen/Qwen2.5-14B-Instruct".
    llm_model_name: str = "Qwen/Qwen3-14B"
    embedding_model_name: str = "BAAI/bge-m3"
    device: str = "cuda"  # GPU 없으면 "cpu" 로 변경
    llm_max_new_tokens: int = 800
    llm_temperature: float = 0.0  # judge 용이므로 결정론적으로

    # ---------------- Vector DB ----------------
    # 2026-07-14 전환(지시): Qdrant → ChromaDB. STEP7/STEP8이 이미 Chroma로 전환돼 있어
    # Gate A(매뉴얼 RAG)도 팀 표준에 맞춤. 매뉴얼은 소규모+매 실행 재색인이라 in-memory
    # (EphemeralClient)로 둔다 — 편집한 매뉴얼이 항상 반영되고 stale 색인 위험 없음.
    # ⚠️ manual_similarity_threshold(0.45)는 cosine 유사도 기준. Chroma는 distance=1-cos_sim
    #    을 반환하므로 어댑터가 (1 - distance)로 유사도 역변환한다(STEP7과 동일, 실측 대조됨).
    vector_backend: str = "chroma"  # "chroma" | "qdrant" (롤백용 코드 보존)
    chroma_collection: str = "manual_chunks"
    # 롤백용(qdrant) — vector_backend="qdrant" 일 때만 사용
    qdrant_location: str = ":memory:"
    qdrant_collection: str = "manual_chunks"

    # ---------------- Gate A: Manual RAG ----------------
    manual_chunk_size: int = 300
    manual_chunk_overlap: int = 50
    manual_top_k: int = 3
    manual_similarity_threshold: float = 0.45  # 미만이면 키워드 재구성 후 1회 재검색 (문서 6-3, 6-4 [1단계])

    # ---------------- Gate B/C 가중치 (트랙 A: 우선순위 기반 수동 배정, 문서 6-5) ----------------
    track: str = "A"  # "A" (라벨 0~20건, 고정 가중치) / "B" (라벨 50건+, 로지스틱 회귀 가중치)

    # 2026-07-10 개정: utterance_signal 0.15→0.30 (마커 개정과 한 세트 — 아래 마커 주석 참고).
    # us를 올리는 대신 sg 0.30→0.20, arc 0.20→0.15로 재배분(rg 0.35 유지).
    weight_a: Dict[str, float] = Field(default_factory=lambda: {
        "reasoning_grounding": 0.35,
        "utterance_signal": 0.30,
        "step_grounding_ratio": 0.20,
        "action_reason_consistency": 0.15,
    })
    # 침묵 트랙 (2026-07-09): 발화 참조가 전혀 없는 후보(utterance step 0 AND
    # reasoning_source 빈 배열)는 rg/us가 데이터 부재로 0에 깔려 이론 상한이
    # 0.50 < T_high 0.70 — 구조적으로 accept 불가였다(run3 실측: 침묵 13건 전원
    # rg=us=0, 최대 conf 0.420). 발화 유무로 가중치를 분기해 침묵 후보도 만점 시
    # conf 1.0이 되게 한다. 침묵 암묵지("본인도 말로 못 푸는 지식")가 이 프로젝트의
    # 핵심 타깃인데 발화 전제 항목 때문에 감점되는 모순 해소.
    # weight_b_silent 는 지금 만들지 않는다 — 트랙 B(라벨 50건+) 전환 시점에
    # 로지스틱 회귀로 재산정할 것.
    weight_a_silent: Dict[str, float] = Field(default_factory=lambda: {
        "step_grounding_ratio": 0.60,
        "action_reason_consistency": 0.40,
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
    # 2026-07-10 개정: motion_obs_run1 24건 발화 전수 대조 결과 반영.
    # 기존 마커는 문어체 기준이라 이 정비공의 구어 패턴(의무형 어미·조건 판독·순서 표현)을
    # 거의 못 잡았음 — ACCEPT 목표 후보 히트율 0.0에서 0.2. 아래는 실발화에서 역추출한 마커.
    # "하면 "은 삭제 — rules.py가 공백 제거 후 부분일치라 "세팅하면 됩니다" 같은 평범 발화에도
    # 걸려, 접지 위장 후보(C1_001~003)의 conf를 밀어 올리는 오탐원이었음. ACCEPT 목표 발화 중
    # "하면" 의존은 없음(딸깍은 "나면", 커넥터는 "먼저"로 잡힘).
    causal_markers: List[str] = Field(default_factory=lambda: [
        "때문에", "안 그러면", "그래야",
        "이면",      # 조건 판독: "황색4 백색5이면은", "오인식이면"
        "나면",      # "딸깍 소리가 나면"
        "거든",      # 설명 종결: "날아가거든요"
    ])
    deviation_markers: List[str] = Field(default_factory=lambda: [
        "원래는", "보통은", "사실",
        "여도", "어도",   # 양보 후 반전: "깨끗해 보여도", "켜졌어도"
        "다가 아니",      # "켜졌다고 다가 아니에요"
    ])
    caution_markers: List[str] = Field(default_factory=lambda: [
        "꼭", "절대", "조심",
        "야 합니다", "야 해",   # 의무형: "줘야 합니다", "봐야 해요", "끼워야 합니다"
        "먼저",                 # 순서 지식: "교체보다 먼저", "24핀 메인 먼저"
    ])
    negation_markers: List[str] = Field(default_factory=lambda: [
        "하면 안", "안 돼", "안됩니다", "안 됩니다",
        "지 않",   # "통전이 되지 않습니다"
    ])

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

    def weights_silent(self) -> Dict[str, float]:
        # TODO: track B 전환 시 weight_b_silent 를 라벨 데이터로 재산정해 분기할 것.
        #       그 전까지는 track 과 무관하게 weight_a_silent 를 쓴다.
        return self.weight_a_silent

    def thresholds(self) -> tuple[float, float]:
        if self.track == "B":
            return self.t_high_b, self.t_low_b
        return self.t_high_a, self.t_low_a


CONFIG = Config()

# 2026-07-14: 판정자(judge) 모델을 실행별로 오버라이드(격리 실험용). 미설정 시 위 기본값 유지.
# 예: STEP6_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct" python main.py  (run6 판정자로 매뉴얼 효과만 격리)
_llm_override = os.environ.get("STEP6_LLM_MODEL")
if _llm_override:
    CONFIG.llm_model_name = _llm_override
