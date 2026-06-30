"""
최종 출력 JSON 스키마 — **단일 진실 공급원(single source of truth)**.

파이프라인 어디서도 JSON 키 문자열을 하드코딩하지 마라. 키가 바뀌면 이 파일 한 곳만 고친다.
스키마는 아직 '최종 아님'(`SCHEMA_VERSION`)이고 키 이름이 바뀔 수 있다.

작업 지시서 4번의 목표 JSON을 그대로 Pydantic으로 옮긴 것이다.
주석의 [LLM]/[VLM]/[STT]/[자동] 은 각 필드를 누가 채우는지(출처)를 뜻한다.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# 스키마 버전. 키/구조가 바뀌면 여기서 올린다. 최종 출력 JSON에도 박혀 나간다.
SCHEMA_VERSION = "1.3"


# ──────────────────────────────────────────────────────────────────────────
# Enum — 자유 문자열 대신 닫힌 집합으로 둬서 오타/혼선을 막는다.
# ──────────────────────────────────────────────────────────────────────────
class EvidenceType(str, Enum):
    """한 항목의 근거가 무엇이냐.

    - UTTERANCE: 발화(말)로 뒷받침됨.
    - ACTION_ONLY: 말은 안 했지만 영상 행동으로 관측됨(말 안 한 중요한 행동).
    """

    UTTERANCE = "utterance"
    ACTION_ONLY = "action_only"


class ReasoningOrigin(str, Enum):
    """reasoning(원인 설명)이 어디서 왔냐 — 할루시네이션 규율의 핵심.

    - UTTERANCE: 숙련자 발화에서 직접 도출.
    - MODEL_INFERRED: LLM 일반지식으로 채움. **발화 근거인 척 위장 금지.**
    """

    UTTERANCE = "utterance"
    MODEL_INFERRED = "model_inferred"


# ──────────────────────────────────────────────────────────────────────────
# 하위 모델
# ──────────────────────────────────────────────────────────────────────────
class Source(BaseModel):
    """원본 추적 정보. clip_start/clip_end 는 'HH:MM:SS' 문자열(스펙 그대로)."""

    video_id: str
    clip_start: Optional[str] = None  # "00:04:12"
    clip_end: Optional[str] = None  # "00:06:48"
    transcript_ref: Optional[str] = None  # "transcripts/<video_id>.json"


class Metadata(BaseModel):
    scenario_id: str  # 원본 파일명 기반
    equipment: Optional[str] = None  # [LLM] 영상/메타에서 식별
    task: Optional[str] = None  # [LLM] 작업유형 자동분류
    keywords: List[str] = Field(default_factory=list)  # [LLM]
    scenario_title: Optional[str] = None  # [LLM] 자동 생성
    source: Source


class DiagnosticStep(BaseModel):
    """진단 절차 한 스텝. evidence 로 발화근거/행동단독을 구분한다."""

    order: int
    action: str  # [VLM 행동] 무엇을 했는가
    evidence: EvidenceType
    # 발화 원문 그대로 보존(정제 단계에서 버리지 않는다). action_only면 None.
    source_utterance: Optional[str] = None
    timestamp: Optional[str] = None  # "HH:MM:SS" — action_only면 None일 수 있음

    # NOTE: source_utterance/ timestamp 가 None인데 evidence=UTTERANCE면 모순.
    # 교차검증은 validators에서(아래) 수행.


class Knowledge(BaseModel):
    situation: str
    situation_source: List[str] = Field(default_factory=list)  # 근거 발화 timestamp들
    tacit_insight: str
    reasoning: Optional[str] = None
    reasoning_source: List[str] = Field(default_factory=list)
    reasoning_origin: ReasoningOrigin = ReasoningOrigin.MODEL_INFERRED
    diagnostic_steps: List[DiagnosticStep] = Field(default_factory=list)
    # TODO(decision): normal_procedure(매뉴얼 정상절차, 델타 판정 기준선) — 후속 단계용. 지금은 미포함.
    # normal_procedure: Optional[str] = None


class TacitKnowledgeCandidate(BaseModel):
    """LLM 융합 단계가 최종적으로 채우는 '암묵지 후보' 1건.

    이 파이프라인은 '후보 생성'까지만 책임진다. 진짜 암묵지인지 판별/검증은 다음 단계.
    """

    id: str  # [자동] video_id + task + 일련번호  e.g. tk_dell7920_mem_boot_001
    schema_version: str = SCHEMA_VERSION
    metadata: Metadata
    knowledge: Knowledge

    # ── 교차 검증 ──────────────────────────────────────────────────────
    def cross_check(self) -> List[str]:
        """할루시네이션 규율 위반 의심 지점을 문자열 리스트로 돌려준다(에러는 아님).

        오케스트레이터가 로그/재시도 판단에 쓴다. 강제 실패가 아니라 '경고'로 둔 이유:
        후보는 빠짐없이 살리는 게 우선이고, 판별은 다음 팀 몫이라서.
        """
        warnings: List[str] = []
        for step in self.knowledge.diagnostic_steps:
            if step.evidence == EvidenceType.UTTERANCE and not step.source_utterance:
                warnings.append(
                    f"step {step.order}: evidence=utterance인데 source_utterance가 비었음(위장 의심)"
                )
            if step.evidence == EvidenceType.ACTION_ONLY and step.source_utterance:
                warnings.append(
                    f"step {step.order}: evidence=action_only인데 source_utterance가 있음(모순)"
                )
        if (
            self.knowledge.reasoning_origin == ReasoningOrigin.UTTERANCE
            and not self.knowledge.reasoning_source
        ):
            warnings.append("reasoning_origin=utterance인데 reasoning_source 근거 timestamp가 없음")
        return warnings


class TacitKnowledgeDocument(BaseModel):
    """한 영상에서 나온 후보들의 묶음(파일 단위 출력).

    한 영상에서 여러 암묵지 후보가 나올 수 있으므로 리스트로 감싼다.
    """

    schema_version: str = SCHEMA_VERSION
    video_id: str
    candidates: List[TacitKnowledgeCandidate] = Field(default_factory=list)
