"""스키마 v1.3 (계획서 §5 그대로) + 중간 계약 타입.

원칙: 중간 타입과 최종 스키마를 절대 섞지 않는다.
metadata 는 코드가 채운다 — LLM 은 knowledge 만 책임진다.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.3"


# ─────────────────────────── 중간 계약 ───────────────────────────

class Utterance(BaseModel):
    start: float
    end: float
    raw_text: str
    normalized_text: str = ""
    repeat_hallucination: bool = False


class Transcript(BaseModel):
    video_id: str
    language: str = "ko"
    model: str = ""
    utterances: List[Utterance] = Field(default_factory=list)


class Detection(BaseModel):
    timestamp: float
    frame_idx: int
    cls: str
    conf: float


class Observation(BaseModel):
    """VLM 관찰 1건. repeat_count>1 이면 dedup 으로 접힌 지속 동작."""
    timestamp: float
    end_timestamp: Optional[float] = None
    actor: str = ""
    action: str
    objects: List[str] = Field(default_factory=list)
    repeat_count: int = 1
    chunk: str = ""  # 어느 청크에서 나왔는지 (디버깅용, 예: "60.0-90.0")


class AlignedWindow(BaseModel):
    window_start: float
    window_end: float
    actions: List[Observation] = Field(default_factory=list)
    utterances: List[Utterance] = Field(default_factory=list)

    @property
    def case(self) -> str:
        if self.actions and self.utterances:
            return "fusion"
        if self.actions:
            return "action_only"
        if self.utterances:
            return "utterance_only"
        return "empty"


# ─────────────────────────── 최종 스키마 v1.3 ───────────────────────────

class EvidenceType(str, Enum):
    utterance = "utterance"
    action_only = "action_only"


class ReasoningOrigin(str, Enum):
    utterance = "utterance"
    model_inferred = "model_inferred"


class Source(BaseModel):
    video_id: str = ""
    clip_start: str = "00:00:00"
    clip_end: str = "00:00:00"
    transcript_ref: str = ""


class Metadata(BaseModel):
    scenario_id: str = ""
    equipment: Optional[str] = None
    task: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    scenario_title: Optional[str] = None
    source: Source = Field(default_factory=Source)


class DiagnosticStep(BaseModel):
    order: int
    action: str
    evidence: EvidenceType = EvidenceType.action_only
    source_utterance: Optional[str] = None
    timestamp: Optional[str] = None


class Knowledge(BaseModel):
    situation: str = ""
    situation_source: List[str] = Field(default_factory=list)
    tacit_insight: str = ""
    reasoning: str = ""
    reasoning_source: List[str] = Field(default_factory=list)
    reasoning_origin: ReasoningOrigin = ReasoningOrigin.model_inferred
    diagnostic_steps: List[DiagnosticStep] = Field(default_factory=list)
    conflict: bool = False
    conflict_detail: Optional[str] = None


class TacitKnowledgeCandidate(BaseModel):
    id: str = ""
    schema_version: str = SCHEMA_VERSION
    metadata: Metadata = Field(default_factory=Metadata)
    knowledge: Knowledge


class TacitKnowledgeDocument(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video_id: str
    candidates: List[TacitKnowledgeCandidate] = Field(default_factory=list)
