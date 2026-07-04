# -*- coding: utf-8 -*-
"""
문서 5단계(암묵지 JSON 생성) 스키마를 그대로 반영한 pydantic 모델.
검증 전용 단계이므로 여기서는 '읽기 + 유효성 확인'만 하고 내용을 새로 만들지 않는다 (문서 6-1, 6-2 원칙).
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    video_id: str
    clip_start: str
    clip_end: str
    transcript_ref: Optional[str] = None


class Metadata(BaseModel):
    scenario_id: str
    equipment: str
    task: str
    keywords: List[str] = Field(default_factory=list)
    scenario_title: Optional[str] = None
    source: SourceInfo


class DiagnosticStep(BaseModel):
    order: int
    action: str
    evidence: Literal["utterance", "action_only"]
    source_utterance: Optional[str] = None
    timestamp: Optional[str] = None


class Knowledge(BaseModel):
    situation: str
    situation_source: List[str] = Field(default_factory=list)
    tacit_insight: str
    reasoning: str
    reasoning_source: List[str] = Field(default_factory=list)
    reasoning_origin: Literal["utterance", "model_inferred"]
    diagnostic_steps: List[DiagnosticStep] = Field(default_factory=list)


class TacitKnowledgeCandidate(BaseModel):
    id: str
    schema_version: str
    metadata: Metadata
    knowledge: Knowledge


class TranscriptSegment(BaseModel):
    """
    transcript_ref 로 참조되는 파일의 세그먼트 형식.
    예: [{"timestamp": "00:04:30", "end": "00:04:33", "text": "일단 케이블부터 다 다시 꽂아보고"}, ...]
    end 은 optional (없으면 timestamp 기준 +5초 창으로 취급).
    """
    timestamp: str
    text: str
    end: Optional[str] = None
