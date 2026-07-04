"""최종 출력 스키마(암묵지 후보 JSON). LLM 융합 결과물이라 STEP5 소유(스펙 2: 단일 진실 공급원)."""

from .tacit_schema import (
    SCHEMA_VERSION,
    DiagnosticStep,
    EvidenceType,
    Knowledge,
    Metadata,
    ReasoningOrigin,
    Source,
    TacitKnowledgeCandidate,
    TacitKnowledgeDocument,
)

__all__ = [
    "SCHEMA_VERSION",
    "TacitKnowledgeCandidate",
    "TacitKnowledgeDocument",
    "Knowledge",
    "Metadata",
    "Source",
    "DiagnosticStep",
    "EvidenceType",
    "ReasoningOrigin",
]
