"""LLM 융합 인터페이스."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from ..schema.intermediate import AlignedWindow, FrameMeta
from ..schema.tacit_schema import TacitKnowledgeDocument


@runtime_checkable
class LLMBackend(Protocol):
    """정렬된 (행동+발화) 구간 → 암묵지 후보 JSON. (예: Qwen2.5-14B)

    스펙 5.7: 3케이스(fusion/action_only/utterance_only) 구분 처리.
    할루시네이션 규율: 입력에 없는 사실 생성 금지. 묶기·분류·라벨링은 허용.
    추론은 reasoning_origin=model_inferred 로 정직하게 태깅(발화 근거인 척 금지).
    출력은 tacit_schema 로 검증.
    """

    def fuse(
        self,
        windows: List[AlignedWindow],
        meta: FrameMeta,
    ) -> TacitKnowledgeDocument:
        """후보 문서 반환(스키마 검증 통과본)."""
        ...
