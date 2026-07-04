"""프롬프트 모음. 반복 수정 용이하게 코드와 분리(스펙 5.3 / 5.7)."""

from .llm_fusion_prompt import build_fusion_messages, FUSION_SYSTEM_PROMPT, LED_DIAGNOSTIC_TABLE
from .vlm_observation import build_observation_messages, VLM_OBSERVATION_SYSTEM_PROMPT

__all__ = [
    "VLM_OBSERVATION_SYSTEM_PROMPT",
    "build_observation_messages",
    "FUSION_SYSTEM_PROMPT",
    "LED_DIAGNOSTIC_TABLE",
    "build_fusion_messages",
]
