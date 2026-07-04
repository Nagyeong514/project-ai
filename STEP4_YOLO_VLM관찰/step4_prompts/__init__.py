"""VLM 관찰 프롬프트(스펙 5.3). LLM 융합 프롬프트는 STEP5의 step5_prompts/에 있다."""

from .vlm_observation import (
    VLM_OBSERVATION_SYSTEM_PROMPT,
    build_observation_messages,
    build_video_observation_messages,
)

__all__ = [
    "VLM_OBSERVATION_SYSTEM_PROMPT",
    "build_observation_messages",
    "build_video_observation_messages",
]
