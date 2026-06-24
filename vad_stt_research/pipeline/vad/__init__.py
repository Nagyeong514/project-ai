from __future__ import annotations

from .base import BaseVAD, SpeechSegment
from .silero_vad import SileroVAD

__all__ = ["get_vad", "BaseVAD", "SpeechSegment", "SileroVAD"]


def get_vad(engine: str = "silero", **kwargs) -> BaseVAD:
    if engine != "silero":
        raise ValueError(f"지원 엔진: silero (요청: '{engine}')")
    return SileroVAD(**kwargs)
