"""컴포넌트 추상 인터페이스(Protocol). 구체 구현은 components/ 에 둔다."""

from .detector import DetectorBackend
from .llm import LLMBackend
from .sampler import FrameSampler
from .stt import STTBackend, TranscriptRefiner
from .vlm import VLMBackend

__all__ = [
    "DetectorBackend",
    "LLMBackend",
    "FrameSampler",
    "STTBackend",
    "TranscriptRefiner",
    "VLMBackend",
]
