"""컴포넌트 추상 인터페이스(Protocol). 구체 구현은 각 STEP의 step{N}_components/ 에 둔다.

LLMBackend는 여기 없다 — tacit_schema.TacitKnowledgeDocument(STEP5 소유)에 의존해서
STEP5_LLM으로_암묵지_후보생성/step5_llm_interface.py 로 옮김(공용 패키지가 특정 STEP
전용 타입에 의존하면 안 됨)."""

from .detector import DetectorBackend
from .sampler import FrameSampler
from .stt import STTBackend, TranscriptRefiner
from .vlm import VLMBackend

__all__ = [
    "DetectorBackend",
    "FrameSampler",
    "STTBackend",
    "TranscriptRefiner",
    "VLMBackend",
]
