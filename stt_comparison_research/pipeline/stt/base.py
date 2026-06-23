"""STT 결과 데이터클래스 및 추상 기반 클래스."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class STTSegment:
    start: float
    end: float
    text: str


@dataclass
class STTResult:
    segments: List[STTSegment] = field(default_factory=list)
    language: str = "ko"
    processing_time_s: float = 0.0
    audio_duration_s: float = 0.0

    @property
    def rtf(self) -> float:
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.processing_time_s / self.audio_duration_s

    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)


class BaseSTT(ABC):
    """모든 STT 엔진이 구현해야 하는 인터페이스."""

    @property
    @abstractmethod
    def name(self) -> str:
        """모델 식별자 (results.csv의 model 컬럼값)."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> STTResult:
        """
        16kHz mono WAV를 받아 STTResult 반환.
        audio_duration_s 는 호출자가 채워줘도 되고 구현체가 채워도 됨.
        """
