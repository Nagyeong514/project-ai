from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class SpeechSegment:
    start: float   # seconds
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


class BaseVAD(ABC):
    """VAD 엔진 공통 인터페이스."""

    @abstractmethod
    def detect(self, audio_path: str) -> List[SpeechSegment]:
        """발화 구간 리스트 반환."""
        ...

    def apply_padding(
        self,
        segments: List[SpeechSegment],
        pad_ms: int,
        audio_duration: float,
    ) -> List[SpeechSegment]:
        pad_s = pad_ms / 1000.0
        padded = []
        for seg in segments:
            start = max(0.0, seg.start - pad_s)
            end = min(audio_duration, seg.end + pad_s)
            padded.append(SpeechSegment(start=start, end=end))
        return padded

    def merge_gaps(
        self,
        segments: List[SpeechSegment],
        merge_gap_ms: int,
    ) -> List[SpeechSegment]:
        if not segments:
            return segments
        gap_s = merge_gap_ms / 1000.0
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.start - merged[-1].end <= gap_s:
                merged[-1] = SpeechSegment(start=merged[-1].start, end=seg.end)
            else:
                merged.append(seg)
        return merged

    def postprocess(
        self,
        segments: List[SpeechSegment],
        audio_duration: float,
        speech_pad_ms: int = 400,
        merge_gap_ms: int = 200,
        max_chunk_s: float = 30.0,
    ) -> List[SpeechSegment]:
        segments = self.apply_padding(segments, speech_pad_ms, audio_duration)
        segments = self.merge_gaps(segments, merge_gap_ms)
        segments = self._split_long_segments(segments, max_chunk_s)
        return segments

    def _split_long_segments(
        self,
        segments: List[SpeechSegment],
        max_chunk_s: float,
    ) -> List[SpeechSegment]:
        result = []
        for seg in segments:
            start = seg.start
            while seg.end - start > max_chunk_s:
                result.append(SpeechSegment(start=start, end=start + max_chunk_s))
                start += max_chunk_s
            result.append(SpeechSegment(start=start, end=seg.end))
        return result
