from dataclasses import dataclass, field
from typing import List, Dict, Any

from faster_whisper import WhisperModel


@dataclass
class STTSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0


@dataclass
class STTResult:
    segments: List[STTSegment] = field(default_factory=list)
    language: str = ""
    processing_time_s: float = 0.0


class FasterWhisperRunner:
    """faster-whisper 기반 STT 실행기. 조건 A/A'/B 공통 사용."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str,
        decoding_params: Dict[str, Any],
        chunk_offset: float = 0.0,
    ) -> STTResult:
        """
        단일 오디오(또는 청크) 전사.
        chunk_offset: 원본 오디오 내 이 청크의 시작 시간 (타임스탬프 재매핑용).
        """
        import time

        t0 = time.perf_counter()
        segments_gen, info = self._model.transcribe(audio_path, **decoding_params)
        segments = []
        for seg in segments_gen:
            segments.append(
                STTSegment(
                    start=seg.start + chunk_offset,
                    end=seg.end + chunk_offset,
                    text=seg.text,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    compression_ratio=seg.compression_ratio,
                )
            )
        elapsed = time.perf_counter() - t0
        return STTResult(
            segments=segments,
            language=info.language,
            processing_time_s=elapsed,
        )

    def transcribe_chunks(
        self,
        chunk_paths_with_offsets: List[tuple],
        decoding_params: Dict[str, Any],
    ) -> STTResult:
        """
        VAD로 추출된 청크 배치 전사 후 타임스탬프 재매핑.
        chunk_paths_with_offsets: [(audio_path, offset_s), ...]
        """
        import time

        all_segments = []
        language = ""
        t0 = time.perf_counter()
        for audio_path, offset in chunk_paths_with_offsets:
            result = self.transcribe(audio_path, decoding_params, chunk_offset=offset)
            all_segments.extend(result.segments)
            if not language:
                language = result.language
        elapsed = time.perf_counter() - t0

        return STTResult(
            segments=all_segments,
            language=language,
            processing_time_s=elapsed,
        )
