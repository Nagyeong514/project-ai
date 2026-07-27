"""faster-whisper 기반 STT 실행기 — large-v3 / large-v3-turbo / medium-ko 공용."""
import time
from typing import Any, Dict, List, Optional

from faster_whisper import WhisperModel

from pipeline.stt.base import BaseSTT, STTResult, STTSegment


class FasterWhisperRunner(BaseSTT):
    """
    model_id 에 HuggingFace 모델명 또는 faster-whisper 내장 모델명을 전달.
    예: "large-v3", "large-v3-turbo", "seastar105/whisper-medium-ko-zeroth"
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "ko",
        decoding_params: Optional[Dict[str, Any]] = None,
    ):
        self._model_id = model_id
        self._language = language
        self._decoding_params: Dict[str, Any] = decoding_params or {}
        self._model = WhisperModel(model_id, device=device, compute_type=compute_type)

    @property
    def name(self) -> str:
        return self._model_id

    def transcribe(self, audio_path: str) -> STTResult:
        params = {
            "language": self._language,
            **self._decoding_params,
        }

        t0 = time.perf_counter()
        segments_gen, info = self._model.transcribe(audio_path, **params)
        segments: List[STTSegment] = []
        for seg in segments_gen:
            segments.append(STTSegment(start=seg.start, end=seg.end, text=seg.text))
        elapsed = time.perf_counter() - t0

        return STTResult(
            segments=segments,
            language=info.language,
            processing_time_s=elapsed,
            audio_duration_s=info.duration,
        )
