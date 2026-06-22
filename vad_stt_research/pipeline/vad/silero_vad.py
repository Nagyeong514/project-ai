from typing import List
import torch
import soundfile as sf

from .base import BaseVAD, SpeechSegment


class SileroVAD(BaseVAD):
    """
    Silero VAD 래퍼.
    threshold, min_speech_duration_ms, min_silence_duration_ms 는 민감도 분석 변수.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 400,
        merge_gap_ms: int = 200,
        max_chunk_s: float = 30.0,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.merge_gap_ms = merge_gap_ms
        self.max_chunk_s = max_chunk_s

        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )

    def detect(self, audio_path: str) -> List[SpeechSegment]:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_tensor = torch.from_numpy(audio)

        get_speech_timestamps = self._utils[0]
        raw = get_speech_timestamps(
            audio_tensor,
            self._model,
            sampling_rate=sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )
        segments = [SpeechSegment(start=s["start"], end=s["end"]) for s in raw]
        audio_duration = len(audio) / sr
        return self.postprocess(
            segments,
            audio_duration,
            speech_pad_ms=self.speech_pad_ms,
            merge_gap_ms=self.merge_gap_ms,
            max_chunk_s=self.max_chunk_s,
        )
