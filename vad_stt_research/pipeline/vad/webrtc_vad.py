from typing import List

import numpy as np
import soundfile as sf
import webrtcvad

from .base import BaseVAD, SpeechSegment

_FRAME_MS = 30   # webrtcvad 지원 프레임: 10 | 20 | 30


class WebRTCVAD(BaseVAD):
    """
    WebRTC VAD 래퍼 — 경량/고속 비교군 (레거시 특성 참고용).
    aggressiveness: 0(관대) ~ 3(공격적)
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        speech_pad_ms: int = 400,
        merge_gap_ms: int = 200,
        max_chunk_s: float = 30.0,
    ):
        self._vad = webrtcvad.Vad(aggressiveness)
        self.speech_pad_ms = speech_pad_ms
        self.merge_gap_ms = merge_gap_ms
        self.max_chunk_s = max_chunk_s

    def detect(self, audio_path: str) -> List[SpeechSegment]:
        audio, sr = sf.read(audio_path, dtype="int16")
        if sr not in (8000, 16000, 32000, 48000):
            raise ValueError(f"WebRTC VAD는 8/16/32/48 kHz만 지원. 현재: {sr}")
        if audio.ndim > 1:
            audio = audio[:, 0]

        frame_len = int(sr * _FRAME_MS / 1000)
        frames = [
            audio[i : i + frame_len]
            for i in range(0, len(audio) - frame_len + 1, frame_len)
        ]

        speech_flags = []
        for frame in frames:
            raw = frame.tobytes()
            try:
                is_speech = self._vad.is_speech(raw, sr)
            except Exception:
                is_speech = False
            speech_flags.append(is_speech)

        segments = self._flags_to_segments(speech_flags, frame_ms=_FRAME_MS)
        audio_duration = len(audio) / sr
        return self.postprocess(
            segments,
            audio_duration,
            speech_pad_ms=self.speech_pad_ms,
            merge_gap_ms=self.merge_gap_ms,
            max_chunk_s=self.max_chunk_s,
        )

    @staticmethod
    def _flags_to_segments(flags: list, frame_ms: int) -> List[SpeechSegment]:
        segments = []
        in_speech = False
        start = 0.0
        for i, flag in enumerate(flags):
            t = i * frame_ms / 1000.0
            if flag and not in_speech:
                start = t
                in_speech = True
            elif not flag and in_speech:
                segments.append(SpeechSegment(start=start, end=t))
                in_speech = False
        if in_speech:
            segments.append(SpeechSegment(start=start, end=len(flags) * frame_ms / 1000.0))
        return segments
