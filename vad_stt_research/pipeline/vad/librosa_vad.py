from typing import List

import librosa
import numpy as np

from .base import BaseVAD, SpeechSegment


class LibrosaVAD(BaseVAD):
    """
    Librosa dB 임계값 기반 규칙적 VAD — 하한 기준점 (규칙 기반).
    """

    def __init__(
        self,
        top_db: float = 40.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        speech_pad_ms: int = 400,
        merge_gap_ms: int = 200,
        max_chunk_s: float = 30.0,
    ):
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.speech_pad_ms = speech_pad_ms
        self.merge_gap_ms = merge_gap_ms
        self.max_chunk_s = max_chunk_s

    def detect(self, audio_path: str) -> List[SpeechSegment]:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        intervals = librosa.effects.split(
            audio,
            top_db=self.top_db,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        segments = [
            SpeechSegment(start=start / sr, end=end / sr)
            for start, end in intervals
        ]
        audio_duration = len(audio) / sr
        return self.postprocess(
            segments,
            audio_duration,
            speech_pad_ms=self.speech_pad_ms,
            merge_gap_ms=self.merge_gap_ms,
            max_chunk_s=self.max_chunk_s,
        )
