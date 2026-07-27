"""
WebRTC VAD 엔진 (GMM 기반 경량 VAD).
Phase 1의 BaseVAD를 상속해 padding/merge/split 후처리를 Silero와 동일하게 공유한다
(엔진 검출 차이만 분리해 비교하기 위함).

webrtcvad는 8/16/32/48kHz, 16-bit mono PCM, 10/20/30ms 프레임만 지원.
연구 데이터는 16kHz mono 16-bit이므로 그대로 사용.
"""
from typing import List

import webrtcvad
import soundfile as sf

from pipeline.vad.base import BaseVAD, SpeechSegment


class WebRTCVAD(BaseVAD):
    def __init__(
        self,
        aggressiveness: int = 2,        # 0(관대)~3(공격적)
        frame_ms: int = 30,             # 10/20/30 만 허용
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 400,
        merge_gap_ms: int = 200,
        max_chunk_s: float = 30.0,
    ):
        self._vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.merge_gap_ms = merge_gap_ms
        self.max_chunk_s = max_chunk_s

    def detect(self, audio_path: str) -> List[SpeechSegment]:
        audio, sr = sf.read(audio_path, dtype="int16")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr not in (8000, 16000, 32000, 48000):
            raise ValueError(f"webrtcvad 미지원 샘플레이트: {sr}")

        frame_len = int(sr * self.frame_ms / 1000)
        n_frames = len(audio) // frame_len
        flags = [
            self._vad.is_speech(audio[i * frame_len:(i + 1) * frame_len].tobytes(), sr)
            for i in range(n_frames)
        ]

        segments = self._group_frames(flags, frame_len / sr)
        audio_duration = len(audio) / sr
        return self.postprocess(
            segments,
            audio_duration,
            speech_pad_ms=self.speech_pad_ms,
            merge_gap_ms=self.merge_gap_ms,
            max_chunk_s=self.max_chunk_s,
        )

    def _group_frames(self, flags: List[bool], frame_s: float) -> List[SpeechSegment]:
        """프레임 발화 플래그를 구간으로 묶음. 짧은 무음은 잇고, 짧은 발화는 버림."""
        min_speech_frames = (self.min_speech_duration_ms / 1000) / frame_s
        min_silence_frames = (self.min_silence_duration_ms / 1000) / frame_s

        segments = []
        in_speech = False
        start = 0
        silence_run = 0
        for i, is_speech in enumerate(flags):
            if is_speech:
                if not in_speech:
                    in_speech, start = True, i
                silence_run = 0
            elif in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    end = i - silence_run + 1
                    if (end - start) >= min_speech_frames:
                        segments.append(SpeechSegment(start * frame_s, end * frame_s))
                    in_speech = False
        if in_speech:
            end = len(flags)
            if (end - start) >= min_speech_frames:
                segments.append(SpeechSegment(start * frame_s, end * frame_s))
        return segments
