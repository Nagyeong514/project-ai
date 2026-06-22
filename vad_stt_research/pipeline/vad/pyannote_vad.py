from typing import List

from .base import BaseVAD, SpeechSegment


class PyannoteVAD(BaseVAD):
    """
    pyannote.audio VAD 래퍼 (WhisperX baseline과 동일 계열).
    HuggingFace 토큰 필요: hf_token 파라미터 또는 HF_TOKEN 환경 변수.
    """

    def __init__(
        self,
        hf_token: str | None = None,
        speech_pad_ms: int = 400,
        merge_gap_ms: int = 200,
        max_chunk_s: float = 30.0,
    ):
        import os
        from pyannote.audio import Pipeline

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN 환경 변수 또는 hf_token 파라미터 필요")

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=token,
        )
        self.speech_pad_ms = speech_pad_ms
        self.merge_gap_ms = merge_gap_ms
        self.max_chunk_s = max_chunk_s

    def detect(self, audio_path: str) -> List[SpeechSegment]:
        import soundfile as sf

        output = self._pipeline(audio_path)
        audio, sr = sf.read(audio_path)
        audio_duration = len(audio) / sr

        segments = [
            SpeechSegment(start=turn.start, end=turn.end)
            for turn, _, _ in output.itertracks(yield_label=True)
        ]
        return self.postprocess(
            segments,
            audio_duration,
            speech_pad_ms=self.speech_pad_ms,
            merge_gap_ms=self.merge_gap_ms,
            max_chunk_s=self.max_chunk_s,
        )
