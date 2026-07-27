"""kresnik/wav2vec2-large-xlsr-korean 기반 STT 실행기 (CTC greedy decode)."""
import shutil
import tempfile
import time
from typing import List

import soundfile as sf
import torch

from pipeline.stt.base import BaseSTT, STTResult, STTSegment
from pipeline.stt.chunking import chunk_audio


class Wav2Vec2XLSRRunner(BaseSTT):
    """
    한국어 파인튜닝 Wav2Vec2-XLSR 기반 CTC(비-자기회귀) 실행기.
    자기회귀 디코더가 없는 만큼 장문 오디오는 어텐션 메모리가 급증하므로,
    Seamless-M4T-v2 러너와 동일한 청킹 헬퍼(기본 25초 창)로 처리한다.
    """

    def __init__(self, model_id: str, device: str = "cuda", window_s: float = 25.0):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._model_id = model_id
        self._device = device
        self._window_s = window_s
        self._processor = Wav2Vec2Processor.from_pretrained(model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    @property
    def name(self) -> str:
        return self._model_id

    def transcribe(self, audio_path: str) -> STTResult:
        audio_duration_s = sf.info(audio_path).duration
        tmp_dir = tempfile.mkdtemp(prefix="wav2vec2_chunks_")

        t0 = time.perf_counter()
        try:
            chunks = chunk_audio(audio_path, window_s=self._window_s, tmp_dir=tmp_dir)
            segments: List[STTSegment] = []
            for start_s, end_s, chunk_path in chunks:
                chunk_array, sr = sf.read(chunk_path, dtype="float32")
                inputs = self._processor(
                    chunk_array, sampling_rate=sr, return_tensors="pt", padding="longest"
                )
                input_values = inputs.input_values.to(self._device)
                with torch.no_grad():
                    logits = self._model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                text = self._processor.batch_decode(predicted_ids)[0]
                segments.append(STTSegment(start=start_s, end=end_s, text=text))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elapsed = time.perf_counter() - t0

        return STTResult(
            segments=segments,
            language="ko",
            processing_time_s=elapsed,
            audio_duration_s=audio_duration_s,
        )
