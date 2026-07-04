"""Meta Seamless-M4T-v2 기반 STT 실행기."""
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

import soundfile as sf

from pipeline.stt.base import BaseSTT, STTResult, STTSegment
from pipeline.stt.chunking import chunk_audio


class SeamlessM4Tv2Runner(BaseSTT):
    """
    facebook/seamless-m4t-v2-large 기반 ASR(S2TT, 입력=출력 언어) 실행기.
    UnitY2 음성 인코더는 짧은 발화 단위로 학습돼 장문 처리 보장이 명시돼 있지 않아,
    안전 마진을 둔 고정 창(기본 25초)으로 청킹해 처리한다.
    """

    def __init__(
        self,
        model_id: str,
        tgt_lang: str = "kor",
        device: str = "cuda",
        window_s: float = 25.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self._model_id = model_id
        self._tgt_lang = tgt_lang
        self._window_s = window_s
        # 기본 generate()는 num_beams=1(그리디)·repetition_penalty=1.0(억제 없음)이라
        # jump-cut 스플라이싱 오디오에서 조기 EOS·반복 루프에 취약함(실측 확인,
        # seamless_누락진단/SEAMLESS_누락원인.md). 여기서 오버라이드 가능하게 함.
        self._generation_kwargs = generation_kwargs or {}
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id).to(device)

    @property
    def name(self) -> str:
        return self._model_id

    def transcribe(self, audio_path: str) -> STTResult:
        audio_duration_s = sf.info(audio_path).duration
        tmp_dir = tempfile.mkdtemp(prefix="seamless_chunks_")

        t0 = time.perf_counter()
        try:
            chunks = chunk_audio(audio_path, window_s=self._window_s, tmp_dir=tmp_dir)
            segments: List[STTSegment] = []
            for start_s, end_s, chunk_path in chunks:
                chunk_array, sr = sf.read(chunk_path, dtype="float32")
                inputs = self._processor(
                    audio=chunk_array, sampling_rate=sr, return_tensors="pt"
                ).to(self._model.device)
                output_tokens = self._model.generate(
                    **inputs, tgt_lang=self._tgt_lang, **self._generation_kwargs
                )
                text = self._processor.decode(output_tokens[0], skip_special_tokens=True)
                segments.append(STTSegment(start=start_s, end=end_s, text=text))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elapsed = time.perf_counter() - t0

        return STTResult(
            segments=segments,
            language=self._tgt_lang,
            processing_time_s=elapsed,
            audio_duration_s=audio_duration_s,
        )
