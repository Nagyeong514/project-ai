"""Kospeech 기반 STT 실행기 — 하한선 baseline."""
import time
from pathlib import Path

from pipeline.stt.base import BaseSTT, STTResult, STTSegment


class KospeechRunner(BaseSTT):
    """
    사전 준비:
      pip install kospeech
      가중치 다운로드: https://github.com/sooftware/kospeech (KsponSpeech LAS checkpoint)
      checkpoint_path: 다운로드한 .pt 파일 경로

    지원 아키텍처: Listen-Attend-Spell (LAS) — KsponSpeech 학습 버전
    입력: 16kHz mono WAV
    """

    def __init__(self, checkpoint_path: str):
        self._checkpoint_path = checkpoint_path
        self._model = self._load_model(checkpoint_path)

    @property
    def name(self) -> str:
        return "kospeech"

    def _load_model(self, checkpoint_path: str):
        # kospeech 패키지 임포트는 무거우므로 실제 사용 시점에만 로드
        try:
            import torch
            from kospeech.models import ListenAttendSpell

            # KsponSpeech LAS 기본 설정으로 모델 초기화 후 checkpoint 로드
            ckpt = torch.load(checkpoint_path, map_location="cuda")
            model = ckpt.get("model", ckpt)
            model.eval()
            return model
        except ImportError as e:
            raise RuntimeError(
                "kospeech 패키지 미설치. `pip install kospeech` 실행 후 재시도."
            ) from e

    def transcribe(self, audio_path: str) -> STTResult:
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        audio_duration_s = waveform.shape[-1] / 16000

        # 모노 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = waveform.cuda()

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.recognize(inputs, input_lengths=torch.tensor([inputs.shape[-1]]))
        elapsed = time.perf_counter() - t0

        # outputs: list of token index tensors or decoded strings depending on version
        text = self._decode_output(outputs)

        return STTResult(
            segments=[STTSegment(start=0.0, end=audio_duration_s, text=text)],
            language="ko",
            processing_time_s=elapsed,
            audio_duration_s=audio_duration_s,
        )

    def _decode_output(self, outputs) -> str:
        """kospeech 버전에 따라 출력 형식이 다를 수 있음 — 필요 시 수정."""
        if isinstance(outputs, list) and len(outputs) > 0:
            out = outputs[0]
            if isinstance(out, str):
                return out
            # token index → 문자열 (vocab 필요)
            return str(out)
        return str(outputs)
