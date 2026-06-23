"""CLOVA Speech / Kakao Speech REST API 래퍼."""
import os
import time
from pathlib import Path

import requests

from pipeline.stt.base import BaseSTT, STTResult, STTSegment


class ClovaSTTRunner(BaseSTT):
    """
    CLOVA Speech Recognition (CSR) API.
    환경변수: CLOVA_API_KEY (네이버 클라우드 플랫폼 → AI·NAVER API → CLOVA Speech)
    docs: https://api.ncloud-docs.com/docs/ai-application-service-clovaspeech
    """

    ENDPOINT = "https://clovaspeech-gw.ncloud.com/recog/v1/stt:sync"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ["CLOVA_API_KEY"]

    @property
    def name(self) -> str:
        return "clova"

    def transcribe(self, audio_path: str) -> STTResult:
        audio_bytes = Path(audio_path).read_bytes()
        audio_duration_s = self._get_duration(audio_path)

        headers = {
            "X-CLOVASPEECH-API-KEY": self._api_key,
            "Content-Type": "application/octet-stream",
        }
        params = {"lang": "Kor", "completion": "sync"}

        t0 = time.perf_counter()
        resp = requests.post(
            self.ENDPOINT,
            headers=headers,
            params=params,
            data=audio_bytes,
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()

        data = resp.json()
        # CLOVA 응답: {"text": "...", "segments": [{"start": ms, "end": ms, "text": "..."}]}
        segments = []
        for seg in data.get("segments", []):
            segments.append(
                STTSegment(
                    start=seg["start"] / 1000.0,
                    end=seg["end"] / 1000.0,
                    text=seg.get("text", ""),
                )
            )
        if not segments:
            segments = [STTSegment(start=0.0, end=audio_duration_s, text=data.get("text", ""))]

        return STTResult(
            segments=segments,
            language="ko",
            processing_time_s=elapsed,  # 네트워크 지연 포함 — 별도 표기 필요
            audio_duration_s=audio_duration_s,
        )

    @staticmethod
    def _get_duration(audio_path: str) -> float:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration


class KakaoSTTRunner(BaseSTT):
    """
    Kakao Speech API (Newtone).
    환경변수: KAKAO_API_KEY (카카오 개발자 REST API 키)
    docs: https://developers.kakao.com/docs/latest/ko/voice/rest-api
    """

    ENDPOINT = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ["KAKAO_API_KEY"]

    @property
    def name(self) -> str:
        return "kakao"

    def transcribe(self, audio_path: str) -> STTResult:
        audio_bytes = Path(audio_path).read_bytes()
        audio_duration_s = self._get_duration(audio_path)

        headers = {
            "Authorization": f"KakaoAK {self._api_key}",
            "Content-Type": "application/octet-stream",
        }

        t0 = time.perf_counter()
        resp = requests.post(
            self.ENDPOINT,
            headers=headers,
            data=audio_bytes,
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()

        data = resp.json()
        # Kakao 응답: {"output": [{"type": "finalResult", "value": "..."}]}
        text = ""
        for item in data.get("output", []):
            if item.get("type") == "finalResult":
                text = item.get("value", "")
                break

        return STTResult(
            segments=[STTSegment(start=0.0, end=audio_duration_s, text=text)],
            language="ko",
            processing_time_s=elapsed,  # 네트워크 지연 포함 — 별도 표기 필요
            audio_duration_s=audio_duration_s,
        )

    @staticmethod
    def _get_duration(audio_path: str) -> float:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
