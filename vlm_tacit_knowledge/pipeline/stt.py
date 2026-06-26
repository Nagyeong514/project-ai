"""구간별 STT (faster-whisper large-v3-turbo).

STT 연구 선정 모델 재사용. 발화 구간 [(start,end)] 각각을 전사.
모델은 1회 로드 후 재사용 (싱글톤).
"""
from __future__ import annotations
import os
from functools import lru_cache

# cuBLAS 경로 (앞 연구 교훈) — 환경에 없으면 호출측에서 export 필요
os.environ.setdefault("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_path, device=device, compute_type=compute_type)


def transcribe_segments(wav_path: str, segments: list[tuple[float, float]], cfg: dict) -> list[str]:
    """각 (start,end) 구간의 전사 텍스트 리스트 반환 (구간 수와 1:1)."""
    s = cfg["stt"]
    model = _load_model(s["model_path"], s["device"], s["compute_type"])
    texts = []
    for start, end in segments:
        segs, _ = model.transcribe(
            wav_path, language=s["language"],
            clip_timestamps=[start, end],   # 해당 구간만 전사
        )
        texts.append(" ".join(x.text.strip() for x in segs).strip())
    return texts


def transcribe_full(wav_path: str, cfg: dict) -> str:
    """조건 A용 전체 전사."""
    s = cfg["stt"]
    model = _load_model(s["model_path"], s["device"], s["compute_type"])
    segs, _ = model.transcribe(wav_path, language=s["language"])
    return " ".join(x.text.strip() for x in segs).strip()
