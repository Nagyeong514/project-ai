"""STT (faster-whisper large-v3-turbo).

최적화(1차 교훈): 매 구간마다 wav를 통째로 재전사하면 느림.
→ 전체 1회 타임스탬프 전사 후, 각 구간 window에 겹치는 발화를 모아 배분.
"""
from __future__ import annotations
import os
from functools import lru_cache

os.environ.setdefault("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_path, device=device, compute_type=compute_type)


@lru_cache(maxsize=4)
def transcribe_timed(wav_path: str, cfg_key: tuple) -> tuple:
    """전체 1회 전사 → ((start,end,text), ...). cfg_key로 캐시."""
    model_path, device, compute_type, language = cfg_key
    model = _load_model(model_path, device, compute_type)
    segs, _ = model.transcribe(wav_path, language=language)
    return tuple((s.start, s.end, s.text.strip()) for s in segs)


def _cfg_key(cfg):
    s = cfg["stt"]
    return (s["model_path"], s["device"], s["compute_type"], s["language"])


def assign_to_segments(wav_path, segments, cfg) -> list[str]:
    """각 구간 window에 겹치는 전사를 모아 텍스트로 반환 (구간 수와 1:1)."""
    timed = transcribe_timed(wav_path, _cfg_key(cfg))
    out = []
    for a, b in segments:
        parts = [t for (s, e, t) in timed if e > a and s < b]  # 겹침
        out.append(" ".join(parts).strip())
    return out


def transcribe_full(wav_path: str, cfg: dict) -> str:
    timed = transcribe_timed(wav_path, _cfg_key(cfg))
    return " ".join(t for (_, _, t) in timed).strip()
