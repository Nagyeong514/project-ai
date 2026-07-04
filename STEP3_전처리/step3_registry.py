"""STEP3(STT+정제, 프레임 추출) 전용 구현 레지스트리. 원칙 1(모델 교체 용이성):
새 구현 = step3_components/ 에 클래스 1개 + 여기 한 줄 + config의 impl 한 줄."""

from __future__ import annotations

from typing import Any, Callable, Dict

from step3_components.stt_whisper import WhisperTurboSTT
from step3_components.transcript_refine import NormalizeRefiner
from tacit_common.config import ComponentConfig

STTS: Dict[str, Callable[..., Any]] = {
    "whisper_turbo": WhisperTurboSTT,
}
REFINERS: Dict[str, Callable[..., Any]] = {
    "normalize": NormalizeRefiner,
}


def _build(table: Dict[str, Callable[..., Any]], cfg: ComponentConfig, kind: str):
    if cfg.impl not in table:
        raise KeyError(
            f"{kind} 구현 '{cfg.impl}' 없음. 등록된 것: {list(table)}\n"
            f"  → 새 구현이면 registry.py 의 {kind} 딕셔너리에 한 줄 추가하세요."
        )
    return table[cfg.impl](**cfg.params)


def build_stt(cfg: ComponentConfig):
    return _build(STTS, cfg, "stt")


def build_refiner(cfg: ComponentConfig):
    return _build(REFINERS, cfg, "transcript_refine")
