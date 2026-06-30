"""
구현 레지스트리 — config의 `impl` 문자열을 실제 클래스로 매핑(팩토리).

원칙 1(모델 교체 용이성)의 핵심 스위치:
  새 모델 추가 = components/ 에 클래스 1개 + 여기 딕셔너리 한 줄 + config의 impl 한 줄.
  기존 코드는 안 건드린다.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .components import (
    MotionGuidedSampler,
    QwenLLMFusion,
    QwenVLActionExtractor,
    NormalizeRefiner,
    UltralyticsYOLODetector,
    UniformSampler,
    WhisperTurboSTT,
    WindowAligner,
)
from .config import ComponentConfig

# 컴포넌트 종류별 레지스트리. key=config의 impl, value=생성자.
SAMPLERS: Dict[str, Callable[..., Any]] = {
    "uniform": UniformSampler,
    "motion_guided": MotionGuidedSampler,
}
DETECTORS: Dict[str, Callable[..., Any]] = {
    "yolo_ultralytics": UltralyticsYOLODetector,
}
VLMS: Dict[str, Callable[..., Any]] = {
    "qwen3_vl": QwenVLActionExtractor,
}
STTS: Dict[str, Callable[..., Any]] = {
    "whisper_turbo": WhisperTurboSTT,
}
REFINERS: Dict[str, Callable[..., Any]] = {
    "normalize": NormalizeRefiner,
}
LLMS: Dict[str, Callable[..., Any]] = {
    "qwen2_5_14b": QwenLLMFusion,
}
ALIGNERS: Dict[str, Callable[..., Any]] = {
    "window": WindowAligner,
}


def _build(table: Dict[str, Callable[..., Any]], cfg: ComponentConfig, kind: str):
    if cfg.impl not in table:
        raise KeyError(
            f"{kind} 구현 '{cfg.impl}' 없음. 등록된 것: {list(table)}\n"
            f"  → 새 구현이면 registry.py 의 {kind} 딕셔너리에 한 줄 추가하세요."
        )
    return table[cfg.impl](**cfg.params)


def build_sampler(cfg: ComponentConfig):
    return _build(SAMPLERS, cfg, "sampler")


def build_detector(cfg: ComponentConfig):
    return _build(DETECTORS, cfg, "detector")


def build_vlm(cfg: ComponentConfig):
    return _build(VLMS, cfg, "vlm")


def build_stt(cfg: ComponentConfig):
    return _build(STTS, cfg, "stt")


def build_refiner(cfg: ComponentConfig):
    return _build(REFINERS, cfg, "transcript_refine")


def build_llm(cfg: ComponentConfig):
    return _build(LLMS, cfg, "llm")


def build_aligner(cfg: ComponentConfig):
    return _build(ALIGNERS, cfg, "aligner")
