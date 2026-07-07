"""STEP4(YOLO 검출 + VLM 관찰) 전용 구현 레지스트리. 원칙 1(모델 교체 용이성):
새 구현 = step4_components/ 에 클래스 1개 + 여기 한 줄 + config의 impl 한 줄.

sampler(SAMPLERS)는 등록만 되고 이 STEP의 러너가 실제로 build_sampler를 호출하지
않는다 — 본선(native_video)에서 완전 미사용(2026-07-04 분할 계획서 결정, 삭제 안 함)."""

from __future__ import annotations

from typing import Any, Callable, Dict

from step4_components.detector_noop import NoopDetector
from step4_components.detector_yolo import UltralyticsYOLODetector
from step4_components.sampler_motion import MotionGuidedSampler, UniformSampler
from step4_components.vlm_qwen import QwenVLActionExtractor
from tacit_common.config import ComponentConfig

DETECTORS: Dict[str, Callable[..., Any]] = {
    "yolo_ultralytics": UltralyticsYOLODetector,
    "noop": NoopDetector,
}
VLMS: Dict[str, Callable[..., Any]] = {
    "qwen3_vl": QwenVLActionExtractor,
}
SAMPLERS: Dict[str, Callable[..., Any]] = {  # 미사용 — 위 모듈 docstring 참고
    "uniform": UniformSampler,
    "motion_guided": MotionGuidedSampler,
}


def _build(table: Dict[str, Callable[..., Any]], cfg: ComponentConfig, kind: str):
    if cfg.impl not in table:
        raise KeyError(
            f"{kind} 구현 '{cfg.impl}' 없음. 등록된 것: {list(table)}\n"
            f"  → 새 구현이면 registry.py 의 {kind} 딕셔너리에 한 줄 추가하세요."
        )
    return table[cfg.impl](**cfg.params)


def build_detector(cfg: ComponentConfig):
    return _build(DETECTORS, cfg, "detector")


def build_vlm(cfg: ComponentConfig):
    return _build(VLMS, cfg, "vlm")


def build_sampler(cfg: ComponentConfig):  # 미사용(본선 아님) — 호출부 없음
    return _build(SAMPLERS, cfg, "sampler")
