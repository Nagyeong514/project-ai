"""STEP5(정렬 + LLM 융합) 전용 구현 레지스트리. 원칙 1(모델 교체 용이성):
새 구현 = step5_components/ 에 클래스 1개 + 여기 한 줄 + config의 impl 한 줄."""

from __future__ import annotations

from typing import Any, Callable, Dict

from step5_components.aligner import WindowAligner
from step5_components.llm_fusion import QwenLLMFusion
from tacit_common.config import ComponentConfig

ALIGNERS: Dict[str, Callable[..., Any]] = {
    "window": WindowAligner,
}
LLMS: Dict[str, Callable[..., Any]] = {
    "qwen2_5_14b": QwenLLMFusion,
}


def _build(table: Dict[str, Callable[..., Any]], cfg: ComponentConfig, kind: str):
    if cfg.impl not in table:
        raise KeyError(
            f"{kind} 구현 '{cfg.impl}' 없음. 등록된 것: {list(table)}\n"
            f"  → 새 구현이면 registry.py 의 {kind} 딕셔너리에 한 줄 추가하세요."
        )
    return table[cfg.impl](**cfg.params)


def build_aligner(cfg: ComponentConfig):
    return _build(ALIGNERS, cfg, "aligner")


def build_llm(cfg: ComponentConfig):
    return _build(LLMS, cfg, "llm")
