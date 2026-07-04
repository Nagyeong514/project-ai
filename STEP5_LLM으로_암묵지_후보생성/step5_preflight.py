"""STEP5(정렬 + LLM 융합) 전용 프리플라이트. 공통 뼈대는 tacit_common/preflight_base.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import importlib
from typing import List

from tacit_common.preflight_base import run_preflight as _run_preflight


def _config_dependent_imports(cfg) -> List[str]:
    missing: List[str] = []

    def _need(mod: str, why: str):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({why}에 필요, {e})")

    if cfg.llm.impl == "qwen2_5_14b":
        backend = cfg.llm.params.get("backend", "hf_transformers")
        if backend == "vllm":
            _need("vllm", "llm.backend=vllm")  # 이 클러스터에선 근본적으로 불가(CLAUDE.md §2) — 실수 조기 발견용
        else:
            for mod in ["transformers", "accelerate"]:
                _need(mod, f"llm.backend={backend}")
            if cfg.llm.params.get("quantization") == "nf4":
                _need("bitsandbytes", "llm.params.quantization=nf4")
    return missing


def _path_check(cfg) -> List[str]:
    problems: List[str] = []
    if cfg.llm.params.get("device", "").startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("llm이 cuda를 요구하는데 CUDA 사용 불가")
        except ImportError:
            pass
    return problems


def run_preflight(config_path: str = "../파이프라인_통합실행/config.yaml"):
    return _run_preflight(config_path, ["torch"], _config_dependent_imports, _path_check)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    args = ap.parse_args()
    run_preflight(args.config)
