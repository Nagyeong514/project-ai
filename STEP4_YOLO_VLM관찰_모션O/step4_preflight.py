"""STEP4(YOLO 검출 + VLM 관찰) 전용 프리플라이트. 공통 뼈대는 tacit_common/preflight_base.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import importlib
import os
from typing import List

from tacit_common.preflight_base import run_preflight as _run_preflight


def _config_dependent_imports(cfg) -> List[str]:
    missing: List[str] = []

    def _need(mod: str, why: str):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({why}에 필요, {e})")

    if cfg.detector.impl == "yolo_ultralytics":
        _need("ultralytics", "detector.impl=yolo_ultralytics")

    if cfg.vlm.impl == "qwen3_vl":
        for mod in ["transformers", "accelerate", "qwen_vl_utils"]:
            _need(mod, "vlm.impl=qwen3_vl")
        if cfg.vlm.params.get("quantization") == "nf4":
            _need("bitsandbytes", "vlm.params.quantization=nf4")
    return missing


def _path_check(cfg) -> List[str]:
    problems: List[str] = []
    if cfg.detector.impl == "yolo_ultralytics":
        weights = cfg.detector.params.get("weights_path", "")
        if not weights or not os.path.exists(weights):
            problems.append(f"YOLO weights_path 없음: {weights!r}")

    needs_gpu = any(
        c.params.get("device", "").startswith("cuda") for c in (cfg.detector, cfg.vlm)
    )
    if needs_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("detector/vlm이 cuda를 요구하는데 CUDA 사용 불가")
        except ImportError:
            pass
    return problems


def run_preflight(config_path: str = "../파이프라인_통합실행/config.yaml"):
    return _run_preflight(config_path, ["cv2", "numpy", "PIL", "torch"],
                           _config_dependent_imports, _path_check)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    args = ap.parse_args()
    run_preflight(args.config)
