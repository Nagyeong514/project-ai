"""STEP3(STT+정제, 프레임 추출) 전용 프리플라이트 — 무거운 모델 로드 전 5초 안에
죽을 수 있는 것만 검증(CLAUDE.md §0/§1). 공통 뼈대는 tacit_common/preflight_base.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import importlib
import os
import shutil
from typing import List

from tacit_common.preflight_base import run_preflight as _run_preflight


def _config_dependent_imports(cfg) -> List[str]:
    missing: List[str] = []

    def _need(mod: str, why: str):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({why}에 필요, {e})")

    if cfg.stt.impl == "whisper_turbo":
        backend = cfg.stt.params.get("backend", "faster_whisper")
        _need("faster_whisper" if backend == "faster_whisper" else "whisper", f"stt.backend={backend}")
    return missing


def _path_check(cfg) -> List[str]:
    problems: List[str] = []
    ffmpeg_bin = cfg.frame_extraction.ffmpeg_bin
    resolved = ffmpeg_bin or shutil.which("ffmpeg")
    if not resolved or (ffmpeg_bin and not os.path.exists(ffmpeg_bin)):
        problems.append(f"ffmpeg 바이너리를 찾을 수 없음: {ffmpeg_bin!r}")

    if cfg.stt.params.get("device", "").startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("stt가 cuda를 요구하는데 CUDA 사용 불가 — GPU 할당 확인 필요")
        except ImportError:
            pass  # 코어 체크에서 이미 잡혔어야 함
    return problems


def run_preflight(config_path: str = "../파이프라인_통합실행/config.yaml"):
    """run_step3.py 진입점 맨 앞에서 호출. 성공하면 PipelineConfig 반환."""
    return _run_preflight(config_path, ["cv2", "numpy", "torch"], _config_dependent_imports, _path_check)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    args = ap.parse_args()
    run_preflight(args.config)
