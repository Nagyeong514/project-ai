# -*- coding: utf-8 -*-
"""
무거운 모델을 로드하기 전에 5초 안에 죽을 수 있는 것들을 전부 검증한다.
(STEP6_품질검증/preflight.py와 같은 원칙 — docs/실행전_방어_체크리스트.md 참고)

이 파일은 통합 실행(run_all_clips.py/run_one_clip.py)용 — STT+YOLO+VLM+LLM 전부를
한 프로세스가 순차 로딩하므로 4가지 다 검사한다. STEP3/4/5 단독 실행은 각자 폴더의
가벼운 preflight.py(자기 몫만 검사)를 쓴다. 공통 로직은 tacit_common/preflight_base.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in ["STEP3_전처리", "STEP4_YOLO_VLM관찰", "STEP5_LLM으로_암묵지_후보생성"]:
    sys.path.insert(0, str(ROOT / _p))
sys.path.insert(0, str(ROOT))

import importlib
import os
import shutil
from typing import List

from tacit_common.preflight_base import check_config, check_imports, check_python_env

CORE_MODULES = ["cv2", "numpy", "PIL", "torch"]


def check_config_dependent_imports(cfg) -> List[str]:
    """config에 실제로 선택된 impl/backend에 필요한 패키지만 검사한다.
    예: llm.backend가 'hf_transformers'면 vllm은 안 깔려있어도 문제없음(실제로 미사용)."""
    missing: List[str] = []

    def _need(mod: str, why: str):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({why}에 필요, {e})")

    if cfg.detector.impl == "yolo_ultralytics":
        _need("ultralytics", "detector.impl=yolo_ultralytics")

    if cfg.stt.impl == "whisper_turbo":
        backend = cfg.stt.params.get("backend", "faster_whisper")
        _need("faster_whisper" if backend == "faster_whisper" else "whisper", f"stt.backend={backend}")

    if cfg.vlm.impl == "qwen3_vl":
        for mod in ["transformers", "accelerate", "qwen_vl_utils"]:
            _need(mod, "vlm.impl=qwen3_vl")
        if cfg.vlm.params.get("quantization") == "nf4":
            _need("bitsandbytes", "vlm.params.quantization=nf4")

    if cfg.llm.impl == "qwen2_5_14b":
        backend = cfg.llm.params.get("backend", "hf_transformers")
        if backend == "vllm":
            _need("vllm", "llm.backend=vllm")
        else:
            for mod in ["transformers", "accelerate"]:
                _need(mod, f"llm.backend={backend}")
            if cfg.llm.params.get("quantization") == "nf4":
                _need("bitsandbytes", "llm.params.quantization=nf4")

    return missing


def check_paths_and_env(cfg, video_paths: List[str]) -> List[str]:
    """경로/환경변수 프리플라이트."""
    problems: List[str] = []

    for video in video_paths:
        if not video:
            problems.append("video_path가 비어있음 (config.video_path 또는 --video로 지정)")
        elif not os.path.exists(video):
            problems.append(f"영상 파일 없음: {video}")

    if cfg.detector.impl == "yolo_ultralytics":
        weights = cfg.detector.params.get("weights_path", "")
        if not weights or not os.path.exists(weights):
            problems.append(f"YOLO weights_path 없음: {weights!r}")

    ffmpeg_bin = cfg.frame_extraction.ffmpeg_bin
    resolved_ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not resolved_ffmpeg or not (os.path.exists(ffmpeg_bin) if ffmpeg_bin else True):
        problems.append(f"ffmpeg 바이너리를 찾을 수 없음: {ffmpeg_bin!r}")

    # STEP3/4/5 산출물 경로들이 실제로 쓰기 가능한지 확인(경로 오타를 무거운 모델 로드 전에 잡기 위함)
    write_check_dirs = [
        cfg.paths.resolve(cfg.paths.step3_frames_dir),
        cfg.paths.resolve(cfg.paths.step3_transcript_dir),
        cfg.paths.resolve(cfg.paths.step4_detections_dir),
        cfg.paths.resolve(cfg.paths.step4_observations_dir),
        cfg.paths.resolve(cfg.paths.step5_aligned_windows_dir),
        cfg.paths.resolve(cfg.paths.step5_tacit_json_dir),
    ]
    for d in write_check_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except OSError as e:
            problems.append(f"'{d}' 폴더에 쓰기 불가: {e}")

    needs_gpu = any(
        c.params.get("device", "").startswith("cuda")
        for c in (cfg.detector, cfg.vlm, cfg.stt, cfg.llm)
    )
    if needs_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("config가 cuda를 요구하는데 CUDA 사용 불가 — GPU 할당 확인 필요")
        except ImportError:
            pass  # CORE 체크에서 이미 잡혔어야 함

    return problems


def run_preflight(config_path: str = "config.yaml", video_paths: List[str] | None = None):
    """run_all_clips.py / run_one_clip.py 진입점 맨 앞에서 호출한다. 실패하면 sys.exit(1).
    성공하면 로드된 PipelineConfig를 반환(호출부에서 다시 로드할 필요 없음)."""
    all_problems: List[str] = []

    print("[1/4] 코어 패키지 import 확인")
    core_missing = check_imports(CORE_MODULES)
    all_problems.extend(core_missing)
    print("      OK" if not core_missing else "")

    print("[2/4] 파이썬 환경 확인")
    check_python_env()

    print("[3/4] config 검증 + impl별 필요 패키지 확인")
    cfg, config_problems = check_config(config_path)
    all_problems.extend(config_problems)
    if cfg is not None:
        cfg_missing = check_config_dependent_imports(cfg)
        all_problems.extend(cfg_missing)
        if not config_problems and not cfg_missing:
            print("      OK")

    print("[4/4] 경로/GPU 프리플라이트")
    if cfg is not None:
        paths = video_paths if video_paths is not None else [cfg.video_path]
        path_problems = check_paths_and_env(cfg, paths)
        all_problems.extend(path_problems)
        if not path_problems:
            print("      OK")
    else:
        print("      건너뜀 (config 로드 실패)")

    if all_problems:
        print("\n[FAIL] 프리플라이트 실패 — 무거운 모델 로드 안 함:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\n[PASS] 프리플라이트 통과 — 본 실행 진행합니다.\n")
    return cfg


if __name__ == "__main__":
    # 단독 실행: 본 실행 전에 값싼 자원으로 미리 확인만 하고 싶을 때.
    #   srun -p RTX6000 -w n5 --gres=gpu:1 --time=00:02:00 python3 preflight.py
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    run_preflight(args.config, [args.video] if args.video else None)
