# -*- coding: utf-8 -*-
"""
무거운 모델을 로드하기 전에 5초 안에 죽을 수 있는 것들을 전부 검증한다.
(STEP6_품질검증/preflight.py와 같은 원칙 — docs/실행전_방어_체크리스트.md 참고)

STEP3는 STEP6과 다른 점이 하나 있다: `tacit_pipeline`의 모든 어댑터(YOLO/STT/VLM/LLM)가
`_load()` 안에서만 무거운 패키지를 지연 import하도록 이미 설계돼 있어서(예:
`detector_yolo.py`의 `from ultralytics import YOLO`는 `_load()` 안에만 있음),
`import tacit_pipeline` 자체는 ultralytics/torch/transformers/faster_whisper 없이도
0.4초 만에 끝난다(실측, 2026-07-03). 즉 STEP6처럼 "main() 첫 줄 체크가 이미 늦다"는
문제가 여기선 없다 — `run.py`/`run_all_clips.py` 맨 앞에서 이 프리플라이트를 부르는 것만으로
충분히 강제된다. 다만 필요 패키지는 config의 impl 선택(예: llm.backend)에 따라 달라지므로,
이 파일은 고정 리스트가 아니라 **로드된 config를 보고 실제로 필요한 것만** 검사한다.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# config 값과 무관하게 항상 필요(코어: config 로더, 영상 전처리 공용).
CORE_MODULES = ["pydantic", "yaml", "cv2", "numpy", "PIL", "torch"]

# config의 impl/backend 선택에 따라 조건부로 필요한 것들.
# (impl 문자열, backend 문자열 or None) -> 모듈명. registry.py의 실제 매핑과 짝을 맞춘다.
CONDITIONAL_MODULES = {
    ("detector.impl", "yolo_ultralytics"): "ultralytics",
    ("stt.impl", "whisper_turbo"): {"faster_whisper": "faster_whisper", "openai_whisper": "whisper"},
    ("vlm.impl", "qwen3_vl"): ["transformers", "accelerate", "qwen_vl_utils"],
    ("llm.impl", "qwen2_5_14b"): {"hf_transformers": ["transformers", "accelerate"], "vllm": ["vllm"]},
}


def check_imports(missing_only: bool = True) -> List[str]:
    """[1] 필수 패키지 import 확인. CORE는 무조건, CONDITIONAL은 config 로드 후에만 정확히 알 수 있어
    여기서는 CORE만 본다 — 나머지는 check_config_dependent_imports()가 담당."""
    missing = []
    for mod in CORE_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({e})")
    return missing


def check_config_dependent_imports(cfg) -> List[str]:
    """config에 실제로 선택된 impl/backend에 필요한 패키지만 검사한다.
    예: llm.backend가 'hf_transformers'면 vllm은 안 깔려있어도 문제없음(실제로 미사용).
    """
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


def check_python_env() -> List[str]:
    """[2] venv/경로 확인. 이 클러스터 고유 함정(LD_LIBRARY_PATH 덮어쓰기 사고,
    reference-cluster-specs 참고)도 여기서 눈에 보이게 로그로 남긴다."""
    print(f"      python executable : {sys.executable}")
    print(f"      python version    : {sys.version.split()[0]}")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"      LD_LIBRARY_PATH   : {ld or '(비어있음)'}")
    if ld and "cuda" not in ld.lower():
        print("      [주의] LD_LIBRARY_PATH에 cuda 관련 경로가 안 보임 — "
              "덮어써서 libnvrtc.so.13이 날아갔을 가능성 확인 필요")
    return []


def check_config(config_path: str):
    """[3] config 검증. PipelineConfig.load()가 예외 없이 성공하면 통과(이미 pydantic).
    반환값: 성공하면 로드된 PipelineConfig, 실패하면 None."""
    from tacit_pipeline.config import PipelineConfig

    try:
        return PipelineConfig.load(config_path), []
    except Exception as e:  # FileNotFoundError, pydantic ValidationError 등
        return None, [f"config 로드 실패: {e}"]


def check_paths_and_env(cfg, video_paths: List[str]) -> List[str]:
    """[4] 경로/환경변수 프리플라이트."""
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

    ffmpeg_bin = cfg.vlm.params.get("ffmpeg_bin")
    resolved_ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not resolved_ffmpeg or not (os.path.exists(resolved_ffmpeg) if ffmpeg_bin else True):
        problems.append(f"ffmpeg 바이너리를 찾을 수 없음: {ffmpeg_bin!r}")

    for d in (cfg.output_dir, cfg.transcript_dir):
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
    """run.py / run_all_clips.py 진입점 맨 앞에서 호출한다. 실패하면 sys.exit(1).
    성공하면 로드된 PipelineConfig를 반환(호출부에서 다시 로드할 필요 없음)."""
    all_problems: List[str] = []

    print("[1/4] 코어 패키지 import 확인")
    core_missing = check_imports()
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
