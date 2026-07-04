# -*- coding: utf-8 -*-
"""
00_사전연구(01_STT선정, 02_VAD재검증_v2) 실험 스크립트를 돌리기 전에 먼저 실행하는 프리플라이트.

STEP3_전처리/STEP6_품질검증과 다른 점: 여기는 `run.py` 하나로 모이는 단일 진입점이 없고
독립 실험 스크립트가 10개 넘게 있다(run_experiment_parallel.py, run_vad_experiment.py,
run_seamless_fix.py, run_ablation_stage.py, run_final_comparison.py, ...). 스크립트마다
프리플라이트를 끼워 넣는 대신, **아무 실험 스크립트나 돌리기 전에 먼저 한 번 이 파일을
단독 실행**하는 용도로 쓴다:

    srun -p RTX6000 -w n5 --gres=gpu:1 --time=00:02:00 python3 00_사전연구/preflight.py

사용법:
    srun ... python3 00_사전연구/preflight.py

참고: docs/실행전_방어_체크리스트.md
"""
from __future__ import annotations

import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 01_STT선정/requirements_v2.txt + 02_VAD재검증_v2 스크립트 실제 import를 훑어서 뽑은 목록.
# ⚠️ 01_STT선정/requirements_v2.txt가 `-r requirements.txt`로 v1 파일을 참조하는데 그 파일
#    자체가 폴더에 없다(유실 추정) — 그래서 여기 리스트는 "실제 import문"을 근거로 만들었지,
#    requirements 파일을 그대로 신뢰하지 않았다. 새 실험 스크립트를 추가하면 이 리스트도 갱신할 것.
REQUIRED_MODULES = [
    "torch",
    "yaml",
    "faster_whisper",   # 01_STT선정 STT 러너
    "jiwer",             # CER/WER 채점
    "soundfile",          # 오디오 I/O
    "transformers",       # SeamlessM4Tv2ForSpeechToText, Wav2Vec2ForCTC
    "sentencepiece",       # Seamless-M4T-v2 토크나이저
    "silero_vad",          # 02_VAD재검증_v2 VAD 트리밍
]


def check_imports() -> list[str]:
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({e})")
    return missing


def check_python_env() -> list[str]:
    print(f"      python executable : {sys.executable}")
    print(f"      python version    : {sys.version.split()[0]}")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"      LD_LIBRARY_PATH   : {ld or '(비어있음)'}")
    # requirements_v2.txt 주석은 "LD_LIBRARY_PATH로 nvidia-cublas-cu12 lib/ 경로를 잡아줘야
    # 한다"고 했으나, 2026-07-03 n5에서 실제로 WhisperModel(device='cuda')을 로딩해보니
    # LD_LIBRARY_PATH 보정 없이도 성공했다(pip가 깐 nvidia-cublas-cu12/nvidia-cudnn-cu12
    # 패키지의 RPATH로 ctranslate2가 자체 해결하는 것으로 보임). 그래서 여기서는 경고를
    # 내지 않는다 — 실측 안 하고 옛 comment만 믿었으면 거짓 경고를 계속 냈을 뻔했다.
    return []


def check_gpu() -> list[str]:
    problems = []
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append("CUDA 사용 불가 — GPU 할당 확인 필요")
    except ImportError:
        pass  # 이미 check_imports에서 잡혔을 것
    return problems


def run_preflight() -> None:
    all_problems: list[str] = []

    print("[1/3] 필수 패키지 import 확인")
    missing = check_imports()
    all_problems.extend(missing)
    if not missing:
        print("      OK")

    print("[2/3] 파이썬 환경 확인")
    check_python_env()

    print("[3/3] GPU 확인")
    gpu_problems = check_gpu()
    all_problems.extend(gpu_problems)
    if not gpu_problems:
        print("      OK")

    if all_problems:
        print("\n[FAIL] 프리플라이트 실패:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\n[PASS] 프리플라이트 통과 — 실험 스크립트 실행 가능.")


if __name__ == "__main__":
    run_preflight()
