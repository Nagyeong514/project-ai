# -*- coding: utf-8 -*-
"""
무거운 모델(임베딩) 로드 전 5초 안에 죽을 수 있는 것들을 전부 검증한다.
STEP7_DB/preflight.py와 동일 원칙(docs/실행전_방어_체크리스트.md 참고) — 이 폴더의
실제 의존성(qdrant-client, sentence-transformers, fastapi 등)에 맞게 목록만 교체.

이 파일 자체는 표준 라이브러리(importlib/os/sys/glob)만 쓴다 — "패키지 없는지
확인하는 코드"가 그 패키지가 없어서 죽는 모순을 피하기 위함.
"""
from __future__ import annotations

import glob
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# requirements.txt 기준. 여기 이름이 바뀌면 이 리스트도 같이 바꿀 것.
REQUIRED_MODULES = [
    "qdrant_client",
    "sentence_transformers",
    "pydantic",
    "requests",
    "fastapi",
    "uvicorn",
]


def check_imports() -> list[str]:
    """[1] 필수 패키지 import 확인."""
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({e})")
    return [f"누락된 패키지: {missing} → pip install -r requirements.txt 로 설치"] if missing else []


def check_python_env() -> list[str]:
    """[2] venv/경로 확인 — 정보성 로그."""
    print(f"      python executable : {sys.executable}")
    print(f"      python version    : {sys.version.split()[0]}")
    return []


def check_config() -> list[str]:
    """[3] config 검증. pydantic Config()가 예외 없이 생성되면 통과."""
    try:
        from config import CONFIG  # noqa: F401  (import 자체가 검증)
    except Exception as e:  # pydantic ValidationError 포함
        return [f"config.py 검증 실패: {e}"]
    return []


def check_paths_and_env() -> list[str]:
    """[4] 경로/환경변수 프리플라이트 — 입력 JSON 존재, DB 폴더 쓰기 가능, (GPU 쓸 때만) CUDA."""
    problems: list[str] = []
    from config import CONFIG

    inputs = glob.glob(os.path.join(CONFIG.input_dir, "*.json"))
    if not inputs:
        problems.append(f"'{CONFIG.input_dir}'에 적재할 암묵지 JSON이 없음")

    try:
        os.makedirs(CONFIG.qdrant_path, exist_ok=True)
        test_file = os.path.join(CONFIG.qdrant_path, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
    except OSError as e:
        problems.append(f"'{CONFIG.qdrant_path}' 폴더에 쓰기 불가: {e}")

    if CONFIG.device == "cuda":
        try:
            import torch  # torch는 check_imports 대상은 아니지만(선택 의존성), 여기서 실측
            if not torch.cuda.is_available():
                problems.append("config.device=cuda인데 CUDA 사용 불가 — GPU 할당 확인 필요")
        except ModuleNotFoundError:
            problems.append("device=cuda인데 torch가 없어 GPU 확인 불가")

    return problems


def run_preflight() -> None:
    """ingest.py/app.py의 진입 함수 맨 첫 줄에서 호출한다. 실패하면 즉시 sys.exit(1)."""
    steps = [
        ("[1/4] 필수 패키지 import 확인", check_imports),
        ("[2/4] 파이썬 환경 확인", check_python_env),
        ("[3/4] config 검증", check_config),
        ("[4/4] 경로/환경 프리플라이트", check_paths_and_env),
    ]

    all_problems: list[str] = []
    for label, fn in steps:
        print(label)
        problems = fn()
        all_problems.extend(problems)
        if not problems:
            print("      OK")

    if all_problems:
        print("\n[FAIL] 프리플라이트 실패 — 무거운 임베딩 모델 로드 안 함:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\n[PASS] 프리플라이트 통과 — 본 실행 진행합니다.\n")


if __name__ == "__main__":
    # 단독 실행: 본 실행(ingest.py/app.py) 전에 값싼 자원으로 미리 확인만 하고 싶을 때.
    run_preflight()
