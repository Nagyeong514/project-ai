# -*- coding: utf-8 -*-
"""
무거운 모델을 로드하기 전에 5초 안에 죽을 수 있는 것들을 전부 검증한다.

원칙: "무거운 거(모델 로드) 위로는 전부 5초 안에 죽을 수 있는 검증만, 무거운 거
아래로 내려간 에러만 진짜 GPU/로직 문제로 취급한다."

⚠️ 이 파일 자체는 반드시 표준 라이브러리(importlib/os/sys/glob)만 써야 한다.
   langgraph/langchain/torch 같은 무거운 패키지를 이 파일이 import해버리면,
   "그 패키지가 없는지 확인하는 코드"가 그 패키지 없어서 죽는 모순이 생긴다.
   실제로 2026-07-03 사고가 정확히 이 패턴이었다: main.py 최상단에서
   `from graph import build_graph`를 무조건 실행해서, langgraph가 없으면
   run() 안의 어떤 체크에도 도달하지 못하고 죽었다. 그래서:
     1) 이 프리플라이트는 무조건 다른 어떤 프로젝트 모듈보다 먼저 실행돼야 하고,
     2) main.py 쪽에서도 embedding_utils/llm_utils/graph의 import를
        run_preflight() 호출 '이후'로 옮겨야 한다(그래야 실제로 강제됨).
"""
from __future__ import annotations

import glob
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# requirements.txt 기준. 여기 이름이 바뀌면 이 리스트도 같이 바꿀 것.
REQUIRED_MODULES = [
    "langgraph",
    "langchain_core",
    "langchain_huggingface",
    "langchain_qdrant",
    "qdrant_client",
    "transformers",
    "accelerate",
    "torch",
    "sentencepiece",
    "pydantic",
    "sentence_transformers",  # 2026-07-03: HuggingFaceEmbeddings 런타임 요구, 실행해보고 발견
    "distro",  # 2026-07-03: langchain_core 내부에서 요구, 로그인노드/n5 패키지 불일치로 발견
]


def check_imports() -> list[str]:
    """[1] 필수 패키지 import 확인 — 2026-07-03 실제 사고 원인. 제일 먼저 확인한다.

    ModuleNotFoundError만 잡으면 안 된다 — langchain_core처럼 지연 import를 쓰는
    패키지는 내부에서 다른 하위 의존성이 없을 때 일반 ImportError를 직접 던진다
    (예: langchain_core.runnables가 'distro' 없어서 raise ImportError(...) — 이건
    ModuleNotFoundError가 아니라 ImportError라서 좁게 잡으면 그냥 크래시한다.
    실측: 로그인 노드엔 distro가 있는데 n5엔 없어서 이 경로로 실제로 터짐).
    """
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({e})")
    return missing


def check_python_env() -> list[str]:
    """[2] venv/경로 확인. `which python3`에 해당하는 정보를 로그로 남긴다.

    셸 레벨 `set -euo pipefail` + `source venv/bin/activate`를 대체하진 못한다
    (그건 sbatch 스크립트 쪽 책임) — 여기서는 최소한 "지금 이 프로세스가 어느
    파이썬으로 떠 있는지"를 실행 로그에 박아서, 나중에 "엉뚱한 파이썬으로
    돌았다" 같은 걸 사후에라도 확인할 수 있게 한다.
    """
    print(f"      python executable : {sys.executable}")
    print(f"      python version    : {sys.version.split()[0]}")
    return []  # 정보성 로그. 판단 기준이 없어 강제 실패는 안 시킴.


def check_config() -> list[str]:
    """[3] config 검증. pydantic Config()가 예외 없이 생성되면 통과.

    타입/필수값 오류는 `from config import CONFIG`를 import하는 시점에
    pydantic이 이미 예외를 던지므로, 여기 도달했다는 것 자체가 통과를 의미한다.
    """
    try:
        from config import CONFIG  # noqa: F401  (import 자체가 검증)
    except Exception as e:  # pydantic ValidationError 포함
        return [f"config.py 검증 실패: {e}"]
    return []


def check_paths_and_env(require_gpu: bool = True) -> list[str]:
    """[4] 경로/환경변수 프리플라이트 — 출력폴더 쓰기 가능, 입력파일 존재, GPU."""
    problems: list[str] = []

    manuals = glob.glob(os.path.join(BASE_DIR, "manuals", "*"))
    if not manuals:
        problems.append("manuals/ 폴더가 비어있음 (Gate A RAG용 매뉴얼 최소 1개 필요)")

    inputs = glob.glob(os.path.join(BASE_DIR, "input", "*.json"))
    if not inputs:
        problems.append("input/ 폴더에 검증할 후보 JSON이 없음")

    output_dir = os.path.join(BASE_DIR, "output")
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
    except OSError as e:
        problems.append(f"output/ 폴더에 쓰기 불가: {e}")

    if require_gpu:
        try:
            import torch  # torch는 check_imports에서 이미 확인됐어야 정상 도달
            if not torch.cuda.is_available():
                problems.append("CUDA 사용 불가(torch.cuda.is_available()=False) — GPU 할당 확인 필요")
        except ModuleNotFoundError:
            problems.append("torch가 없어 GPU 확인 불가(1단계에서 이미 걸렸어야 함)")

    return problems


def run_preflight(require_gpu: bool = True) -> None:
    """main.py의 run() 맨 첫 줄에서 호출한다. 실패하면 즉시 sys.exit(1).

    호출 순서가 중요하다: main.py가 embedding_utils/llm_utils/graph를
    import하기 '전에' 이 함수부터 호출돼야 한다. 그래야 check_imports()가
    실패했을 때 무거운 모듈 import 자체를 안 밟고 죽는다.
    """
    steps = [
        ("[1/4] 필수 패키지 import 확인", lambda: check_imports_as_problems()),
        ("[2/4] 파이썬 환경 확인", check_python_env),
        ("[3/4] config 검증", check_config),
        ("[4/4] 경로/GPU 프리플라이트", lambda: check_paths_and_env(require_gpu)),
    ]

    all_problems: list[str] = []
    for label, fn in steps:
        print(label)
        problems = fn()
        all_problems.extend(problems)
        if not problems:
            print("      OK")

    if all_problems:
        print("\n[FAIL] 프리플라이트 실패 — 무거운 모델 로드 안 함:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\n[PASS] 프리플라이트 통과 — 본 실행 진행합니다.\n")


def check_imports_as_problems() -> list[str]:
    missing = check_imports()
    if missing:
        return [f"누락된 패키지: {missing} → pip install -r requirements.txt 로 설치"]
    return []


if __name__ == "__main__":
    # 단독 실행: 본 실행(main.py) 전에 값싼 자원으로 미리 확인만 하고 싶을 때.
    #   srun -p RTX6000 -w n5 --gres=gpu:1 --time=00:02:00 python3 preflight.py
    run_preflight()
