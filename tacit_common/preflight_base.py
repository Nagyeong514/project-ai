"""
STEP3/4/5 + 파이프라인_통합실행이 공유하는 프리플라이트 뼈대(CLAUDE.md §0/§1 원칙:
무거운 모델 로드 위로는 전부 5초 안에 죽을 수 있는 검증만).

각 STEP은 자신에게 필요한 조건부 모듈 검사 함수만 만들어서 run_preflight()에 넘긴다 —
고정 리스트로 다 검사하지 않고, config에서 실제로 선택된 impl/backend에 필요한
패키지만 검사한다는 원칙은 그대로 유지.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Callable, List, Optional

CORE_MODULES = ["pydantic", "yaml"]


def check_imports(modules: List[str]) -> List[str]:
    missing = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({e})")
    return missing


def check_python_env() -> None:
    print(f"      python executable : {sys.executable}")
    print(f"      python version    : {sys.version.split()[0]}")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"      LD_LIBRARY_PATH   : {ld or '(비어있음)'}")
    if ld and "cuda" not in ld.lower():
        print("      [주의] LD_LIBRARY_PATH에 cuda 관련 경로가 안 보임 — "
              "덮어써서 libnvrtc.so.13이 날아갔을 가능성 확인 필요")


def check_config(config_path: str):
    """PipelineConfig.load()가 예외 없이 성공하면 통과. 반환: (cfg 또는 None, 문제 리스트)."""
    from tacit_common.config import PipelineConfig

    try:
        return PipelineConfig.load(config_path), []
    except Exception as e:  # FileNotFoundError, pydantic ValidationError 등
        return None, [f"config 로드 실패: {e}"]


def run_preflight(
    config_path: str,
    extra_core_modules: List[str],
    config_dependent_check: Callable[[object], List[str]],
    path_check: Optional[Callable[[object], List[str]]] = None,
):
    """각 STEP의 preflight.py가 이 함수 하나만 호출하면 된다. 실패 시 sys.exit(1)."""
    all_problems: List[str] = []

    print("[1/4] 코어 패키지 import 확인")
    core_missing = check_imports(CORE_MODULES + extra_core_modules)
    all_problems.extend(core_missing)
    print("      OK" if not core_missing else "")

    print("[2/4] 파이썬 환경 확인")
    check_python_env()

    print("[3/4] config 검증 + impl별 필요 패키지 확인")
    cfg, config_problems = check_config(config_path)
    all_problems.extend(config_problems)
    if cfg is not None:
        cfg_missing = config_dependent_check(cfg)
        all_problems.extend(cfg_missing)
        if not config_problems and not cfg_missing:
            print("      OK")

    print("[4/4] 경로/GPU 프리플라이트")
    if cfg is not None and path_check is not None:
        path_problems = path_check(cfg)
        all_problems.extend(path_problems)
        if not path_problems:
            print("      OK")
    else:
        print("      건너뜀 (config 로드 실패)" if cfg is None else "      (경로 검사 없음)")

    if all_problems:
        print("\n[FAIL] 프리플라이트 실패 — 무거운 모델 로드 안 함:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\n[PASS] 프리플라이트 통과 — 본 실행 진행합니다.\n")
    return cfg
