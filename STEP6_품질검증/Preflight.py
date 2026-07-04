"""
실행 전 방어 체크 (PREFLIGHT_CHECKLIST.md 1~4단계 실행).

핵심 규칙: 이 파일은 **표준 라이브러리만** import한다.
"패키지가 없는지 확인하는 코드"가 그 패키지(또는 서드파티)가 없어서 같이 죽으면
프리플라이트의 존재 이유가 사라진다. 그래서 importlib.util로 "설치 여부만" 검사한다.

무거운 모델 로드보다 반드시 먼저, main.py의 run() 첫 줄에서 호출한다.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# [1단계] import 체크
#
# ⚠️ 유지보수 규칙 (PREFLIGHT_CHECKLIST.md 1단계 참고):
#   새 서드파티 패키지를 코드에 추가하면 이 리스트도 반드시 같이 갱신한다.
#   가능하면 requirements.txt를 단일 소스로 삼아 파싱하는 쪽이 더 안전하다
#   (아래 _load_required_from_requirements 참고 — requirements.txt가 있으면 그걸 우선 사용).
#
# 여기 이름은 "import 이름" 기준(pip 이름과 다를 수 있음. 예: pip은 pyyaml, import는 yaml).
# ---------------------------------------------------------------------------
FALLBACK_REQUIRED_IMPORTS = [
    "torch",
    "transformers",
    "langgraph",   # 2026-07-03 사고의 실제 원인. 절대 빼지 말 것.
    "pydantic",
    "yaml",
]


def _load_required_from_requirements(req_path: str) -> list[str] | None:
    """requirements.txt가 있으면 거기서 패키지명을 끌어온다(단일 소스 우선).

    import 이름과 pip 이름이 다른 경우(pyyaml→yaml 등)를 위한 매핑만 최소로 둔다.
    파싱이 애매하면 None을 반환해 FALLBACK 리스트를 쓰게 한다.
    """
    if not os.path.isfile(req_path):
        return None

    pip_to_import = {
        "pyyaml": "yaml",
        "pillow": "PIL",
        "scikit-learn": "sklearn",
        "opencv-python": "cv2",
    }
    names: list[str] = []
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # "torch==2.3.0", "transformers>=4.40", "pkg[extra]" 등에서 이름만 추출
                pkg = line.split("==")[0].split(">=")[0].split("<=")[0]
                pkg = pkg.split("~=")[0].split("<")[0].split(">")[0].split("[")[0]
                pkg = pkg.strip().lower()
                if pkg:
                    names.append(pip_to_import.get(pkg, pkg))
    except OSError:
        return None
    return names or None


def check_imports(required: list[str] | None = None) -> None:
    """필수 패키지가 실제로 import 가능한지 확인. 없으면 즉시 죽는다."""
    if required is None:
        req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        required = _load_required_from_requirements(req_path) or FALLBACK_REQUIRED_IMPORTS

    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            "[preflight] 필수 패키지 누락: "
            + ", ".join(missing)
            + f"\n  현재 파이썬: {sys.executable}"
            + "\n  → venv가 맞는지, pip install 됐는지 확인. (모델 로드 전에 걸러냄)"
        )


# ---------------------------------------------------------------------------
# [2단계] venv/경로 체크
# ---------------------------------------------------------------------------
def check_python_env() -> None:
    """지금 어느 파이썬으로 떠 있는지 로그에 박제한다."""
    print(f"[preflight] python: {sys.executable}", flush=True)
    print(f"[preflight] version: {sys.version.split()[0]}", flush=True)
    # LD_LIBRARY_PATH 덮어쓰기 함정 감지: CUDA toolkit 경로가 사라졌으면 경고만.
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ld and "cuda" not in ld.lower():
        print(
            "[preflight] ⚠️ LD_LIBRARY_PATH에 cuda 경로가 안 보임. "
            "덮어쓰기로 libnvrtc.so.13이 사라졌을 수 있음(reference-cluster-specs 참고).",
            flush=True,
        )


# ---------------------------------------------------------------------------
# [4단계] 경로/환경변수 프리플라이트
# (3단계 인자/config 검증은 config.py의 pydantic이 import 시점에 자동 수행)
# ---------------------------------------------------------------------------
def check_paths(
    output_dir: str,
    input_files: list[str] | None = None,
    required_env: list[str] | None = None,
) -> None:
    """출력 폴더 쓰기 가능? 입력 파일 실존? 필수 환경변수 존재? 하나라도 아니면 죽는다."""
    # 출력 폴더: 만들고, 실제로 써지는지 확인
    os.makedirs(output_dir, exist_ok=True)
    probe = os.path.join(output_dir, ".preflight_write_test")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        raise RuntimeError(f"[preflight] 출력 폴더 쓰기 불가: {output_dir} ({e})") from e

    # 입력 파일 실존 확인
    for path in input_files or []:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[preflight] 입력 파일 없음: {path}")

    # 필수 환경변수 확인
    missing_env = [k for k in (required_env or []) if k not in os.environ]
    if missing_env:
        raise EnvironmentError(
            "[preflight] 환경변수 누락: " + ", ".join(missing_env)
        )


def check_gpu(require: bool = True) -> None:
    """GPU가 실제로 보이는지 확인. torch는 여기서만 import(1단계 통과가 전제)."""
    import torch  # noqa: PLC0415 - 1단계 통과 후에만 도달

    if require and not torch.cuda.is_available():
        raise RuntimeError(
            "[preflight] CUDA GPU가 안 보임. --gres=gpu 할당 여부, 드라이버 확인."
        )
    if torch.cuda.is_available():
        print(f"[preflight] GPU {torch.cuda.device_count()}개 인식됨", flush=True)


def run_preflight(
    output_dir: str,
    input_files: list[str] | None = None,
    required_env: list[str] | None = None,
    require_gpu: bool = True,
) -> None:
    """1~4단계를 순서대로 실행. main.py의 run() 첫 줄, 무거운 import보다 먼저 호출."""
    check_imports()          # 1단계
    check_python_env()       # 2단계
    check_paths(output_dir, input_files, required_env)  # 4단계 (3단계는 config import 시 자동)
    check_gpu(require=require_gpu)
    print("[preflight] 모든 체크 통과. 무거운 모델 로드로 진행.", flush=True)