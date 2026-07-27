"""preflight: 무거운 것(모델 로드) 위로는 전부 5초 안에 죽을 수 있는 검증만.

표준 라이브러리 + yaml/pydantic(config 로드에 필수)만 사용 —
"패키지 없는지 확인하는 코드가 그 패키지 없어서 죽는" 모순 방지.

검사 순서(체크리스트 그대로):
  1. import 체크 — 스테이지별로 실제 쓰는 패키지만(config-aware)
  2. 환경 체크 — sys.executable, LD_LIBRARY_PATH 로그
  3. config 검증 — pydantic 이 load_cfg 시점에 자동 수행
  4. 경로/GPU 체크 — 영상, weights, ffmpeg, cuda
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

# 스테이지 → 필요한 패키지 (config-aware: 실행할 스테이지 것만 검사)
STAGE_IMPORTS = {
    "stt": ["faster_whisper"],
    "frames": [],  # ffmpeg 바이너리만 필요
    "yolo": ["ultralytics"],
    "vlm": ["torch", "transformers", "qwen_vl_utils", "bitsandbytes", "PIL"],
    "fusion": ["torch", "transformers", "bitsandbytes"],
}


def _fail(msg: str):
    print(f"[PREFLIGHT FAIL] {msg}")
    sys.exit(1)


def run_preflight(cfg, stages: list):
    print(f"[preflight] python: {sys.executable}")
    print(f"[preflight] LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(unset)')[:200]}")

    # 1. import 체크 (선택된 스테이지 것만)
    needed = sorted({m for s in stages for m in STAGE_IMPORTS.get(s, [])})
    for mod in needed:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            _fail(f"필수 패키지 import 실패: {mod} ({e})\n"
                  f"  → pip install 후 재시도. GPU 노드/로그인 노드 패키지 불일치 주의")
    print(f"[preflight] imports ok: {needed or '(none)'}")

    # 4. 경로 체크
    for v in cfg.paths.videos:
        if not Path(v).exists():
            _fail(f"영상 없음: {v}\n  → config.yaml 의 paths.videos 를 실제 경로로 채우세요")
    if "frames" in stages and shutil.which(cfg.paths.ffmpeg_bin) is None \
            and not Path(cfg.paths.ffmpeg_bin).exists():
        _fail(f"ffmpeg 없음: {cfg.paths.ffmpeg_bin}")
    if "yolo" in stages and cfg.yolo.enabled and not Path(cfg.paths.yolo_weights).exists():
        _fail(f"YOLO weights 없음: {cfg.paths.yolo_weights}")

    # GPU 체크 (GPU 필요한 스테이지가 선택됐을 때만 torch 로드)
    if any(s in stages for s in ("stt", "vlm", "fusion")):
        import torch  # 위 import 체크 통과 후라 안전

        if not torch.cuda.is_available():
            _fail("CUDA 사용 불가 — GPU 노드에서 실행 중인지, srun --gres=gpu 확인")
        n = torch.cuda.device_count()
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"[preflight] GPU{i}: {p.name} {p.total_memory / 1024**3:.1f}GiB")

    print("[preflight] OK — 무거운 로드로 진행")
