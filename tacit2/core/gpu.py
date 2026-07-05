"""GPU 위생. unload 누락(배치 전멸 사건)의 재발을 '패턴'으로 막는다.

규칙: 모든 스테이지는 자기 모델을 release() 로 내리고 끝난다.
스테이지-우선 실행 구조라 모델 두 개가 GPU 에 공존하는 순간 자체가 없지만,
그래도 각 스테이지 끝에서 반드시 호출해 다음 스테이지에 깨끗한 GPU 를 넘긴다.
"""
from __future__ import annotations

import gc
import os


def set_alloc_conf():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def release(*objs):
    """모델/프로세서 객체들을 del + gc + empty_cache. torch 미로드 상태여도 안전."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def log_mem(tag: str):
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"[GPU{i}] {tag}: allocated={alloc:.2f}GiB reserved={reserv:.2f}GiB")
    except Exception:
        pass
