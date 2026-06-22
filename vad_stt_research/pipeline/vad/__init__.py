from __future__ import annotations

import importlib

from .base import BaseVAD, SpeechSegment

__all__ = ["get_vad", "BaseVAD", "SpeechSegment"]

# 엔진 이름 → (모듈 파일명, 클래스명) 매핑
# 엔진별 의존성(torch, webrtcvad 등)은 실제 호출 시점에만 로드됨
_REGISTRY: dict[str, tuple[str, str]] = {
    "silero":   ("silero_vad",   "SileroVAD"),
    "pyannote": ("pyannote_vad", "PyannoteVAD"),
    "webrtc":   ("webrtc_vad",   "WebRTCVAD"),
    "librosa":  ("librosa_vad",  "LibrosaVAD"),
}


def get_vad(engine: str, **kwargs) -> BaseVAD:
    if engine not in _REGISTRY:
        raise ValueError(
            f"Unknown VAD engine: '{engine}'. Available: {sorted(_REGISTRY)}"
        )
    module_name, class_name = _REGISTRY[engine]
    mod = importlib.import_module(f".{module_name}", package=__package__)
    cls: type[BaseVAD] = getattr(mod, class_name)
    return cls(**kwargs)
