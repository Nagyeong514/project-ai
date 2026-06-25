"""
VAD 엔진 팩토리. Silero(Phase 1 재사용) vs WebRTC(신규) 비교용.

Phase 1 코드(`vad_stt_research/`)를 재사용하므로 그 경로를 sys.path에 주입한다.
프로젝트 루트의 pipeline.py와 충돌하지 않도록 Phase1 경로를 맨 앞에 둔다.
"""
import os
import sys

_PHASE1 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vad_stt_research")
)
if _PHASE1 not in sys.path:
    sys.path.insert(0, _PHASE1)

from pipeline.vad import SileroVAD  # noqa: E402
from .webrtc_vad import WebRTCVAD  # noqa: E402

# 두 엔진 공통 후처리 파라미터 (공정 비교 — 엔진별 검출만 다르게)
_COMMON = dict(
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=400,
    merge_gap_ms=200,
    max_chunk_s=30.0,
)


def get_engine(name: str, **override):
    params = {**_COMMON, **override}
    if name == "silero":
        return SileroVAD(threshold=params.pop("threshold", 0.5), **params)
    if name == "webrtc":
        return WebRTCVAD(aggressiveness=params.pop("aggressiveness", 2), **params)
    raise ValueError(f"지원 엔진: silero | webrtc (요청: '{name}')")
