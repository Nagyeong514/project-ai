"""공용 중간 데이터 계약(스펙 3: 타임스탬프가 뼈대). 최종 출력 스키마(tacit_schema)는
이 패키지가 아니라 STEP5의 step5_schema/에 있다 — LLM 융합 결과물이라 STEP5 소유."""

from .intermediate import (
    AlignedWindow,
    BBox,
    Clip,
    Detection,
    FrameDetections,
    FrameMeta,
    ActionDescription,
    Transcript,
    Utterance,
    hhmmss_to_seconds,
    seconds_to_hhmmss,
)

__all__ = [
    "AlignedWindow",
    "ActionDescription",
    "Detection",
    "FrameDetections",
    "FrameMeta",
    "BBox",
    "Transcript",
    "Utterance",
    "Clip",
    "seconds_to_hhmmss",
    "hhmmss_to_seconds",
]
