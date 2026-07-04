"""검출 없음(no-op) 디텍터.

용도: YOLO를 건너뛰고 싶을 때(예: VLM 단독 검증 런). ultralytics가 실행 시
CUDA_VISIBLE_DEVICES 를 덮어쓰는 부작용을 피하려고 GPU/모델을 일절 건드리지 않는다.
detect() 는 빈 리스트를 반환 → orchestrator 의 flat_dets=[] 경로(YOLO 실패 시와 동일)로 흐른다.
"""

from __future__ import annotations

from typing import Dict, List

from ..interfaces.detector import FrameRef
from ..schema.intermediate import FrameDetections, FrameMeta


class NoopDetector:
    """아무 것도 검출하지 않는 디텍터(테스트/우회용)."""

    def __init__(self, **_ignored):
        # config.params 가 뭐가 와도 무시(weights_path/device 등).
        pass

    @property
    def names(self) -> Dict[int, str]:
        return {}

    def detect(self, frames: List[FrameRef], meta: FrameMeta) -> List[FrameDetections]:
        return []
