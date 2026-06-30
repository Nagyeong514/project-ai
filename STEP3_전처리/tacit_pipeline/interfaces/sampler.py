"""프레임 샘플링 인터페이스 — 전략 교체 가능(uniform/motion/hybrid)."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from ..schema.intermediate import FrameDetections, FrameMeta, Transcript
from .detector import FrameRef


@runtime_checkable
class FrameSampler(Protocol):
    """영상에서 'VLM에 넣을 의미있는 프레임'만 고른다.

    스펙 5.1: 순수 motion(픽셀 차분) 금지 — 스마트글래스 ego-motion 때문.
    motion 신호는 'YOLO가 잡은 hand/도구/부품 bbox의 움직임'으로 정의하고,
    샘플 기준 = (손/객체 움직임) ∪ (해당 구간 발화 존재) ∪ (hand-부품 근접/접촉).
    """

    def sample(
        self,
        meta: FrameMeta,
        coarse_detections: List[FrameDetections] | None = None,
        transcript: Transcript | None = None,
    ) -> List[FrameRef]:
        """샘플된 프레임 참조 리스트(초 단위 timestamp 포함) 반환.

        coarse_detections: ego-motion 무시용으로 미리 성긴 검출을 넘길 수 있음(옵션).
        transcript: '발화 존재 구간' 기준을 쓰기 위함(옵션).
        """
        ...
