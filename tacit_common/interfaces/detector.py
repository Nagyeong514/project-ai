"""YOLO 검출 인터페이스."""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from ..schema.intermediate import FrameDetections, FrameMeta


@runtime_checkable
class DetectorBackend(Protocol):
    """프레임 → 검출(위치/좌표). (예: 튜닝된 YOLO .pt + ultralytics)

    출력은 프레임별 {timestamp(초), class, conf, bbox}. 프레임번호↔초 변환은 FrameMeta(fps) 사용.
    """

    @property
    def names(self) -> Dict[int, str]:
        """모델의 idx→name 매핑(constants.validate_model_names 로 대조용)."""
        ...

    def detect(self, frames: List["FrameRef"], meta: FrameMeta) -> List[FrameDetections]:
        """선택된 프레임들에 대한 검출 결과 리스트."""
        ...


# 가벼운 프레임 참조 타입: 디코드된 이미지를 통째로 들고 다니면 무거우니
# (frame_idx, 선택적 이미지 핸들) 만 전달한다. 실제 이미지 로딩은 어댑터가 책임.
class FrameRef:
    """샘플러가 고른 프레임 1개의 참조. image는 numpy 배열 등(어댑터가 채움, 옵션)."""

    def __init__(self, frame_idx: int, timestamp: float, image=None):
        self.frame_idx = frame_idx
        self.timestamp = timestamp  # 초
        self.image = image  # Optional[np.ndarray] — 없으면 어댑터가 video에서 재디코드
