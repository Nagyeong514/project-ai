"""VLM 행동추출 인터페이스."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from typing import Optional

from ..schema.intermediate import ActionDescription, FrameDetections, FrameMeta
from .detector import FrameRef


@runtime_checkable
class VLMBackend(Protocol):
    """프레임/영상 + YOLO 검출(위치 힌트) → 구조화된 '관찰 로그'. (예: Qwen3-VL-8B, 4bit NF4)

    VLM은 '눈'이다: 보이는 사실만 기록(관찰), 해석·묶기·진단은 하지 않는다(후속 LLM 몫).
    출력 형식은 프롬프트로 강하게 고정한다(prompts/vlm_observation.py).

    입력 모드(결정됨): **본선 = observe_video(네이티브 비디오+fps)**. describe_actions(프레임)는 옵션.
    """

    def observe_video(
        self,
        video_path: str,
        meta: FrameMeta,
        injected_parts: Optional[list] = None,
    ) -> List[ActionDescription]:
        """[본선] 영상에서 fps로 프레임 추출 후 관찰 로그 반환(초 단위 timestamp)."""
        ...

    def observe_frames(
        self,
        frame_paths: List[str],
        times: List[float],
        injected_parts: Optional[list] = None,
    ) -> List[ActionDescription]:
        """[코어] 이미 추출된 프레임(YOLO와 공용)에 대한 관찰 로그."""
        ...

    def describe_actions(
        self,
        frames: List[FrameRef],
        detections_by_frame: List[FrameDetections],
    ) -> List[ActionDescription]:
        """[옵션] 미리 샘플된 프레임 리스트에 대한 관찰 로그."""
        ...

    def unload(self) -> None:
        """GPU 메모리 해제(순차 실행: VLM→언로드→LLM). 단일 GPU 환경 필수."""
        ...
