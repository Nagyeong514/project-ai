"""
단계 간 **데이터 계약(중간 산출물 타입)**.

원칙 3: 타임스탬프가 뼈대다. 모든 중간 산출물은 **공통 시간 단위(초, float)** 로 정규화해서 들고 다닌다.
(YOLO는 프레임 번호, STT는 초 단위로 나오므로 변환에 FPS가 필요 → FrameMeta 참조)

이 타입들은 최종 출력(tacit_schema)과 별개다. 최종 JSON 키와 섞지 마라.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


def seconds_to_hhmmss(t: Optional[float]) -> Optional[str]:
    """공통 초 단위 → 최종 스키마용 'HH:MM:SS' 문자열. None은 그대로 통과."""
    if t is None:
        return None
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def hhmmss_to_seconds(label: Optional[str]) -> Optional[float]:
    """'HH:MM:SS' 또는 'MM:SS' → 초(float). 역변환. 파싱 실패/None이면 None.

    네이티브 비디오 모드에서 VLM이 뱉은 timestamp 문자열을 공통 초 단위로 되돌릴 때 쓴다.
    """
    if not label:
        return None
    try:
        parts = [float(p) for p in str(label).strip().split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0.0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, s = 0.0, 0.0, parts[0]
    else:
        return None
    return h * 3600 + m * 60 + s


class FrameMeta(BaseModel):
    """영상 메타데이터 — 프레임번호↔초 변환의 단일 기준."""

    video_id: str
    path: str
    fps: float
    width: int = 0
    height: int = 0
    n_frames: int = 0

    def frame_to_sec(self, frame_idx: int) -> float:
        return frame_idx / self.fps if self.fps else 0.0

    def sec_to_frame(self, t: float) -> int:
        return int(round(t * self.fps))


class BBox(BaseModel):
    """바운딩박스. 포맷은 config로 바뀔 수 있음(기본: 픽셀 xywh, 좌상단 기준).

    TODO(decision): 정규화(0~1) 좌표를 쓸지 픽셀을 쓸지 — config.detector.bbox_format 로 노출.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class Detection(BaseModel):
    """YOLO 검출 1건. 공통 초 단위 timestamp 필수."""

    timestamp: float  # 초(float) — frame_to_sec 로 변환된 값
    frame_idx: int
    cls: str  # constants.Classes 의 값과 대조 검증됨
    conf: float
    bbox: BBox


class FrameDetections(BaseModel):
    """한 프레임의 검출 묶음."""

    timestamp: float
    frame_idx: int
    detections: List[Detection] = Field(default_factory=list)


class Utterance(BaseModel):
    """STT segment 1개 = 발화 1건. 원문(raw_text)은 끝까지 보존한다(스키마 source_utterance용).

    설계결정 (b): 정제 stage는 결정적 작업만(정규화 + 반복감지). **근거성 판단은 여기서 안 한다
    → 융합 LLM이 raw 발화로 직접** (한국어 구어체 정규식 근거판단은 실측상 0건이라 폐기).
    """

    start: float  # 초
    end: float  # 초
    raw_text: str  # STT 원문 그대로(절대 손실 금지)
    normalized_text: str  # 영어/기술용어 정규화 적용본
    # 연속 동일 발화(Whisper 끝부분 환각 의심). 삭제 안 하고 플래그만 → 융합서 무시.
    repeat_hallucination: bool = False


class Transcript(BaseModel):
    """STT 전체 결과(파일로 저장됨 → 스키마 transcript_ref가 가리킴)."""

    video_id: str
    language: Optional[str] = None
    model: Optional[str] = None
    utterances: List[Utterance] = Field(default_factory=list)


class ActionDescription(BaseModel):
    """VLM '관찰 로그(observation)' 1건 (스펙 5.3 버전 A).

    VLM은 '눈'이다 — 관찰 가능한 사실만 기록한다(묶기/해석/암묵지 판단 금지, 그건 후속 LLM).
    timestamp는 VLM이 계산하지 않는다. 프레임 샘플링 단계가 부여한 시각을 그대로 부착한다(5.6).
    """

    timestamp: float  # 초 — 프레임 샘플링이 부여한 시각(VLM이 재계산 안 함)
    actor: Optional[str] = None  # 행동 주체("오른손"/"왼손" 등). 손가락 단위 분리는 action에 서술.
    action: str  # 관측된 구체 동작(추상동사 금지 — '확인/점검' X, 눈에 보이는 동작 O)
    objects: List[str] = Field(default_factory=list)  # objects_visible — 보이는 객체(YOLO 클래스명)
    raw: Optional[str] = None  # VLM 원출력(디버깅용)


class AlignedWindow(BaseModel):
    """타임스탬프 정렬 산출물 = LLM 융합의 입력 단위.

    발화와 행동은 동시에 안 일어난다(말 먼저/행동 먼저/무언). 1:1 매칭하지 않고
    ±N초 윈도우로 '구간'을 묶어 LLM에 넘긴다.
    """

    window_start: float  # 초
    window_end: float  # 초
    actions: List[ActionDescription] = Field(default_factory=list)
    utterances: List[Utterance] = Field(default_factory=list)
    detections: List[Detection] = Field(default_factory=list)  # 위치 힌트(코드레벨 판단용)

    @property
    def case(self) -> str:
        """LLM 융합 3케이스 분류(스펙 5.7). 'fusion' | 'action_only' | 'utterance_only' | 'empty'."""
        has_a = bool(self.actions)
        has_u = bool(self.utterances)
        if has_a and has_u:
            return "fusion"
        if has_a:
            return "action_only"
        if has_u:
            return "utterance_only"
        return "empty"


class Clip(BaseModel):
    """긴 영상을 자른 '작업 단위 클립' 메타(스키마 clip_start/clip_end 용)."""

    clip_id: str
    start: float  # 초
    end: float  # 초
    windows: List[AlignedWindow] = Field(default_factory=list)
