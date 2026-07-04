"""
config.yaml 로더 — **모델 교체의 단일 스위치**(원칙 1).

각 컴포넌트(stt/detector/vlm/llm/sampler/aligner)는 `impl`(어떤 구현을 쓸지)과
`params`(그 구현에 넘길 파라미터)를 가진다. 새 모델 추가 = registry에 클래스 등록 +
config의 `impl` 한 줄 교체. 기존 코드는 안 건드린다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


# project-ai/ 루트(tacit_common의 부모) — paths: 섹션의 상대경로를 여기 기준으로 해석한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ComponentConfig(BaseModel):
    """한 컴포넌트의 설정. impl=registry 키, params=구현 생성자 인자."""

    impl: str  # registry에 등록된 이름 (예: "whisper_turbo")
    params: Dict[str, Any] = Field(default_factory=dict)


class FrameExtractionConfig(BaseModel):
    """ffmpeg 프레임 추출 정책(STEP3 소관). registry로 갈아끼울 대안 구현이 없어서
    ComponentConfig가 아니라 고정 필드로 둔다 — 예전엔 vlm.params에 얹혀 있었는데
    (VLM이 프레임 추출을 직접 호출하던 시절의 흔적), STEP3/4 분리로 "프레임 추출은
    STEP3, 그걸 소비하는 VLM 추론은 STEP4"가 되면서 이리로 옮겼다."""

    fps_short: float = 0.5
    fps_long: float = 0.5
    long_video_threshold_sec: float = 120.0
    fps_override: float | None = None  # 지정하면 영상 길이와 무관하게 항상 이 fps
    long_side: int = 480
    ffmpeg_bin: str | None = None


class PathsConfig(BaseModel):
    """STEP 간 중간 산출물 경로(스펙: 결정지점1 — 파일로 주고받기). project-ai 루트
    기준 상대경로로 적어두고 PROJECT_ROOT를 붙여 절대경로로 쓴다 — STEP3/4/5
    단독 실행과 파이프라인_통합실행 양쪽 다 같은 config.yaml 하나를 보므로,
    어느 쪽에서 실행하든 동일한 절대 위치를 가리켜야 한다."""

    step3_frames_dir: str = "STEP3_전처리/output/_frames"
    step3_transcript_dir: str = "STEP3_전처리/transcripts"
    step4_detections_dir: str = "STEP4_YOLO_VLM관찰/output/detections"
    step4_observations_dir: str = "STEP4_YOLO_VLM관찰/output/vlm_observations"
    step5_aligned_windows_dir: str = "STEP5_LLM으로_암묵지_후보생성/output/aligned_windows"
    step5_tacit_json_dir: str = "STEP5_LLM으로_암묵지_후보생성/output/tacit_json"

    def resolve(self, rel: str) -> str:
        return str(PROJECT_ROOT / rel)


class PipelineConfig(BaseModel):
    """config.yaml 전체 매핑."""

    # 입출력
    video_path: str = ""  # TODO(fill): 내일 영상 경로
    output_dir: str = "output"
    transcript_dir: str = "transcripts"

    # 영상 메타 (검출이 못 읽을 때의 fallback / 강제값 — frame_extraction.fps_override와는
    # 별개다. 이건 FrameMeta.fps용 폴백, 저건 ffmpeg 추출 fps 정책)
    fps_override: float | None = None  # None이면 영상에서 읽음

    # 장비명. LLM이 추측하면 metadata.equipment가 None으로 남아 6단계 Gate A 검색
    # 쿼리(f"{equipment} {task}")에 "None"이 그대로 박히는 사고가 났었다(2026-07-04) —
    # 그래서 코드가 authoritative하게 채운다(orchestrator._finalize_metadata). 기종이
    # 하나뿐인 동안은 이 상수로 충분하고, 늘어나면 videos_map_path처럼 video_id별
    # 매핑으로 확장한다.
    equipment: str = ""

    # STEP 간 파일 경로 + 프레임 추출 정책
    paths: PathsConfig = Field(default_factory=PathsConfig)
    frame_extraction: FrameExtractionConfig = Field(default_factory=FrameExtractionConfig)

    # 컴포넌트들 (전부 교체 가능)
    sampler: ComponentConfig
    detector: ComponentConfig
    vlm: ComponentConfig
    stt: ComponentConfig
    transcript_refine: ComponentConfig
    aligner: ComponentConfig  # 타임스탬프 정렬(윈도우 묶기). 스펙 5.6
    llm: ComponentConfig

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"config 파일 없음: {path}\n  → config.example.yaml 을 복사해서 config.yaml 로 만드세요."
            )
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
