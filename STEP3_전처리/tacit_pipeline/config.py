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


class ComponentConfig(BaseModel):
    """한 컴포넌트의 설정. impl=registry 키, params=구현 생성자 인자."""

    impl: str  # registry에 등록된 이름 (예: "whisper_turbo")
    params: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """config.yaml 전체 매핑."""

    # 입출력
    video_path: str = ""  # TODO(fill): 내일 영상 경로
    output_dir: str = "output"
    transcript_dir: str = "transcripts"

    # 영상 메타 (검출이 못 읽을 때의 fallback / 강제값)
    fps_override: float | None = None  # None이면 영상에서 읽음

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
