"""
config를 pydantic BaseModel로 받는 예시 (PREFLIGHT_CHECKLIST.md 3단계).

@dataclass에서 옮겨온 이유:
argparse로 사람이 매번 손으로 인자를 타이핑하면 오타(exit 2)가 난다.
config를 yaml 파일에 박아두고 pydantic으로 받으면, 필수값 누락/타입 오류/허용 안 된 값이
**config를 로드하는 시점에 즉시 예외**로 죽는다 — 무거운 모델 로드 전에.

사용:
    from config import Config
    config = Config.from_yaml("configs/step6.yaml")   # 값 이상하면 여기서 즉사
"""

from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class Config(BaseModel):
    # 필수값 (yaml에 없으면 로드 시점에 즉시 ValidationError)
    model_path: str
    input_file: str
    output_dir: str

    # 선택값 (기본값 있음)
    batch_size: int = 4
    track: Literal["manual", "auto"] = "manual"
    device: Literal["cuda", "cpu"] = "cuda"

    # pydantic v2는 정의되지 않은 필드가 yaml에 있으면 무시가 기본이라,
    # 오타 필드(예: batchsize)를 잡으려면 extra="forbid"로 막는다.
    model_config = {"extra": "forbid"}

    @field_validator("batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size는 1 이상이어야 함(입력값: {v})")
        return v

    @field_validator("track")
    @classmethod
    def track_known(cls, v: str) -> str:
        # Literal이 이미 막지만, 에러 메시지를 사람이 읽기 좋게 덮어씀.
        # track="a" 같은 오타가 여기서 명확한 메시지로 죽는다.
        if v not in ("manual", "auto"):
            raise ValueError(f'track은 "manual" 또는 "auto"만 허용(입력값: "{v}")')
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # data가 이상하면(필수값 누락, 타입 오류, 오타 필드) 여기서 즉시 죽는다.
        return cls(**data)