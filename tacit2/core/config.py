"""config.yaml → pydantic 검증. 오타/누락은 여기서 5초 안에 죽는다(체크리스트 3단계)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class PathsCfg(BaseModel):
    videos: List[str]
    output_dir: str = "output"
    yolo_weights: str
    ffmpeg_bin: str = "ffmpeg"

    @field_validator("videos")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("paths.videos 가 비었습니다")
        return v


class SttCfg(BaseModel):
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ko"
    vad_filter: bool = True
    vad_threshold: float = 0.5

    @field_validator("compute_type")
    @classmethod
    def _no_bf16(cls, v):
        if "bf16" in v or "bfloat" in v:
            raise ValueError("Turing(sm75)은 bf16 불가 — float16 을 쓰세요")
        return v


class FramesCfg(BaseModel):
    fps: float = 0.5
    long_side: int = 480
    jpeg_q: int = 5


class YoloCfg(BaseModel):
    enabled: bool = True
    imgsz: int = 640
    conf: float = 0.25
    batch: int = 16


class VlmCfg(BaseModel):
    model_name: str
    device: str = "cuda:0"
    quantization: Optional[str] = "nf4"
    max_pixels: int = 36864
    chunk_sec: float = 30.0
    chunk_overlap_sec: float = 2.0
    max_frames_per_chunk: int = 15
    max_new_tokens: int = 1200
    repetition_penalty: float = 1.2
    retry_temperature: float = 0.3


class FusionCfg(BaseModel):
    model_name: str
    device: str = "cuda:0"
    quantization: Optional[str] = "nf4"
    max_new_tokens: int = 2048
    temperature: float = 0.2
    max_retries: int = 2
    window_sec: float = 4.0
    merge_cap_sec: float = 20.0


class Cfg(BaseModel):
    paths: PathsCfg
    stt: SttCfg = Field(default_factory=SttCfg)
    frames: FramesCfg = Field(default_factory=FramesCfg)
    yolo: YoloCfg
    vlm: VlmCfg
    fusion: FusionCfg


def load_cfg(path: str = "config.yaml") -> Cfg:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} 없음 — tacit2/config.yaml 을 복사·수정해서 쓰세요")
    with open(p, encoding="utf-8") as f:
        return Cfg.model_validate(yaml.safe_load(f))
