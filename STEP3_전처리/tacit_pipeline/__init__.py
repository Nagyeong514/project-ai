"""암묵지 후보 생성 전처리 파이프라인.

이 파이프라인은 후보 '생성'까지만 담당한다. 진짜 암묵지인지 판별/검증은 다음 단계.
"""

from .config import PipelineConfig
from .orchestrator import Pipeline

__all__ = ["PipelineConfig", "Pipeline"]
