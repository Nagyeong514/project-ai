"""구체 구현(어댑터) 모음."""

from .aligner import WindowAligner
from .detector_noop import NoopDetector
from .detector_yolo import UltralyticsYOLODetector
from .llm_fusion import QwenLLMFusion
from .sampler_motion import MotionGuidedSampler, UniformSampler
from .stt_whisper import WhisperTurboSTT
from .transcript_refine import NormalizeRefiner
from .vlm_qwen import QwenVLActionExtractor

__all__ = [
    "WindowAligner",
    "NoopDetector",
    "UltralyticsYOLODetector",
    "QwenLLMFusion",
    "MotionGuidedSampler",
    "UniformSampler",
    "WhisperTurboSTT",
    "NormalizeRefiner",
    "QwenVLActionExtractor",
]
