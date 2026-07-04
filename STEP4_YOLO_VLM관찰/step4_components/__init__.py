"""STEP4(YOLO 검출 + VLM 관찰) 구체 구현 모음.

sampler_motion.py(UniformSampler/MotionGuidedSampler)는 registry에 등록만 되고
orchestrator/러너 어디서도 .sample()을 호출하지 않는 완전 미사용 코드다(config 값과
무관하게 항상 그렇다) — 삭제하지 않고 표시만 해둠(2026-07-04 분할 계획서 결정).
detector_noop.py는 반대로 "현재 config에서는 비활성이지만 언제든 켤 수 있는 대안"이다
(VRAM 부족 시 YOLO를 건너뛰던 실제 사용 이력 있음, [[project-server-migration-step3]])."""

from .detector_noop import NoopDetector
from .detector_yolo import UltralyticsYOLODetector
from .sampler_motion import MotionGuidedSampler, UniformSampler  # 미사용 — 위 설명 참고
from .vlm_qwen import QwenVLActionExtractor

__all__ = [
    "NoopDetector",
    "UltralyticsYOLODetector",
    "MotionGuidedSampler",
    "UniformSampler",
    "QwenVLActionExtractor",
]
