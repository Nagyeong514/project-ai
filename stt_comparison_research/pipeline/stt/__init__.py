"""STT 엔진 팩토리."""
from typing import Any, Dict

from pipeline.stt.base import BaseSTT, STTResult, STTSegment


def get_stt(engine_cfg: Dict[str, Any]) -> BaseSTT:
    """
    experiment_config.yaml 의 models.{name} 블록을 받아 대응 runner 반환.
    engine: faster_whisper | kospeech | clova_api | kakao_api
    """
    engine = engine_cfg["engine"]

    if engine == "faster_whisper":
        from pipeline.stt.faster_whisper_runner import FasterWhisperRunner
        from pipeline.stt.faster_whisper_runner import FasterWhisperRunner

        return FasterWhisperRunner(model_id=engine_cfg["model_id"])

    if engine == "kospeech":
        from pipeline.stt.kospeech_runner import KospeechRunner

        ckpt = engine_cfg.get("checkpoint")
        if not ckpt:
            raise ValueError("kospeech engine requires 'checkpoint' path in config.")
        return KospeechRunner(checkpoint_path=ckpt)

    if engine == "clova_api":
        from pipeline.stt.api_runner import ClovaSTTRunner

        return ClovaSTTRunner()

    if engine == "kakao_api":
        from pipeline.stt.api_runner import KakaoSTTRunner

        return KakaoSTTRunner()

    raise ValueError(f"Unknown STT engine: {engine}")


__all__ = ["get_stt", "BaseSTT", "STTResult", "STTSegment"]
