"""
조건 A — Vanilla 단독 (버퍼드 순차, Whisper 기본값)
역할: out-of-box 대조군
특징: condition_on_previous_text=True (Whisper 기본 → 할루시네이션 위험)
"""
import time
from pathlib import Path

from pipeline.stt.faster_whisper_runner import FasterWhisperRunner, STTResult


DECODING_PARAMS_VANILLA = {
    "language": "ko",
    "beam_size": 5,
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "condition_on_previous_text": True,   # Whisper 기본값
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
}


def run_condition_a(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    n_repeats: int = 3,
    warmup: int = 1,
) -> dict:
    """
    단일 오디오에 대해 조건 A 실행.
    RTF는 warmup 1회 제외 후 n_repeats 회 평균±SD.
    """
    runner = FasterWhisperRunner(model_size, device, compute_type)
    audio_duration = _get_audio_duration(audio_path)

    # warmup
    for _ in range(warmup):
        runner.transcribe(audio_path, DECODING_PARAMS_VANILLA)

    # 측정
    rtf_list = []
    last_result: STTResult | None = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = runner.transcribe(audio_path, DECODING_PARAMS_VANILLA)
        elapsed = time.perf_counter() - t0
        rtf_list.append(elapsed / audio_duration)
        last_result = result

    import numpy as np

    return {
        "condition": "A",
        "audio_path": str(audio_path),
        "audio_duration_s": audio_duration,
        "rtf_mean": float(np.mean(rtf_list)),
        "rtf_std": float(np.std(rtf_list)),
        "rtf_values": rtf_list,
        "segments": last_result.segments if last_result else [],
        "vad_time_s": None,
    }


def _get_audio_duration(audio_path: str) -> float:
    import soundfile as sf

    info = sf.info(audio_path)
    return info.duration
