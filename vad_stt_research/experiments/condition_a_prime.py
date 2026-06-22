"""
조건 A' — 배치, VAD 없음
역할: 배치 추론의 순수 효과 분리 (A↔A' 차이 = 배치 효과)
특징: 통일 디코딩 파라미터 사용, 외부 VAD 없음
"""
import time

import numpy as np
import soundfile as sf

from pipeline.stt.faster_whisper_runner import FasterWhisperRunner, STTResult


DECODING_PARAMS_UNIFIED = {
    "language": "ko",
    "beam_size": 5,
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "condition_on_previous_text": False,  # 통일값
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
}


def run_condition_a_prime(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    n_repeats: int = 3,
    warmup: int = 1,
) -> dict:
    """
    오디오 전체를 단일 배치로 투입 (VAD 없음, 통일 파라미터).
    """
    runner = FasterWhisperRunner(model_size, device, compute_type)
    audio, sr = sf.read(audio_path)
    audio_duration = len(audio) / sr

    for _ in range(warmup):
        runner.transcribe(audio_path, DECODING_PARAMS_UNIFIED)

    rtf_list = []
    last_result: STTResult | None = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = runner.transcribe(audio_path, DECODING_PARAMS_UNIFIED)
        elapsed = time.perf_counter() - t0
        rtf_list.append(elapsed / audio_duration)
        last_result = result

    return {
        "condition": "A_prime",
        "audio_path": str(audio_path),
        "audio_duration_s": audio_duration,
        "rtf_mean": float(np.mean(rtf_list)),
        "rtf_std": float(np.std(rtf_list)),
        "rtf_values": rtf_list,
        "segments": last_result.segments if last_result else [],
        "vad_time_s": None,
    }
