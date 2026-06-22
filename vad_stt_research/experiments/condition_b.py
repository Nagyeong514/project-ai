"""
조건 B — VAD + 배치 (제안 파이프라인)
역할: 전체 파이프라인 효과 측정
흐름: 오디오 → Silero VAD → 청크 추출 → 배치 STT → 타임스탬프 재매핑

A' ↔ B 차이 = VAD(무음 제거)의 순수 효과
A  ↔ B 차이 = 전체 파이프라인 효과
"""
import tempfile
import time

import numpy as np
import soundfile as sf

from pipeline.vad import get_vad
from pipeline.merge.chunk_extractor import extract_chunks, compute_silence_ratio
from pipeline.stt.faster_whisper_runner import FasterWhisperRunner, STTResult
from experiments.condition_a_prime import DECODING_PARAMS_UNIFIED


def run_condition_b(
    audio_path: str,
    vad_params: dict | None = None,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    n_repeats: int = 3,
    warmup: int = 1,
) -> dict:
    """
    VAD 전처리 후 배치 STT 실행.
    vad_params: configs/experiment_config.yaml의 vad 섹션.
    """
    if vad_params is None:
        vad_params = {
            "engine": "silero",
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
            "merge_gap_ms": 200,
            "max_chunk_s": 30.0,
        }

    engine = vad_params.pop("engine", "silero")
    vad = get_vad(engine, **vad_params)
    runner = FasterWhisperRunner(model_size, device, compute_type)

    audio, sr = sf.read(audio_path)
    audio_duration = len(audio) / sr

    # VAD 시간 별도 계측 (손익분기점 분석용)
    t_vad_start = time.perf_counter()
    segments = vad.detect(audio_path)
    vad_time = time.perf_counter() - t_vad_start

    silence_ratio = compute_silence_ratio(audio_path, segments)

    with tempfile.TemporaryDirectory(prefix="vad_chunks_") as chunk_dir:
        chunk_info = extract_chunks(audio_path, segments, output_dir=chunk_dir)

        # warmup
        for _ in range(warmup):
            _run_batch(runner, chunk_info)

        rtf_list = []
        last_result: STTResult | None = None
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            result = _run_batch(runner, chunk_info)
            elapsed = time.perf_counter() - t0
            rtf_list.append(elapsed / audio_duration)
            last_result = result

    return {
        "condition": "B",
        "audio_path": str(audio_path),
        "audio_duration_s": audio_duration,
        "silence_ratio": silence_ratio,
        "n_chunks": len(segments),
        "rtf_mean": float(np.mean(rtf_list)),
        "rtf_std": float(np.std(rtf_list)),
        "rtf_values": rtf_list,
        "vad_time_s": vad_time,
        "segments": last_result.segments if last_result else [],
    }


def _run_batch(runner: FasterWhisperRunner, chunk_info: list) -> STTResult:
    all_segments = []
    language = ""
    total_time = 0.0
    for chunk_path, offset in chunk_info:
        result = runner.transcribe(chunk_path, DECODING_PARAMS_UNIFIED, chunk_offset=offset)
        all_segments.extend(result.segments)
        language = language or result.language
        total_time += result.processing_time_s
    from pipeline.stt.faster_whisper_runner import STTResult
    return STTResult(segments=all_segments, language=language, processing_time_s=total_time)
