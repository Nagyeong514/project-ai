"""
batch_addon — BatchedInferencePipeline 조건 (A'_batch, B_batch).

기존 비배치 실험(experiments/)은 완료분이라 수정하지 않는다. 이 모듈만 신규로,
기존 runner/vad/decoding 파라미터를 재사용해 '배치 여부'만 추가 변수로 둔다.

────────────────────────────────────────────────────────────────────
통제 요약 (faster-whisper 1.2.1 BatchedInferencePipeline 소스 기준)
────────────────────────────────────────────────────────────────────
1) 자르는 기준을 clip_timestamps로 직접 주입한다.
   - 배치 파이프라인은 30초 초과 오디오를 '분할 없이는' 처리 불가
     (clip_timestamps 없고 vad_filter=False면 RuntimeError; transcribe.py:416).
   - clip_timestamps를 주면 내장 VAD를 돌리지 않고 그 구간을 그대로 청크로 쓴다
     (transcribe.py:426~, "vad_filter will be ignored if clip_timestamps is used").
   - A'_batch: 고정 30초 연속 윈도우(전체 100% 커버, VAD 아님 = '무음 제거 없음'의 배치 대응물).
   - B_batch : 조건 B와 '동일한' 외부 Silero VAD 세그먼트를 주입 → VAD 경로 완전 동일.
   => 비배치 대응 조건과 차이가 오직 '배치'에서만 오도록 통제.

2) condition_on_previous_text:
   배치 내부에서 False로 하드코딩됨(transcribe.py:547). 인자로 줘도 무시된다.
   비배치 A'/B도 이미 False(DECODING_PARAMS_UNIFIED) → 전 조건 동일하게 유지됨.
   따라서 이 파라미터는 A'↔A'_batch, B↔B_batch 비교에 혼입되지 않는다.

3) without_timestamps:
   배치 디폴트 True, 비배치 transcribe 디폴트 False. 이는 배치 추론의 본질적 차이라
   제거 불가(배치는 청크 단위 타임스탬프만 부여). WER/CER/할루시/RTF에는 영향이 미미하나
   세그먼트 단위 timestamp_drift는 의미가 약해진다 → 이 애드온은 4개 지표만 기록.

4) batch_size: 8GB VRAM에서 OOM 가능 → 기본 8, OOM 시 4로 자동 폴백.
   사용된 batch_size와 폴백 여부를 결과에 기록.

5) 30초 초과 clip 방어: 배치는 clip이 30초를 넘으면 앞 30초만 전사(transcribe.py:438).
   우리 청크는 VAD max_chunk_s=30·고정창=30이라 정상적으로는 안 넘지만, 안전하게 분할한다.
"""
import time

import numpy as np
import soundfile as sf
from faster_whisper import BatchedInferencePipeline

from pipeline.stt.faster_whisper_runner import STTSegment, STTResult, FasterWhisperRunner
from pipeline.vad import get_vad
from experiments.condition_a_prime import DECODING_PARAMS_UNIFIED

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


WINDOW_S = 30.0  # Whisper 윈도우 상한 = 고정창 길이 = clip 최대 길이

# 비배치 통일 디코딩값에서 배치 transcribe가 받는 키만 추린다.
# 제외:
#   language                    → transcribe(language=...) 인자로 직접 전달
#   condition_on_previous_text  → 배치 내부 False 하드코딩(받아도 무시), 비배치도 False라 동일
_BATCH_EXCLUDE = {"language", "condition_on_previous_text"}
_BATCH_DECODE = {k: v for k, v in DECODING_PARAMS_UNIFIED.items() if k not in _BATCH_EXCLUDE}


def _split_over_30(clips: list[dict]) -> list[dict]:
    """30초 초과 clip을 30초 단위로 분할(방어용). 정상 입력(≤30s)이면 그대로 통과."""
    out = []
    for c in clips:
        start, end = float(c["start"]), float(c["end"])
        while end - start > WINDOW_S:
            out.append({"start": start, "end": start + WINDOW_S})
            start += WINDOW_S
        if end > start:
            out.append({"start": start, "end": end})
    return out


def _fixed_windows(duration_s: float, window_s: float = WINDOW_S) -> list[dict]:
    """전체 오디오를 [0, duration]까지 window_s 연속 구간으로 분할. VAD 아님(기계적 분할)."""
    clips = []
    t = 0.0
    while t < duration_s:
        end = min(t + window_s, duration_s)
        clips.append({"start": t, "end": end})
        t = end
    return clips


def _vad_clips(audio_path: str, vad_params: dict) -> tuple[list[dict], float, int]:
    """조건 B와 동일한 외부 Silero VAD 세그먼트를 clip_timestamps(초) 형식으로 반환."""
    vad_params = dict(vad_params)  # caller 원본 보호
    engine = vad_params.pop("engine", "silero")
    vad = get_vad(engine, **vad_params)

    t0 = time.perf_counter()
    segments = vad.detect(audio_path)  # 조건 B와 동일 호출 → 동일 청크 경계
    vad_time = time.perf_counter() - t0

    clips = [{"start": float(s.start), "end": float(s.end)} for s in segments]
    clips = _split_over_30(clips)
    return clips, vad_time, len(segments)


def _run_batched_once(
    pipeline: BatchedInferencePipeline,
    audio_path: str,
    clip_timestamps: list[dict],
    batch_size: int,
    language: str = "ko",
) -> tuple[STTResult, float]:
    """배치 1회 전사 → (STTResult, elapsed_s). 제너레이터를 모두 소비한 시간을 잰다."""
    t0 = time.perf_counter()
    seg_gen, info = pipeline.transcribe(
        audio_path,
        language=language,
        clip_timestamps=clip_timestamps,  # 내장 VAD 우회 + 자르는 기준 통제
        vad_filter=False,                 # clip_timestamps 있으면 무시되나 의도 명시
        without_timestamps=True,          # 배치 네이티브(디폴트). 비배치와의 본질적 차이(상단 주석 3)
        batch_size=batch_size,
        **_BATCH_DECODE,
    )
    segments = [
        STTSegment(
            start=s.start,
            end=s.end,
            text=s.text,
            avg_logprob=getattr(s, "avg_logprob", 0.0) or 0.0,
            no_speech_prob=getattr(s, "no_speech_prob", 0.0) or 0.0,
            compression_ratio=getattr(s, "compression_ratio", 0.0) or 0.0,
        )
        for s in seg_gen
    ]
    elapsed = time.perf_counter() - t0
    return STTResult(segments=segments, language=info.language, processing_time_s=elapsed), elapsed


def _measure_batched(
    pipeline: BatchedInferencePipeline,
    audio_path: str,
    clip_timestamps: list[dict],
    audio_duration: float,
    batch_size: int,
    batch_size_fallback: int,
    n_repeats: int,
    warmup: int,
    language: str = "ko",
) -> tuple[STTResult, list[float], int, bool]:
    """
    warmup + n_repeats 측정. OOM이면 batch_size를 폴백값으로 낮춰 처음부터 재측정.
    반환: (마지막 STTResult, rtf_list, 사용된 batch_size, oom_fallback 여부)
    """
    bs = batch_size
    oom_fallback = False
    while True:
        try:
            for _ in range(warmup):
                _run_batched_once(pipeline, audio_path, clip_timestamps, bs, language)

            rtf_list = []
            last: STTResult | None = None
            for _ in range(n_repeats):
                res, elapsed = _run_batched_once(pipeline, audio_path, clip_timestamps, bs, language)
                rtf_list.append(elapsed / audio_duration)
                last = res
            return last, rtf_list, bs, oom_fallback

        except RuntimeError as e:
            # ctranslate2/torch CUDA OOM은 RuntimeError로 올라온다.
            if "out of memory" in str(e).lower() and bs > batch_size_fallback:
                if _HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_fallback = True
                bs = batch_size_fallback
                continue
            raise


def run_condition_a_prime_batch(
    audio_path: str,
    runner: FasterWhisperRunner,
    batch_size: int = 8,
    batch_size_fallback: int = 4,
    n_repeats: int = 3,
    warmup: int = 1,
) -> dict:
    """
    A'_batch — 고정 30초 윈도우(VAD 없음) + BatchedInferencePipeline.
    A' ↔ A'_batch 차이 = 배치 효과(자르는 기준이 양쪽 모두 '전체 커버, VAD 무관'이라 통제됨).
    runner의 로드된 모델을 재사용(중복 로드 방지).
    """
    pipeline = BatchedInferencePipeline(model=runner._model)
    audio, sr = sf.read(audio_path)
    audio_duration = len(audio) / sr

    clip_timestamps = _fixed_windows(audio_duration)  # 기계적 고정창 = '무음 제거 없음'
    last, rtf_list, bs, oom = _measure_batched(
        pipeline, audio_path, clip_timestamps, audio_duration,
        batch_size, batch_size_fallback, n_repeats, warmup,
        language=DECODING_PARAMS_UNIFIED["language"],
    )
    return {
        "condition": "A_prime_batch",
        "audio_path": str(audio_path),
        "audio_duration_s": audio_duration,
        "rtf_mean": float(np.mean(rtf_list)),
        "rtf_std": float(np.std(rtf_list)),
        "rtf_values": rtf_list,
        "segments": last.segments if last else [],
        "vad_time_s": None,           # VAD 미사용
        "n_chunks": len(clip_timestamps),
        "batched": True,
        "batch_size": bs,
        "oom_fallback": oom,
    }


def run_condition_b_batch(
    audio_path: str,
    runner: FasterWhisperRunner,
    vad_params: dict | None = None,
    batch_size: int = 8,
    batch_size_fallback: int = 4,
    n_repeats: int = 3,
    warmup: int = 1,
) -> dict:
    """
    B_batch — 조건 B와 동일한 Silero VAD 청크를 clip_timestamps로 주입 + 배치.
    B ↔ B_batch 차이 = 배치 효과(VAD 경로가 완전 동일).
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

    pipeline = BatchedInferencePipeline(model=runner._model)
    audio, sr = sf.read(audio_path)
    audio_duration = len(audio) / sr

    # VAD 시간 별도 계측(조건 B와 동일 의미). RTF 측정 구간 밖.
    clip_timestamps, vad_time, n_seg = _vad_clips(audio_path, vad_params)
    last, rtf_list, bs, oom = _measure_batched(
        pipeline, audio_path, clip_timestamps, audio_duration,
        batch_size, batch_size_fallback, n_repeats, warmup,
        language=DECODING_PARAMS_UNIFIED["language"],
    )
    return {
        "condition": "B_batch",
        "audio_path": str(audio_path),
        "audio_duration_s": audio_duration,
        "rtf_mean": float(np.mean(rtf_list)),
        "rtf_std": float(np.std(rtf_list)),
        "rtf_values": rtf_list,
        "segments": last.segments if last else [],
        "vad_time_s": vad_time,
        "n_chunks": n_seg,
        "batched": True,
        "batch_size": bs,
        "oom_fallback": oom,
    }
