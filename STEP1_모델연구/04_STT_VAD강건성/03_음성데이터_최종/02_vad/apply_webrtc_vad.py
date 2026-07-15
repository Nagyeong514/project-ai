# ======================================================================
# [인수인계 헤더] 용도: WebRTC VAD 전처리 — VAD 방식 비교 실험 조건 C의 입력 생성 (Silero와 동일 병합/패딩 규칙)
# 입력: dataset/audio/clip{1..4}.wav (16kHz mono)
# 출력: dataset/audio_webrtc/clip{1..4}.wav, webrtc_trim_log.csv, webrtc_timing_log.csv
# 실행 예시: (프로젝트 루트에서) .venv/bin/python scripts/apply_webrtc_vad.py
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
WebRTC VAD 전처리 스크립트 (VAD 방식 비교 실험 조건 C).

dataset/audio/clip{n}.wav(원본)에서 webrtcvad(aggressiveness=2, frame_duration_ms=30)로
프레임 단위 발화 여부를 판정하고, apply_vad.py(Silero, VadOptions: min_silence_duration_ms=500,
speech_pad_ms=300)와 동일한 병합/패딩 규칙을 적용해 dataset/audio_webrtc/clip{n}.wav로 저장한다.
min_speech_duration_ms=200도 Silero와 동일하게 맞춰 짧은 잡음 블립을 제거한다(공정 비교 목적,
사용자 지시에 명시되지는 않았으나 apply_vad.py의 VadOptions 기본값과 동일하게 통일함).
"""

import csv
import time
import wave
from pathlib import Path

import numpy as np
import webrtcvad

PROJ = Path(__file__).resolve().parent.parent
AUDIO_IN = PROJ / "dataset" / "audio"
AUDIO_OUT = PROJ / "dataset" / "audio_webrtc"
LOG_PATH = AUDIO_OUT / "webrtc_trim_log.csv"
TIMING_LOG_PATH = AUDIO_OUT / "webrtc_timing_log.csv"

CLIPS = ["clip1", "clip2", "clip3", "clip4"]
SR = 16000

AGGRESSIVENESS = 2
FRAME_MS = 30
FRAME_LEN = int(SR * FRAME_MS / 1000)  # 480 samples

# apply_vad.py(Silero, VadOptions)와 동일하게 맞춘 값
MIN_SILENCE_MS = 500
MIN_SPEECH_MS = 200
SPEECH_PAD_MS = 300


def read_wav_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR, f"{path} sample rate != {SR}"
        assert w.getsampwidth() == 2, f"{path} not 16-bit PCM"
        assert w.getnchannels() == 1, f"{path} not mono"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def write_wav_int16(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.astype(np.int16).tobytes())


def frame_speech_flags(vad: "webrtcvad.Vad", audio: np.ndarray) -> list:
    n_frames = len(audio) // FRAME_LEN
    flags = []
    for i in range(n_frames):
        frame = audio[i * FRAME_LEN:(i + 1) * FRAME_LEN]
        is_speech = vad.is_speech(frame.tobytes(), SR)
        flags.append(is_speech)
    return flags


def flags_to_raw_segments(flags: list) -> list:
    """연속된 True 프레임을 (start_frame, end_frame) 구간으로 묶는다."""
    segments = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(flags)))
    return segments


def merge_close_segments(segments: list, min_silence_frames: float) -> list:
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        gap = start - merged[-1][1]
        if gap < min_silence_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


def drop_short_segments(segments: list, min_speech_frames: float) -> list:
    return [(s, e) for s, e in segments if (e - s) >= min_speech_frames]


def pad_and_merge(segments: list, pad_frames: int, n_frames: int) -> list:
    padded = [(max(0, s - pad_frames), min(n_frames, e + pad_frames)) for s, e in segments]
    if not padded:
        return []
    merged = [list(padded[0])]
    for s, e in padded[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def main() -> None:
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    trim_rows = []
    timing_rows = []

    min_silence_frames = MIN_SILENCE_MS / FRAME_MS
    min_speech_frames = MIN_SPEECH_MS / FRAME_MS
    pad_frames = round(SPEECH_PAD_MS / FRAME_MS)

    for clip in CLIPS:
        vad = webrtcvad.Vad(AGGRESSIVENESS)
        in_path = AUDIO_IN / f"{clip}.wav"
        audio = read_wav_int16(in_path)
        orig_sec = len(audio) / SR

        t0 = time.perf_counter()
        flags = frame_speech_flags(vad, audio)
        raw_segments = flags_to_raw_segments(flags)
        merged = merge_close_segments(raw_segments, min_silence_frames)
        kept = drop_short_segments(merged, min_speech_frames)
        final_segments = pad_and_merge(kept, pad_frames, len(flags))
        vad_infer_sec = time.perf_counter() - t0

        if not final_segments:
            print(f"[{clip}] 발화 구간 감지 실패 — 원본 그대로 사용")
            write_wav_int16(AUDIO_OUT / f"{clip}.wav", audio)
            trim_rows.append({"clip_id": clip, "original_sec": f"{orig_sec:.3f}",
                               "trimmed_sec": f"{orig_sec:.3f}", "segment_count": 0})
            timing_rows.append({"clip": clip, "vad_infer_sec": f"{vad_infer_sec:.4f}"})
            continue

        chunks = [audio[s * FRAME_LEN:e * FRAME_LEN] for s, e in final_segments]
        trimmed = np.concatenate(chunks)
        trimmed_sec = len(trimmed) / SR

        write_wav_int16(AUDIO_OUT / f"{clip}.wav", trimmed)
        trim_rows.append({"clip_id": clip, "original_sec": f"{orig_sec:.3f}",
                           "trimmed_sec": f"{trimmed_sec:.3f}", "segment_count": len(final_segments)})
        timing_rows.append({"clip": clip, "vad_infer_sec": f"{vad_infer_sec:.4f}"})
        print(f"[{clip}] {orig_sec:.1f}s -> {trimmed_sec:.1f}s "
              f"(발화 구간 {len(final_segments)}개, VAD 처리시간 {vad_infer_sec:.3f}s)")

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "original_sec", "trimmed_sec", "segment_count"])
        w.writeheader()
        w.writerows(trim_rows)

    with open(TIMING_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "vad_infer_sec"])
        w.writeheader()
        w.writerows(timing_rows)

    print("WebRTC VAD 전처리 완료:", LOG_PATH, "/", TIMING_LOG_PATH)


if __name__ == "__main__":
    main()
