# ======================================================================
# [인수인계 헤더] 용도: Silero VAD로 원본 오디오에서 발화 구간만 추출해 이어붙임 (확정 파이프라인의 VAD 전처리 단계)
# 입력: dataset/audio/clip{1..4}.wav (16kHz mono)
# 출력: dataset/audio_vad/clip{1..4}.wav, vad_trim_log.csv, vad_timing_log.csv
# 실행 예시: (프로젝트 루트에서) .venv/bin/python scripts/apply_vad.py
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
VAD 전처리 스크립트.

dataset/audio/clip{n}.wav (원본, 16kHz mono)에서 faster-whisper에 내장된
Silero VAD(get_speech_timestamps)로 발화 구간만 검출해 이어붙인 뒤
dataset/audio_vad/clip{n}.wav 로 저장한다. 3개 STT 모델 모두 이 VAD 처리본을
공통 입력으로 사용해 "동일 audio 입력" 조건(계획서 14번)을 유지한다.

계획서 8번 원 전제("3분 연속 발화이므로 무음구간 적음")가 실제 데이터와 맞지
않음(직접 청취로 확인: 무음/작업소음 구간이 다수)이 확인되어, VAD 적용으로
전환한 근거는 STT_연구계획서_CLI.md 5-3절 참고.
"""

import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

PROJ = Path(__file__).resolve().parent.parent
AUDIO_IN = PROJ / "dataset" / "audio"
AUDIO_OUT = PROJ / "dataset" / "audio_vad"
LOG_PATH = AUDIO_OUT / "vad_trim_log.csv"
TIMING_LOG_PATH = AUDIO_OUT / "vad_timing_log.csv"

CLIPS = ["clip1", "clip2", "clip3", "clip4"]
SR = 16000

# speech_pad_ms: 발화 경계에 여유를 둬서 단어 앞뒤가 잘리지 않게 함.
VAD_OPTIONS = VadOptions(
    threshold=0.5,
    min_speech_duration_ms=200,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)


def read_wav_float(path: Path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR, f"{path} sample rate != {SR}"
        assert w.getsampwidth() == 2, f"{path} not 16-bit PCM"
        assert w.getnchannels() == 1, f"{path} not mono"
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def write_wav_int16(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    rows = ["clip,orig_sec,speech_sec,n_segments,ratio"]
    timing_rows = ["clip,vad_infer_sec"]

    for clip in CLIPS:
        in_path = AUDIO_IN / f"{clip}.wav"
        audio = read_wav_float(in_path)
        orig_sec = len(audio) / SR

        t0 = time.perf_counter()
        segments = get_speech_timestamps(audio, VAD_OPTIONS, sampling_rate=SR)
        vad_infer_sec = time.perf_counter() - t0
        timing_rows.append(f"{clip},{vad_infer_sec:.4f}")

        if not segments:
            print(f"[{clip}] 발화 구간 감지 실패 — 원본 그대로 사용")
            write_wav_int16(AUDIO_OUT / f"{clip}.wav", audio)
            rows.append(f"{clip},{orig_sec:.3f},{orig_sec:.3f},0,1.0000")
            continue

        chunks = [audio[seg["start"]:seg["end"]] for seg in segments]
        trimmed = np.concatenate(chunks)
        speech_sec = len(trimmed) / SR

        write_wav_int16(AUDIO_OUT / f"{clip}.wav", trimmed)
        ratio = speech_sec / orig_sec
        rows.append(f"{clip},{orig_sec:.3f},{speech_sec:.3f},{len(segments)},{ratio:.4f}")
        print(f"[{clip}] {orig_sec:.1f}s -> {speech_sec:.1f}s (발화 구간 {len(segments)}개, 비율 {ratio:.2%})")

    LOG_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    TIMING_LOG_PATH.write_text("\n".join(timing_rows) + "\n", encoding="utf-8")
    print("VAD 전처리 완료:", LOG_PATH, "/", TIMING_LOG_PATH)


if __name__ == "__main__":
    main()
