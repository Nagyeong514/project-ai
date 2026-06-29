"""발화 구간 검출 (WebRTC VAD + pad400).

앞 연구(vad_model_comparison)에서 WebRTC 채택, (vad_stt_research)에서 pad400 확정.
입력: 16k mono wav.  출력: [(start_sec, end_sec), ...] 발화 구간 리스트.
"""
from __future__ import annotations
import contextlib
import wave
import webrtcvad


def read_wave(path: str):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        assert wf.getnchannels() == 1, "mono wav 필요"
        assert wf.getsampwidth() == 2, "16-bit PCM 필요"
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sr, len(pcm) / 2 / sr  # pcm, sample_rate, duration_sec


def detect_segments(wav_path: str, cfg: dict) -> list[tuple[float, float]]:
    v = cfg["vad"]
    pcm, sr, dur = read_wave(wav_path)
    vad = webrtcvad.Vad(v["aggressiveness"])

    frame_ms = v["frame_ms"]
    bytes_per = int(sr * frame_ms / 1000) * 2
    flags = [
        vad.is_speech(pcm[i:i + bytes_per], sr)
        for i in range(0, len(pcm) - bytes_per, bytes_per)
    ]

    # 연속 speech 묶기 + 짧은 무음(merge_gap_ms 이하) 병합
    segs: list[list[float]] = []
    cur = None
    gap = 0
    for idx, fl in enumerate(flags):
        t = idx * frame_ms / 1000
        if fl:
            if cur is None:
                cur = [t, t + frame_ms / 1000]
            else:
                cur[1] = t + frame_ms / 1000
            gap = 0
        elif cur is not None:
            gap += frame_ms
            if gap > v["merge_gap_ms"]:
                segs.append(cur)
                cur = None
                gap = 0
            else:
                cur[1] = t + frame_ms / 1000
    if cur is not None:
        segs.append(cur)

    # 최소 길이 필터 + pad400 적용 (영상 경계로 클램프)
    pad = v["pad_ms"] / 1000
    out = []
    for a, b in segs:
        if b - a < v["min_segment_sec"]:
            continue
        out.append((max(0.0, a - pad), min(dur, b + pad)))

    # pad로 겹친 구간 병합
    merged: list[list[float]] = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def duration(wav_path: str) -> float:
    return read_wave(wav_path)[2]
