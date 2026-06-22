"""
할루시네이션 감지 (조작적 정의 구현).

판정 기준 (둘 중 하나):
  1. 정답 무음 구간(≥ 3초) 내에 출력된 토큰
  2. 동일 n-gram(n≥4)이 3회 이상 연속 반복
"""
from typing import List, Tuple

from pipeline.stt.faster_whisper_runner import STTSegment
from pipeline.vad.base import SpeechSegment


def detect_hallucinations(
    pred_segments: List[STTSegment],
    silence_intervals: List[Tuple[float, float]],
    min_silence_s: float = 3.0,
    ngram_n: int = 4,
    ngram_repeat_count: int = 3,
) -> dict:
    """
    silence_intervals: 정답 기준 무음 구간 목록 [(start, end), ...]
    반환: {count_per_hour, events}  (오디오 총 길이는 caller가 전달)
    """
    events = []

    # 기준 1: 무음 구간 내 토큰
    long_silences = [(s, e) for s, e in silence_intervals if e - s >= min_silence_s]
    for seg in pred_segments:
        if not seg.text.strip():
            continue
        for silence_start, silence_end in long_silences:
            if seg.start >= silence_start and seg.end <= silence_end:
                events.append({
                    "type": "token_in_silence",
                    "segment_start": seg.start,
                    "segment_end": seg.end,
                    "text": seg.text,
                })

    # 기준 2: n-gram 반복
    all_words = []
    for seg in pred_segments:
        all_words.extend(seg.text.strip().split())

    ngram_events = _find_ngram_repeats(all_words, n=ngram_n, threshold=ngram_repeat_count)
    events.extend(ngram_events)

    return {"events": events, "total_count": len(events)}


def compute_hallucination_rate(events: list, audio_duration_s: float) -> float:
    """시간당 할루시네이션 발생 횟수."""
    hours = audio_duration_s / 3600.0
    return len(events) / hours if hours > 0 else 0.0


def get_silence_intervals_from_gt(
    gt_segments: List[SpeechSegment],
    audio_duration: float,
) -> List[Tuple[float, float]]:
    """정답 발화 구간으로부터 무음 구간 역산."""
    silences = []
    prev_end = 0.0
    for seg in sorted(gt_segments, key=lambda s: s.start):
        if seg.start > prev_end:
            silences.append((prev_end, seg.start))
        prev_end = max(prev_end, seg.end)
    if prev_end < audio_duration:
        silences.append((prev_end, audio_duration))
    return silences


def _find_ngram_repeats(words: List[str], n: int, threshold: int) -> list:
    events = []
    if len(words) < n * threshold:
        return events

    for i in range(len(words) - n * threshold + 1):
        ngram = tuple(words[i : i + n])
        count = 1
        j = i + n
        while j + n <= len(words) and tuple(words[j : j + n]) == ngram:
            count += 1
            j += n
        if count >= threshold:
            events.append({
                "type": "ngram_repeat",
                "ngram": " ".join(ngram),
                "repeat_count": count,
                "word_index": i,
            })
    return events
