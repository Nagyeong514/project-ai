import os
import tempfile
from typing import List, Tuple

import soundfile as sf
import numpy as np

from pipeline.vad.base import SpeechSegment


def extract_chunks(
    audio_path: str,
    segments: List[SpeechSegment],
    output_dir: str | None = None,
) -> List[Tuple[str, float]]:
    """
    VAD 세그먼트에 따라 오디오를 청크로 분리해 임시 파일로 저장.
    반환: [(chunk_file_path, original_start_time_s), ...]
    """
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vad_chunks_")
    os.makedirs(output_dir, exist_ok=True)

    chunk_info = []
    for idx, seg in enumerate(segments):
        start_sample = int(seg.start * sr)
        end_sample = min(int(seg.end * sr), len(audio))
        chunk_audio = audio[start_sample:end_sample]

        chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        sf.write(chunk_path, chunk_audio, sr)
        chunk_info.append((chunk_path, seg.start))

    return chunk_info


def compute_silence_ratio(audio_path: str, speech_segments: List[SpeechSegment]) -> float:
    """무음 비율 = 1 - (발화 총 길이 / 전체 오디오 길이)."""
    audio, sr = sf.read(audio_path)
    total_duration = len(audio) / sr
    speech_duration = sum(seg.duration for seg in speech_segments)
    return 1.0 - min(speech_duration / total_duration, 1.0)
