"""오디오 청킹 헬퍼 — 짧은 발화 단위로 학습된 모델(Seamless-M4T-v2, Wav2Vec2-XLSR)이
장문 오디오를 처리할 수 있도록 고정 창 단위로 WAV를 분할한다."""
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf


def chunk_audio(
    wav_path: str,
    window_s: float = 25.0,
    tmp_dir: Optional[str] = None,
) -> List[Tuple[float, float, str]]:
    """
    16kHz mono WAV를 window_s초 단위로 잘라 임시 WAV 파일로 저장한다.

    Args:
        wav_path: 원본 WAV 경로.
        window_s: 청크 길이(초).
        tmp_dir: 청크 저장 디렉터리. None이면 새 임시 디렉터리를 만든다.

    Returns:
        (start_s, end_s, chunk_wav_path) 튜플 리스트. 호출자가 tmp_dir 정리를 책임진다.
    """
    data, sr = sf.read(wav_path, dtype="float32")
    window_samples = int(window_s * sr)

    out_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="stt_chunks_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Tuple[float, float, str]] = []
    for idx, start_sample in enumerate(range(0, len(data), window_samples)):
        end_sample = min(start_sample + window_samples, len(data))
        chunk_path = out_dir / f"chunk_{idx:04d}.wav"
        sf.write(chunk_path, data[start_sample:end_sample], sr)
        chunks.append((start_sample / sr, end_sample / sr, str(chunk_path)))

    return chunks
