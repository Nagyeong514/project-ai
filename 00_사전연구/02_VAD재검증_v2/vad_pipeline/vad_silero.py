"""
Silero VAD 전처리 — 무음/비발화 구간을 잘라내고 발화 구간만 이어붙인 WAV를 저장한다.

silero-vad 패키지의 read_audio/save_audio는 내부적으로 torchaudio.load/save를 쓰는데,
이 venv의 torchaudio(2.11.0+cu130)는 torchcodec 없이는 오디오 I/O가 안 됨(실측 확인,
"TorchCodec is required" 에러). torchcodec을 추가로 설치하는 대신, 프로젝트 전체에서
이미 쓰고 있는 soundfile로 오디오 로드/저장을 직접 하고, VAD 판정(get_speech_timestamps/
collect_chunks)만 silero_vad 패키지 함수를 쓰는 방식으로 우회했다.
"""
import soundfile as sf
import torch
from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


def trim_with_vad(wav_path: str, out_path: str, **vad_kwargs) -> dict:
    """
    Silero VAD로 발화 구간만 추출해 out_path(16kHz mono WAV)에 저장한다.

    Args:
        wav_path: 원본 WAV(16kHz mono) 경로.
        out_path: 트리밍 결과 저장 경로.
        **vad_kwargs: get_speech_timestamps에 전달할 추가 파라미터(threshold 등).

    Returns:
        {"original_duration_s", "trimmed_duration_s", "speech_ratio", "n_segments"}
    """
    data, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"16kHz 오디오만 지원(현재 {sr}Hz): {wav_path}")

    model = _get_model()
    wav = torch.from_numpy(data)
    timestamps = get_speech_timestamps(wav, model, **vad_kwargs)
    chunks = collect_chunks(timestamps, wav)

    sf.write(out_path, chunks.numpy(), sr)

    original_duration = len(data) / sr
    trimmed_duration = chunks.shape[0] / sr
    return {
        "original_duration_s": round(original_duration, 2),
        "trimmed_duration_s": round(trimmed_duration, 2),
        "speech_ratio": round(trimmed_duration / original_duration, 4) if original_duration else 0.0,
        "n_segments": len(timestamps),
    }
