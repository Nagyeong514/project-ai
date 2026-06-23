"""오디오 전처리: yt-dlp 다운로드 → 16kHz mono WAV 변환."""
import subprocess
from pathlib import Path


def download_audio(url: str, output_dir: str, file_id: str) -> str:
    """
    yt-dlp로 영상을 내려받아 16kHz mono WAV로 변환.
    반환: 저장된 WAV 경로
    """
    out_path = Path(output_dir) / f"{file_id}.wav"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(out_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)


def convert_to_wav(input_path: str, output_path: str) -> str:
    """기존 오디오/영상 파일을 16kHz mono WAV로 변환 (ffmpeg 필요)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def get_audio_duration(wav_path: str) -> float:
    """WAV 파일 재생 시간(초) 반환."""
    import soundfile as sf
    info = sf.info(wav_path)
    return info.duration
