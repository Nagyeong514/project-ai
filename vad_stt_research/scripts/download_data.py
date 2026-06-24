"""
VAD 연구 데이터 다운로드 스크립트.
오디오(WAV 16kHz mono) + 수동 한국어 자막(VTT) 저장.

사용법:
  python scripts/download_data.py
"""
import subprocess
import wave
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
VTT_DIR = Path(__file__).parent.parent / "data" / "ground_truth"

VIDEOS = {
    "H01": ("https://youtu.be/8usMKRr-794", "high_silence"),
    "H02": ("https://youtu.be/zIrc8TlOmJM", "high_silence"),
    "H03": ("https://youtu.be/B9GsLAPeA2M", "high_silence"),
    "H04": ("https://youtu.be/M5WArJx1AFQ", "high_silence"),
    "H05": ("https://youtu.be/cRYEIHIdRfc", "high_silence"),
    "L01": ("https://youtu.be/E_FiWwlzzSY", "low_silence"),
    "L02": ("https://youtu.be/WOnrIm6im7Q", "low_silence"),
    "L03": ("https://youtu.be/DTUkwgvl6mE", "low_silence"),
    "L04": ("https://youtu.be/rxzz97oXVq8", "low_silence"),
    "L05": ("https://youtu.be/f0JFQTmhr9w", "low_silence"),
}


def download(file_id: str, url: str) -> bool:
    wav_path = RAW_DIR / f"{file_id}.wav"
    if wav_path.exists():
        print(f"[{file_id}] 이미 존재, 건너뜀")
        return True

    print(f"[{file_id}] 다운로드 시작: {url}")
    result = subprocess.run([
        "yt-dlp",
        "--no-write-auto-subs", "--write-subs", "--sub-lang", "ko", "--sub-format", "vtt",
        "-f", "bestaudio[ext=webm]/bestaudio",
        "--output", str(RAW_DIR / f"{file_id}.%(ext)s"),
        url,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[{file_id}] yt-dlp 실패:\n{result.stderr}")
        return False

    # 다운로드된 오디오 파일 찾기 (확장자 무관)
    audio_files = [
        f for f in RAW_DIR.glob(f"{file_id}.*")
        if f.suffix not in (".wav", ".vtt", ".ko.vtt")
    ]
    if not audio_files:
        print(f"[{file_id}] 오디오 파일 없음")
        return False

    audio_path = audio_files[0]
    print(f"[{file_id}] WAV 변환 중: {audio_path.name}")

    try:
        import av
        container = av.open(str(audio_path))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        pcm_chunks = []
        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                pcm_chunks.append(bytes(f.planes[0]))
        container.close()

        raw = b"".join(pcm_chunks)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw)

        audio_path.unlink()
        dur = len(raw) / 2 / 16000 / 60
        print(f"[{file_id}] 완료: {dur:.1f}분")

        # VTT → ground_truth 폴더로 이동
        vtt_files = list(RAW_DIR.glob(f"{file_id}*.vtt"))
        for vtt in vtt_files:
            dest = VTT_DIR / vtt.name
            vtt.rename(dest)
            print(f"[{file_id}] VTT 저장: {dest.name}")

        return True

    except Exception as e:
        print(f"[{file_id}] 변환 실패: {e}")
        return False


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    VTT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for file_id, (url, group) in VIDEOS.items():
        ok = download(file_id, url)
        results[file_id] = "ok" if ok else "fail"

    print("\n===== 완료 =====")
    for file_id, status in results.items():
        group = VIDEOS[file_id][1]
        print(f"  [{file_id}] {group:12s} {status}")


if __name__ == "__main__":
    main()
