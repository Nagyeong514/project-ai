"""
VAD 연구 데이터 다운로드 스크립트.
오디오(WAV 16kHz mono) + 수동 한국어 자막(VTT) 저장.
다운로드 후 3단계 사전 검증 실행.

사용법:
  python scripts/download_data.py
"""
import re
import subprocess
import wave
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
VTT_DIR = Path(__file__).parent.parent / "data" / "ground_truth"

VIDEOS = {
    "H01": ("https://youtu.be/8usMKRr-794",  "high_silence"),
    "H02": ("https://youtu.be/2B339eQDqMM",  "high_silence"),
    "H03": ("https://youtu.be/RviNZGjJBpI",  "high_silence"),  # 교체: 46.3분, 무음 61.4%
    "H04": ("https://youtu.be/9jcdOCsXSho",  "high_silence"),
    "H05": ("https://youtu.be/cRYEIHIdRfc",  "high_silence"),
    "L01": ("https://youtu.be/E_FiWwlzzSY",  "low_silence"),
    "L02": ("https://youtu.be/IS6PybD5t3M",  "low_silence"),   # 교체: 64.9분, 무음 4.7% (q-XmkvrNvis 음악콘텐츠로 폐기)
    "L03": ("https://youtu.be/DTUkwgvl6mE",  "low_silence"),
    "L04": ("https://youtu.be/1ZtxBPRSMZA",  "low_silence"),   # 교체: 112분, 무음 16.6%(실측). NGLOS는 실측 mid(43%)로 폐기
    "L05": ("https://youtu.be/f0JFQTmhr9w",  "low_silence"),
}


# ──────────────────────────────────────────────
# 검증 ①: VTT 기계번역 및 단어밀도 체크
# ──────────────────────────────────────────────
def validate_vtt(file_id: str, vtt_path: Path, duration_min: float) -> tuple[bool, str]:
    text = vtt_path.read_text(encoding="utf-8")

    # 영문자 비율 (15% 초과 → 기계번역 의심)
    en = sum(c.isascii() and c.isalpha() for c in text)
    ko = sum('가' <= c <= '힣' for c in text)
    if ko > 0 and en / (ko + en) > 0.15:
        return False, f"영문자 비율 {en/(ko+en):.1%} > 15% — 기계번역 의심"

    # 단어밀도 (30 words/min 미만 → 발화 너무 적음)
    clean = re.sub(r'[\(\[\{].*?[\)\]\}]', '', text)
    words = [w for w in clean.split()
             if not re.match(r'\d{2}:\d{2}', w)
             and 'WEBVTT' not in w and '-->' not in w]
    density = len(words) / max(duration_min, 1)
    if density < 30:
        return False, f"단어밀도 {density:.0f} words/min < 30 — 발화 너무 적음"

    return True, f"영문자 {en/(ko+en):.1%}, {density:.0f} words/min"


# ──────────────────────────────────────────────
# 검증 ②: WAV 변환 후 Silero VAD 빠른 스캔
# ──────────────────────────────────────────────
def validate_wav(file_id: str, wav_path: Path) -> tuple[bool, str]:
    try:
        import torch
        import soundfile as sf

        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        get_speech_timestamps = utils[0]

        audio, sr = sf.read(str(wav_path), dtype="float32")
        sample = audio[: sr * 300]  # 처음 5분만
        tensor = torch.from_numpy(sample).to(device)

        ts = get_speech_timestamps(tensor, model, sampling_rate=sr, threshold=0.5)
        speech_s = sum((t["end"] - t["start"]) / sr for t in ts)
        ratio = speech_s / (len(sample) / sr)

        if ratio < 0.05:
            return False, f"5분 샘플 발화율 {ratio:.1%} < 5% — VAD 감지 실패"
        return True, f"5분 샘플 발화율 {ratio:.1%}"

    except Exception as e:
        return False, f"VAD 검증 오류: {e}"


# ──────────────────────────────────────────────
# 다운로드 + 변환 + 검증
# ──────────────────────────────────────────────
def download(file_id: str, url: str) -> str:
    wav_path = RAW_DIR / f"{file_id}.wav"
    if wav_path.exists():
        print(f"[{file_id}] 이미 존재, 건너뜀")
        return "skip"

    print(f"[{file_id}] 다운로드: {url}")
    result = subprocess.run([
        "yt-dlp",
        "--no-write-auto-subs", "--write-subs", "--sub-lang", "ko", "--sub-format", "vtt",
        "-f", "bestaudio[ext=webm]/bestaudio",
        "--output", str(RAW_DIR / f"{file_id}.%(ext)s"),
        url,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[{file_id}] yt-dlp 실패:\n{result.stderr}")
        return "fail"

    # VTT 검증 ①
    vtt_files = list(RAW_DIR.glob(f"{file_id}*.vtt"))
    if not vtt_files:
        print(f"[{file_id}] VTT 없음 — 수동 자막 확인 필요")
        return "fail_no_vtt"

    vtt_path = vtt_files[0]

    # ── 오디오 WAV 변환 ──
    audio_files = [f for f in RAW_DIR.glob(f"{file_id}.*")
                   if f.suffix not in (".wav", ".vtt", ".ko.vtt")]
    if not audio_files:
        print(f"[{file_id}] 오디오 파일 없음")
        return "fail"

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
        dur_min = len(raw) / 2 / 16000 / 60
        print(f"[{file_id}] WAV 완료: {dur_min:.1f}분")

    except Exception as e:
        print(f"[{file_id}] WAV 변환 실패: {e}")
        return "fail_wav"

    # VTT → ground_truth 이동 + 검증 ①
    dest_vtt = VTT_DIR / vtt_path.name
    vtt_path.rename(dest_vtt)

    ok, msg = validate_vtt(file_id, dest_vtt, dur_min)
    if not ok:
        print(f"[{file_id}] ⚠️  GT 검증 실패: {msg}")
        return "fail_gt"
    print(f"[{file_id}] GT 검증 OK: {msg}")

    # WAV 검증 ②
    ok, msg = validate_wav(file_id, wav_path)
    if not ok:
        print(f"[{file_id}] ⚠️  VAD 검증 실패: {msg}")
        return "fail_vad"
    print(f"[{file_id}] VAD 검증 OK: {msg}")

    return "ok"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    VTT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for file_id, (url, group) in VIDEOS.items():
        status = download(file_id, url)
        results[file_id] = status

    print("\n===== 완료 =====")
    for file_id, status in results.items():
        group = VIDEOS[file_id][1]
        mark = "✅" if status in ("ok", "skip") else "❌"
        print(f"  {mark} [{file_id}] {group:12s} {status}")

    fails = [fid for fid, s in results.items() if s not in ("ok", "skip")]
    if fails:
        print(f"\n⚠️  검증 실패 파일: {', '.join(fails)} — 영상 교체 필요")


if __name__ == "__main__":
    main()
