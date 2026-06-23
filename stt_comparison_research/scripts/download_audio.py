"""
yt-dlp로 YouTube 영상을 16kHz mono WAV로 내려받는 스크립트.

사용법:
  python scripts/download_audio.py \
    --url "https://www.youtube.com/watch?v=..." \
    --file-id F01 \
    --type A \
    --output-dir data/raw

결과: data/raw/F01.wav, data/metadata.csv에 행 추가
"""
import argparse
import csv
import os
from pathlib import Path

from pipeline.audio.preprocessor import download_audio, get_audio_duration

METADATA_PATH = Path("data/metadata.csv")
METADATA_COLS = ["file_id", "utterance_type", "duration_s", "wav_path", "url"]


def _init_metadata():
    if not METADATA_PATH.exists():
        with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=METADATA_COLS).writeheader()


def _append_metadata(row: dict):
    _init_metadata()
    with open(METADATA_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=METADATA_COLS).writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--type", required=True, choices=["A", "B"],
                        help="A=낭독·강연 / B=대화·인터뷰")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wav_path = download_audio(args.url, args.output_dir, args.file_id)
    duration = get_audio_duration(wav_path)

    _append_metadata({
        "file_id": args.file_id,
        "utterance_type": args.type,
        "duration_s": round(duration, 1),
        "wav_path": wav_path,
        "url": args.url,
    })
    print(f"저장 완료: {wav_path} ({duration:.1f}s) → metadata 기록됨")


if __name__ == "__main__":
    main()
