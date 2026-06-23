"""
YouTube 수동 자막을 내려받아 GT 텍스트 파일로 저장.
수동 자막이 없는 경우 --srt 로 로컬 SRT 파일을 직접 지정 가능.

사용법 (자동 다운로드):
  python scripts/prepare_ground_truth.py \
    --url "https://www.youtube.com/watch?v=..." \
    --file-id F01

사용법 (로컬 SRT):
  python scripts/prepare_ground_truth.py \
    --srt path/to/subtitle.srt \
    --file-id F01

결과: data/ground_truth/F01.txt (정규화 전 원문 저장)
주의: 정규화(normalizer.py)는 평가 시점에 적용 — GT 파일은 원문 보존.
"""
import argparse
import re
import subprocess
from pathlib import Path

GT_DIR = Path("data/ground_truth")


def _srt_to_text(srt_path: str) -> str:
    """SRT → 순수 텍스트 (타임코드·번호 제거)."""
    text_lines = []
    block = []
    with open(srt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if len(block) >= 3:
                    text_lines.extend(block[2:])
                block = []
            else:
                block.append(line)
    if len(block) >= 3:
        text_lines.extend(block[2:])
    return " ".join(text_lines)


def _download_manual_subtitle(url: str, file_id: str) -> str:
    """yt-dlp로 수동 자막(ko) 다운로드. 없으면 예외 발생."""
    tmp = GT_DIR / f"{file_id}_raw"
    cmd = [
        "yt-dlp",
        "--write-subs", "--no-write-auto-subs",
        "--sub-lang", "ko",
        "--sub-format", "srt",
        "--skip-download",
        "-o", str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    srt_files = list(GT_DIR.glob(f"{file_id}_raw*.srt"))
    if not srt_files:
        raise FileNotFoundError(
            f"수동 자막(ko)을 찾지 못했습니다. 영상에 한국어 수동 자막이 있는지 확인하세요."
        )
    return str(srt_files[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="YouTube URL (수동 자막 다운로드)")
    parser.add_argument("--srt", default=None, help="로컬 SRT 파일 경로")
    parser.add_argument("--file-id", required=True)
    args = parser.parse_args()

    GT_DIR.mkdir(parents=True, exist_ok=True)

    if args.srt:
        srt_path = args.srt
    elif args.url:
        srt_path = _download_manual_subtitle(args.url, args.file_id)
    else:
        raise ValueError("--url 또는 --srt 중 하나는 필수입니다.")

    text = _srt_to_text(srt_path)
    out_path = GT_DIR / f"{args.file_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"GT 저장: {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
