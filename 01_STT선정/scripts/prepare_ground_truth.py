"""
YouTube 수동 자막을 내려받아 GT 텍스트 파일로 저장.
수동 자막이 없는 경우 --vtt / --srt 로 로컬 파일을 직접 지정 가능.

사용법 (자동 다운로드):
  python /home/piai/project-ai/stt_comparison_research/scripts/prepare_ground_truth.py \
    --url "https://www.youtube.com/watch?v=..." \
    --file-id F01 \
    --gt-dir /home/piai/project-ai/stt_comparison_research/data/ground_truth

사용법 (로컬 VTT):
  python /home/piai/project-ai/stt_comparison_research/scripts/prepare_ground_truth.py \
    --vtt path/to/subtitle.ko.vtt \
    --file-id F01 \
    --gt-dir /home/piai/project-ai/stt_comparison_research/data/ground_truth

결과: {gt-dir}/F01.txt (정규화 전 원문 보존 — 정규화는 평가 시점 적용)
"""
import argparse
import re
import subprocess
from pathlib import Path

_DEFAULT_GT_DIR = Path("/home/piai/project-ai/stt_comparison_research/data/ground_truth")


def _vtt_to_text(vtt_path: str) -> str:
    """VTT → 순수 텍스트 (타임코드·태그·헤더 제거, 중복 줄 제거)."""
    blocks = re.split(r"\n\n+", Path(vtt_path).read_text(encoding="utf-8").strip())
    seen, lines = set(), []
    for block in blocks:
        rows = block.strip().splitlines()
        if not any("-->" in r for r in rows):
            continue
        for row in rows:
            if "-->" in row or row.startswith(("WEBVTT", "Kind:", "Language:")):
                continue
            row = re.sub(r"<[^>]+>", "", row).strip()  # <c>, <00:00:..> 태그 제거
            if row and row not in seen:
                seen.add(row)
                lines.append(row)
    return " ".join(lines)


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


def _download_manual_subtitle(url: str, file_id: str, gt_dir: Path) -> str:
    """yt-dlp로 수동 자막(ko, VTT) 다운로드. 없으면 예외 발생."""
    tmp = gt_dir / f"{file_id}_raw"
    cmd = [
        "yt-dlp",
        "--write-subs", "--no-write-auto-subs",
        "--sub-lang", "ko",
        "--sub-format", "vtt",
        "--skip-download",
        "-o", str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    vtt_files = list(gt_dir.glob(f"{file_id}_raw*.vtt"))
    if not vtt_files:
        raise FileNotFoundError(
            "수동 자막(ko)을 찾지 못했습니다. 영상에 한국어 수동 자막이 있는지 확인하세요."
        )
    return str(vtt_files[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="YouTube URL (수동 자막 자동 다운로드)")
    parser.add_argument("--vtt", default=None, help="로컬 VTT 파일 경로")
    parser.add_argument("--srt", default=None, help="로컬 SRT 파일 경로")
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--gt-dir", default=str(_DEFAULT_GT_DIR), help="GT 저장 디렉터리 (절대경로)")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    if args.vtt:
        text = _vtt_to_text(args.vtt)
    elif args.srt:
        text = _srt_to_text(args.srt)
    elif args.url:
        vtt_path = _download_manual_subtitle(args.url, args.file_id, gt_dir)
        text = _vtt_to_text(vtt_path)
    else:
        raise ValueError("--url, --vtt, --srt 중 하나는 필수입니다.")

    out_path = gt_dir / f"{args.file_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"GT 저장: {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
