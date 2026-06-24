"""
VTT 수동 자막 → GT JSON 변환 스크립트.

출력 형식:
  {
    "text": "전체 발화 텍스트",
    "segments": [{"start": 0.0, "end": 2.5}, ...]
  }

- 비발화 주석([음악], (침묵) 등)은 무음으로 처리
- 겹치는 구간은 병합

사용법:
  python scripts/prepare_ground_truth.py
  python scripts/prepare_ground_truth.py --vtt_dir data/ground_truth --out_dir data/ground_truth
"""
import argparse
import json
import re
from pathlib import Path

GT_DIR = Path(__file__).parent.parent / "data" / "ground_truth"

_TS_LINE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s-->\s(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)


def _to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _is_speech(text_lines: list[str]) -> bool:
    clean = re.sub(r"[\(\[\{].*?[\)\]\}]", "", " ".join(text_lines)).strip()
    return bool(clean)


def _clean_text(text_lines: list[str]) -> str:
    lines = [re.sub(r"[\(\[\{].*?[\)\]\}]", "", l).strip() for l in text_lines]
    return " ".join(l for l in lines if l)


def parse_vtt(vtt_path: Path) -> tuple[list[dict], list[tuple[float, float]]]:
    """VTT 파싱. (텍스트 블록 목록, 발화 구간 목록) 반환."""
    raw = vtt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", raw.strip())

    text_blocks = []
    intervals = []

    for block in blocks:
        lines = block.strip().splitlines()
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue
        m = _TS_LINE.search(ts_line)
        if not m:
            continue

        start = _to_sec(m.group(1))
        end = _to_sec(m.group(2))

        text_lines = [
            l for l in lines
            if "-->" not in l
            and not l.strip().isdigit()
            and not l.startswith("WEBVTT")
            and not l.startswith("Kind:")
            and not l.startswith("Language:")
            and l.strip()
        ]

        if end <= start or not _is_speech(text_lines):
            continue

        text = _clean_text(text_lines)
        text_blocks.append({"start": start, "end": end, "text": text})
        intervals.append((start, end))

    return text_blocks, intervals


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def build_gt(vtt_path: Path) -> dict:
    text_blocks, intervals = parse_vtt(vtt_path)
    merged = merge_intervals(intervals)

    # 텍스트: 시간순 블록 텍스트 연결 (중복 제거)
    seen = set()
    texts = []
    for b in text_blocks:
        t = b["text"]
        if t not in seen:
            seen.add(t)
            texts.append(t)

    return {
        "text": " ".join(texts),
        "segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in merged],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtt_dir", default=str(GT_DIR))
    parser.add_argument("--out_dir", default=str(GT_DIR))
    args = parser.parse_args()

    vtt_dir = Path(args.vtt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vtt_files = sorted(vtt_dir.glob("*.vtt"))
    if not vtt_files:
        print("VTT 파일 없음")
        return

    for vtt_path in vtt_files:
        file_id = vtt_path.name.split(".")[0]
        out_path = out_dir / f"{file_id}.json"

        if out_path.exists():
            print(f"[{file_id}] 이미 존재, 건너뜀")
            continue

        gt = build_gt(vtt_path)
        dur_min = gt["segments"][-1]["end"] / 60 if gt["segments"] else 0
        n_seg = len(gt["segments"])
        n_chars = len(gt["text"])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        print(f"[{file_id}] {dur_min:.1f}분 | {n_seg}개 구간 | {n_chars}자 → {out_path.name}")


if __name__ == "__main__":
    main()
