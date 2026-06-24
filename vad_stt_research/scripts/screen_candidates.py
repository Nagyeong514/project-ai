"""
후보 YouTube 영상 사전 스크리닝.
VTT 자막만 다운로드해 무음 비율을 계산 — 오디오 다운로드 없이 수초 내 판별.

사용법:
  python scripts/screen_candidates.py --urls urls.txt --out screening_result.csv

urls.txt 형식 (한 줄에 URL 하나):
  https://www.youtube.com/watch?v=XXXX
  https://www.youtube.com/watch?v=YYYY
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def download_vtt(url: str, out_dir: str) -> Path | None:
    cmd = [
        "yt-dlp",
        "--no-write-auto-subs",
        "--write-subs",
        "--sub-lang", "ko",
        "--sub-format", "vtt",
        "--skip-download",
        "--output", str(Path(out_dir) / "%(id)s.%(ext)s"),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    vtts = list(Path(out_dir).glob("*.vtt"))
    return vtts[0] if vtts else None


_NON_SPEECH = re.compile(r"^[\(\[\{].*?[\)\]\}]$")
_TS_LINE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s-->\s(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)


def _is_speech(text_lines: list[str]) -> bool:
    """블록 텍스트가 실제 발화인지 판별. 비발화 주석만 있으면 False."""
    clean = re.sub(r"[\(\[\{].*?[\)\]\}]", "", " ".join(text_lines)).strip()
    return bool(clean)


def parse_vtt(vtt_path: Path) -> tuple[list[tuple[float, float]], float]:
    """타임스탬프 파싱. (발화 구간 목록, 마지막 종료 시각) 반환.
    비발화 주석([음악], (침묵) 등)은 무음으로 처리."""
    raw = vtt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", raw.strip())

    intervals = []
    last_end = 0.0

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
        last_end = max(last_end, end)

        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()
                      and not l.startswith("WEBVTT") and l.strip()]
        if end > start and _is_speech(text_lines):
            intervals.append((start, end))

    if not intervals:
        return [], last_end

    # 겹치는 구간 병합
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged], last_end


def _to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def compute_silence_ratio(intervals: list[tuple[float, float]], total: float) -> float:
    speech = sum(e - s for s, e in intervals)
    return 1.0 - speech / total if total > 0 else 0.0


def classify(silence_ratio: float, duration_min: float) -> str:
    if silence_ratio >= 0.50:
        if duration_min < 30:
            return "짧음(30분미만-high)"
        return "high_silence"
    if silence_ratio < 0.20:
        if duration_min < 60:
            return "짧음(60분미만-low)"
        return "low_silence"
    return "중간(20~50%)"


def screen_url(url: str, out_dir: str) -> dict:
    print(f"  처리 중: {url}")
    vtt = download_vtt(url, out_dir)
    if not vtt:
        print("    → 수동 한국어 자막 없음")
        return {"url": url, "status": "자막없음"}

    intervals, last_end = parse_vtt(vtt)
    if not intervals:
        return {"url": url, "status": "자막파싱실패"}

    duration_min = last_end / 60
    silence_ratio = compute_silence_ratio(intervals, last_end)
    group = classify(silence_ratio, duration_min)

    print(f"    길이: {duration_min:.1f}분 | 무음: {silence_ratio:.1%} | → {group}")
    return {
        "url": url,
        "status": "ok",
        "duration_min": round(duration_min, 1),
        "silence_ratio": round(silence_ratio, 4),
        "group": group,
        "vtt_file": vtt.name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True, help="URL 목록 텍스트 파일")
    parser.add_argument("--out", default="screening_result.csv")
    args = parser.parse_args()

    urls = [u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip()]
    print(f"총 {len(urls)}개 URL 스크리닝 시작\n")

    rows = []
    for url in urls:
        with tempfile.TemporaryDirectory(prefix="vtt_screen_") as tmp:
            rows.append(screen_url(url, tmp))

    import csv
    fieldnames = ["url", "status", "duration_min", "silence_ratio", "group", "vtt_file"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\n결과 저장: {args.out}")

    ok = [r for r in rows if r.get("status") == "ok"]
    low = [r for r in ok if r.get("group") == "low_silence"]
    high = [r for r in ok if r.get("group") == "high_silence"]
    mid = [r for r in ok if r.get("group") == "중간(20~50%)"]
    short = [r for r in ok if r.get("group") == "짧음(60분미만)"]
    no_sub = [r for r in rows if r.get("status") == "자막없음"]

    print(f"\n=== 스크리닝 결과 ===")
    print(f"  low_silence  (< 20%): {len(low)}개")
    print(f"  high_silence (≥ 50%): {len(high)}개")
    print(f"  중간 (20~50%):        {len(mid)}개 (실험 제외 권장)")
    print(f"  짧음 (60분 미만):     {len(short)}개 (실험 제외)")
    print(f"  자막 없음:            {len(no_sub)}개")


if __name__ == "__main__":
    main()
