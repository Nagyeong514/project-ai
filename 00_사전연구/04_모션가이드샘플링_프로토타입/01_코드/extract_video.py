"""
extract_video.py
선택된 구간을 ffmpeg subprocess로 개별 클립 추출 (MoviePy 불필요)
"""

import csv
import os
import subprocess
import imageio_ffmpeg


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def format_ts(sec, ms=False):
    """초 → 'HH:MM:SS' 또는 'HH:MM:SS.mmm'(STT 대조용 밀리초 포함) 타임스탬프"""
    total_ms = int(round(sec * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}.{msec:03d}" if ms else base


def format_ts_filename(sec):
    """초 → 파일명용 타임스탬프 문자열 (예: 00m07s)"""
    sec = int(round(sec))
    return f"{sec // 60:02d}m{sec % 60:02d}s"


def extract_clips(video_path, clips, output_dir):
    """
    clips: [{"start": float, "end": float, "anchor": float}, ...]
    각 클립을 output_dir 아래에 clip_01_00m07s-00m37s.mp4 형태
    (원본 기준 시작-끝 타임스탬프 포함)로 저장하고,
    clip_timestamps.csv에 타임스탬프 목록을 기록.

    Returns: list of saved paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    ts_rows = []

    for i, c in enumerate(clips):
        start    = c["start"]
        duration = round(c["end"] - c["start"], 2)
        ts_name  = f"{format_ts_filename(start)}-{format_ts_filename(c['end'])}"
        out_path = os.path.join(output_dir, f"clip_{i+1:02d}_{ts_name}.mp4")

        # 재인코딩 방식: 키프레임 제약 없이 타임스탬프에 프레임 단위로 정확히 절단
        cmd = [
            FFMPEG, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-an",
            "-loglevel", "error",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            size_kb = os.path.getsize(out_path) // 1024
            print(f"  [{i+1:02d}] {os.path.basename(out_path)}  "
                  f"({format_ts(start)} ~ {format_ts(c['end'])}, {size_kb}KB)")
            saved.append(out_path)
            ts_rows.append({
                "clip_no"     : i + 1,
                "filename"    : os.path.basename(out_path),
                "start_sec"   : round(start, 1),
                "end_sec"     : round(c["end"], 1),
                "anchor_sec"  : round(c["anchor"], 1),
                "start_ts"    : format_ts(start, ms=True),
                "end_ts"      : format_ts(c["end"], ms=True),
                "duration_sec": duration,
                "motion_score": c.get("motion", ""),
            })
        else:
            print(f"  [{i+1:02d}] 오류: {result.stderr.decode()[:80]}")

    if ts_rows:
        ts_csv = os.path.join(output_dir, "clip_timestamps.csv")
        with open(ts_csv, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ts_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ts_rows)
        print(f"  타임스탬프 CSV 저장 → {ts_csv}")

    return saved


def extract_all_strategies(video_path, results, base_output_dir="strategy_clips"):
    """
    results: select_clip.compare_strategies() 반환값의 두 번째 값
    전략별 서브디렉토리에 클립 추출
    """
    summary = {}
    for strategy_name, r in results.items():
        out_dir = os.path.join(base_output_dir, strategy_name)
        print(f"\n[extract_video] [{strategy_name}] → {out_dir}")
        saved = extract_clips(video_path, r["clips"], out_dir)
        summary[strategy_name] = saved
    return summary


if __name__ == "__main__":
    from select_clip import compare_strategies

    VIDEO  = "다시 돌아온 1인칭 컴퓨터 조립으로 다 알려드립니다..mp4"
    winner, results = compare_strategies("motion_scores.csv", n_clips=12, clip_window=15)

    winning_clips = results[f"{'A_Threshold_AKS' if winner=='A' else 'B_CDF_MGSampler' if winner=='B' else 'C_AKS_Window'}"]["clips"]
    extract_clips(VIDEO, winning_clips, f"output_best_{winner}")
