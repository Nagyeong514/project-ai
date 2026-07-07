import math
import os
import shutil
import sys
from pathlib import Path

import cv2

from .extract_video import extract_clips, format_ts
from .motion_score import calculate_motion_score
from .select_clip import compare_strategies


STRATEGY_LABEL = {
    "A": "A_Threshold_AKS",
    "B": "B_CDF_MGSampler",
    "C": "C_AKS_Window",
}

# 짧은 영상용: 샘플링 후 유지율(원본 커버리지) 최소 목표
TARGET_RETENTION = 0.70


def build_summary_text(winner_key, results, video_name, output_dir,
                       clips_location="selected_clips"):
    """A/B/C 전략 비교 결과를 설명문으로 만든다."""
    lines = []
    lines.append(f"영상: {video_name}")
    lines.append(f"결과 폴더: {output_dir}")
    lines.append("")
    lines.append("A/B/C 전략 비교 요약")
    lines.append("-" * 40)

    for key, name in STRATEGY_LABEL.items():
        metrics = results[name]["metrics"]
        composite = (
            0.40 * metrics.get("temporal_coverage", 0.0)
            + 0.35 * (metrics.get("motion_relevance", 0.0) / 100.0)
            + 0.25 * metrics.get("uniformity", 0.0)
        )
        lines.append(
            f"[{key}] {name}: 클립수={metrics.get('n_clips', 0)}, "
            f"커버리지={metrics.get('temporal_coverage', 0.0):.4f}, "
            f"균등성={metrics.get('uniformity', 0.0):.4f}, "
            f"모션관련도={metrics.get('motion_relevance', 0.0):.2f}%, "
            f"종합점수={composite:.4f}"
        )

    winner_name = STRATEGY_LABEL[winner_key]
    lines.append("")
    lines.append(f"선택된 전략: {winner_name}")
    lines.append(
        "이유: A/B/C 중 종합점수가 가장 높은 전략을 선택했습니다."
    )
    lines.append("")
    lines.append("선택된 클립 타임스탬프 (원본 기준)")
    lines.append("-" * 40)
    for i, c in enumerate(results[winner_name]["clips"]):
        lines.append(
            f"클립{i+1:02d}: {format_ts(c['start'], ms=True)} ~ "
            f"{format_ts(c['end'], ms=True)} "
            f"(앵커 {format_ts(c['anchor'], ms=True)}, "
            f"모션스코어 {c.get('motion', 0.0):.2f})"
        )
    lines.append("")
    lines.append(f"저장 위치: {clips_location} 폴더 안에 분리된 영상이 저장됩니다.")
    return "\n".join(lines)


def _union_seconds(clips):
    """클립 구간들의 합집합 길이(초) — 겹침을 중복 계산하지 않는 실제 커버 시간"""
    intervals = sorted((c["start"], c["end"]) for c in clips)
    total = 0.0
    cur_s, cur_e = None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def choose_clip_window(duration_sec, n_clips,
                       target=TARGET_RETENTION, base_window=5):
    """유지율 target 이상이 되도록 클립 윈도우(앵커 ±초)를 계산"""
    needed = math.ceil(target * duration_sec / (2 * n_clips))
    return max(base_window, needed)


def build_result_table(stats_list):
    """영상별 원본 길이 / 샘플링 후 / 유지율 / 용량 요약표 문자열 생성"""
    lines = []
    lines.append("=" * 78)
    lines.append("  [샘플링 결과 요약]  원본 대비 유지율")
    lines.append("=" * 78)
    lines.append("영상 | 원본 길이 | 샘플링 후 | 유지율 | 용량")
    lines.append("-" * 78)
    for s in stats_list:
        lines.append(
            f"{s['video_name']} | {s['duration_sec']:.0f}초 | "
            f"{s['sampled_sec']:.0f}초 ({s['n_clips']}개) | "
            f"{s['retention_pct']:.1f}% | "
            f"{s['orig_mb']:.0f}MB → {s['sampled_mb']:.0f}MB"
        )
    lines.append("-" * 78)
    tot_dur = sum(s["duration_sec"] for s in stats_list)
    tot_samp = sum(s["sampled_sec"] for s in stats_list)
    tot_n = sum(s["n_clips"] for s in stats_list)
    tot_o = sum(s["orig_mb"] for s in stats_list)
    tot_s = sum(s["sampled_mb"] for s in stats_list)
    lines.append(
        f"전체 | {tot_dur:.0f}초 | {tot_samp:.0f}초 ({tot_n}개) | "
        f"{tot_samp / tot_dur * 100:.1f}% | {tot_o:.0f}MB → {tot_s:.0f}MB"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def run_abc_pipeline(video_path, output_root="abc_pipeline_outputs",
                     files_dir=None, clips_dir=None, reuse_csv=True):
    """기존 A/B/C 파이프라인을 그대로 연결해 결과 폴더를 만든다.

    files_dir / clips_dir를 지정하면 CSV·요약과 분리 영상을 각각
    해당 폴더에 저장한다. (미지정 시 output_root/영상명 아래에 함께 저장)
    클립 윈도우는 유지율 TARGET_RETENTION 이상이 되도록 자동 조절된다.

    Returns: 결과 통계 dict (요약표 생성용)
    """
    video_path = os.path.abspath(video_path)
    video_name = Path(video_path).stem
    output_dir = files_dir or os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{video_name}_motion_scores.csv")
    clip_dir = clips_dir or os.path.join(output_dir, "selected_clips")

    print(f"[ABC 파이프라인] 입력 영상: {video_path}")
    print(f"[ABC 파이프라인] 결과 폴더: {output_dir}")

    if reuse_csv and os.path.exists(csv_path):
        print(f"[ABC 파이프라인] 기존 모션 스코어 CSV 재사용: {csv_path}")
    else:
        calculate_motion_score(video_path, csv_path, sample_interval=1.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")
    duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    n_clips = max(3, int(duration_sec / 40))
    clip_window = choose_clip_window(duration_sec, n_clips)

    # 유지율(실제 커버 시간 기준)이 목표 미만이면 윈도우를 넓혀 재시도
    for _ in range(10):
        winner_key, results = compare_strategies(
            csv_path,
            n_clips=n_clips,
            clip_window=clip_window,
        )
        winning_clips = results[STRATEGY_LABEL[winner_key]]["clips"]
        coverage = _union_seconds(winning_clips) / duration_sec
        if coverage >= TARGET_RETENTION:
            break
        clip_window += 2
        print(f"[ABC 파이프라인] 유지율 {coverage*100:.1f}% < "
              f"{TARGET_RETENTION*100:.0f}% → 윈도우 ±{clip_window}s로 확대")

    print(f"[ABC 파이프라인] 클립 윈도우 ±{clip_window}s | "
          f"실제 커버리지 {coverage*100:.1f}%")

    winner_name = STRATEGY_LABEL[winner_key]
    saved = extract_clips(video_path, winning_clips, clip_dir)

    ts_csv = os.path.join(clip_dir, "clip_timestamps.csv")
    if os.path.exists(ts_csv) and os.path.abspath(clip_dir) != os.path.abspath(output_dir):
        shutil.copy2(ts_csv, os.path.join(output_dir, f"{video_name}_clip_timestamps.csv"))

    summary_text = build_summary_text(winner_key, results, video_name, output_dir,
                                      clips_location=clip_dir)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_text)

    print("\n" + "=" * 60)
    print(summary_text)
    print("=" * 60)
    print(f"분리된 영상 개수: {len(saved)}")
    print(f"요약 파일: {summary_path}")

    stats = {
        "video_name"   : video_name,
        "duration_sec" : duration_sec,
        "n_clips"      : len(saved),
        "sampled_sec"  : sum(c["end"] - c["start"] for c in winning_clips),
        "retention_pct": sum(c["end"] - c["start"] for c in winning_clips)
                         / duration_sec * 100,
        "coverage_pct" : coverage * 100,
        "clip_window"  : clip_window,
        "orig_mb"      : os.path.getsize(video_path) / 2**20,
        "sampled_mb"   : sum(os.path.getsize(p) for p in saved) / 2**20,
        "output_dir"   : output_dir,
    }
    return stats


if __name__ == "__main__":
    video_arg = sys.argv[1] if len(sys.argv) > 1 else "다시 돌아온 1인칭 컴퓨터 조립으로 다 알려드립니다..mp4"
    stats = run_abc_pipeline(video_arg)
    print("\n" + build_result_table([stats]))
