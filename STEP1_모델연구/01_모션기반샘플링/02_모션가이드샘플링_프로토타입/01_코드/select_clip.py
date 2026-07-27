"""
select_clip.py
Motion Score CSV → 3가지 샘플링 전략 비교 + AKS Coverage 기반 최적 전략 선택

전략 A: Threshold + AKS Coverage     (원본 아이디어, 임계값 기반)
전략 B: CDF Inverse Sampling          (MGSampler, 논문 방식)
전략 C: AKS Time-Window ArgMax        (순수 AKS Coverage, 시간 균등 분할)
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _anchors_to_clips(anchors_sec, window_sec, total_duration):
    """앵커 타임스탬프 리스트 → (start, end) 클립 리스트"""
    clips = []
    for t in anchors_sec:
        s = round(max(0.0, t - window_sec), 1)
        e = round(min(total_duration, t + window_sec), 1)
        clips.append({"start": s, "end": e, "anchor": round(t, 1)})
    return clips


def _metrics(clips, scores_arr, times_arr, total_duration):
    """전략별 평가 지표 계산"""
    anchors = sorted([c["anchor"] for c in clips])
    if len(anchors) < 2:
        return {}

    gaps = [anchors[i+1] - anchors[i] for i in range(len(anchors)-1)]

    # 시간적 커버리지: 1 - max_gap/total (높을수록 빈틈 없음)
    temporal_coverage = round(1.0 - max(gaps) / total_duration, 4)

    # 균등성: 1 - CV(gaps) (높을수록 고르게 분산)
    cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 1.0
    uniformity = round(max(0.0, 1.0 - cv), 4)

    # 모션 관련성: 선택된 앵커들의 motion_score 평균 백분위
    sel_scores = []
    for a in anchors:
        idx = int(np.abs(times_arr - a).argmin())
        sel_scores.append(scores_arr[idx])
    pcts = [float(np.mean(scores_arr <= s)) * 100 for s in sel_scores]
    motion_relevance = round(np.mean(pcts), 2)

    # 압축률: 총 클립 길이 / 영상 길이
    total_clip_sec = sum(c["end"] - c["start"] for c in clips)
    compression_pct = round(total_clip_sec / total_duration * 100, 1)

    return {
        "n_clips"          : len(clips),
        "total_clip_min"   : round(total_clip_sec / 60, 1),
        "compression_pct"  : compression_pct,
        "temporal_coverage": temporal_coverage,
        "uniformity"       : uniformity,
        "motion_relevance" : motion_relevance,
        "max_gap_min"      : round(max(gaps) / 60, 1),
    }


def _composite_score(m):
    """AR 글래스 행동인식 기준 종합 점수 (가중합)"""
    return (0.40 * m["temporal_coverage"]
          + 0.35 * (m["motion_relevance"] / 100)
          + 0.25 * m["uniformity"])


# ──────────────────────────────────────────────────────────────────────────────
# 전략 구현
# ──────────────────────────────────────────────────────────────────────────────

def strategy_a_threshold_aks(times, scores, total_duration,
                              threshold_pct=75, clip_window=15,
                              merge_gap=10, coverage_min_gap=120):
    """
    전략 A: 상위 N% 임계값 → ±window 클립 확장 → 근접 병합 → AKS 최소 간격 필터
    """
    thresh = np.percentile(scores[scores > 0], threshold_pct)
    high_t = times[scores >= thresh]
    if len(high_t) == 0:
        return []

    raw = sorted([(max(0, t - clip_window), min(total_duration, t + clip_window))
                  for t in high_t])

    # 병합
    merged = []
    for s, e in raw:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # AKS Coverage: 최소 간격 필터
    covered, last = [], -9999
    for s, e in merged:
        if s - last >= coverage_min_gap:
            anchor = round((s + e) / 2, 1)
            covered.append({"start": round(s, 1), "end": round(e, 1), "anchor": anchor})
            last = s

    return covered


def strategy_b_cdf_sampling(times, scores, total_duration,
                             n_clips=12, clip_window=15):
    """
    전략 B: 누적 분포 함수(CDF) 역변환 → 모션 밀도 비례 샘플링 (MGSampler)
    """
    cum = np.cumsum(scores)
    if cum[-1] == 0:
        cum = np.cumsum(np.ones_like(scores))
    cdf = cum / cum[-1]

    targets  = np.linspace(1 / (2 * n_clips), 1 - 1 / (2 * n_clips), n_clips)
    anchors  = sorted(set(float(times[int(np.abs(cdf - t).argmin())]) for t in targets))
    return _anchors_to_clips(anchors, clip_window, total_duration)


def strategy_c_aks_window(times, scores, total_duration,
                           n_clips=12, clip_window=15):
    """
    전략 C: AKS 시간 균등 분할 + 윈도우 내 ArgMax (순수 Coverage 보장)
    """
    window_size = len(scores) // n_clips
    anchors = []
    for w in range(n_clips):
        s = w * window_size
        e = min((w + 1) * window_size, len(scores))
        local_best = int(np.argmax(scores[s:e]))
        anchors.append(float(times[s + local_best]))
    return _anchors_to_clips(sorted(set(anchors)), clip_window, total_duration)


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────

def compare_strategies(
    csv_path     = "motion_scores.csv",
    n_clips      = 12,
    clip_window  = 15,
):
    """
    3가지 전략을 모두 실행하고 비교표를 출력한 뒤 최적 전략과 클립 목록을 반환.

    Returns
    -------
    winner : str  ("A" | "B" | "C")
    results: dict  {strategy_name: {"clips": [...], "metrics": {...}}}
    """
    df = pd.read_csv(csv_path)
    times  = df["time_sec"].values.astype(float)
    scores = df["motion_score"].values.astype(float)
    total_duration = float(times[-1])

    print(f"\n[select_clip] CSV 로드 완료 | {len(df)}개 구간 | 총 {total_duration/60:.1f}분")
    print(f"[select_clip] motion_score 평균={scores.mean():.2f} | 최대={scores.max():.2f}\n")

    # ── 전략 실행 ──
    clips_a = strategy_a_threshold_aks(
        times, scores, total_duration,
        threshold_pct=75, clip_window=clip_window,
        merge_gap=clip_window, coverage_min_gap=total_duration / n_clips,
    )
    clips_b = strategy_b_cdf_sampling(
        times, scores, total_duration,
        n_clips=n_clips, clip_window=clip_window,
    )
    clips_c = strategy_c_aks_window(
        times, scores, total_duration,
        n_clips=n_clips, clip_window=clip_window,
    )

    strategies = {
        "A_Threshold_AKS" : clips_a,
        "B_CDF_MGSampler" : clips_b,
        "C_AKS_Window"    : clips_c,
    }

    # 각 클립에 앵커 시점의 motion_score 수치를 부착
    for clips in strategies.values():
        for c in clips:
            idx = int(np.abs(times - c["anchor"]).argmin())
            c["motion"] = round(float(scores[idx]), 2)

    # ── 지표 계산 ──
    results = {}
    for name, clips in strategies.items():
        m = _metrics(clips, scores, times, total_duration)
        results[name] = {"clips": clips, "metrics": m}

    # ── 비교표 출력 ──
    header = f"{'전략':<20} {'클립수':>5} {'총길이(분)':>10} {'압축률(%)':>10} " \
             f"{'커버리지':>10} {'균등성':>8} {'모션관련도':>10} {'종합점수':>8}"
    print("=" * len(header))
    print("  [전략 비교표]  AR 글래스 행동인식 기준")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    scores_map = {}
    for name, r in results.items():
        m = r["metrics"]
        if not m:
            print(f"{name:<20}  (선택 클립 없음)")
            scores_map[name] = 0.0
            continue
        cs = _composite_score(m)
        scores_map[name] = cs
        print(f"{name:<20} {m['n_clips']:>5} {m['total_clip_min']:>10.1f} "
              f"{m['compression_pct']:>9.1f}% "
              f"{m['temporal_coverage']:>10.4f} {m['uniformity']:>8.4f} "
              f"{m['motion_relevance']:>9.1f}% {cs:>8.4f}")

    print("=" * len(header))

    winner_name = max(scores_map, key=scores_map.get)
    winner_key  = winner_name[0]  # "A", "B", "C"

    print(f"\n[select_clip] 최적 전략: {winner_name}  (종합점수 {scores_map[winner_name]:.4f})")

    # 각 전략 선택 구간 상세 출력
    for name, r in results.items():
        tag = " ← 최적" if name == winner_name else ""
        print(f"\n  [{name}]{tag}")
        for i, c in enumerate(r["clips"]):
            s, e, a = c["start"], c["end"], c["anchor"]
            ms = scores[int(np.abs(times - a).argmin())]
            print(f"    클립{i+1:02d}: {s/60:.1f}분 ~ {e/60:.1f}분  "
                  f"(앵커 {a/60:.1f}분, motion={ms:.2f})")

    return winner_key, results


if __name__ == "__main__":
    winner, results = compare_strategies(
        csv_path   = "motion_scores.csv",
        n_clips    = 12,
        clip_window= 15,
    )
