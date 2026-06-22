"""
타임스탬프 정밀도 평가 (세그먼트 레벨, 연구 1차).
Timestamp Drift (Δt): 오디오 후반부(≥ 50분) 발화 시작 시점의 |예측 - 정답| 평균.
"""
from typing import List, Tuple

import numpy as np

from pipeline.stt.faster_whisper_runner import STTSegment
from pipeline.vad.base import SpeechSegment


def compute_timestamp_drift(
    pred_segments: List[STTSegment],
    gt_segments: List[SpeechSegment],
    late_section_start_min: float = 50.0,
    match_tolerance_s: float = 5.0,
) -> dict:
    """
    각 GT 세그먼트에 가장 가까운 예측 세그먼트를 매칭해 |Δt| 계산.
    match_tolerance_s: 이 범위 내에 있어야 매칭으로 인정.
    """
    late_start_s = late_section_start_min * 60.0
    all_drifts = []
    late_drifts = []

    for gt_seg in gt_segments:
        best_delta, best_pred = _find_closest(pred_segments, gt_seg.start, match_tolerance_s)
        if best_pred is not None:
            delta = abs(best_pred.start - gt_seg.start)
            all_drifts.append(delta)
            if gt_seg.start >= late_start_s:
                late_drifts.append(delta)

    return {
        "mean_drift_all_s": float(np.mean(all_drifts)) if all_drifts else None,
        "mean_drift_late_s": float(np.mean(late_drifts)) if late_drifts else None,
        "n_matched_all": len(all_drifts),
        "n_matched_late": len(late_drifts),
        "drift_by_time": _drift_by_time_bucket(pred_segments, gt_segments, match_tolerance_s),
    }


def _find_closest(
    pred_segments: List[STTSegment],
    target_start: float,
    tolerance_s: float,
) -> Tuple[float, "STTSegment | None"]:
    best_delta = float("inf")
    best_seg = None
    for seg in pred_segments:
        delta = abs(seg.start - target_start)
        if delta < best_delta and delta <= tolerance_s:
            best_delta = delta
            best_seg = seg
    return best_delta, best_seg


def _drift_by_time_bucket(
    pred_segments: List[STTSegment],
    gt_segments: List[SpeechSegment],
    tolerance_s: float,
    bucket_min: float = 10.0,
) -> List[dict]:
    """오디오를 bucket_min 분 단위로 나눠 구간별 평균 드리프트 계산."""
    max_time = max((gt.start for gt in gt_segments), default=0.0)
    bucket_s = bucket_min * 60.0
    n_buckets = int(max_time / bucket_s) + 1

    buckets = [[] for _ in range(n_buckets)]
    for gt_seg in gt_segments:
        _, best_pred = _find_closest(pred_segments, gt_seg.start, tolerance_s)
        if best_pred is not None:
            b = int(gt_seg.start / bucket_s)
            buckets[b].append(abs(best_pred.start - gt_seg.start))

    result = []
    for i, drifts in enumerate(buckets):
        result.append({
            "bucket_start_min": i * bucket_min,
            "bucket_end_min": (i + 1) * bucket_min,
            "mean_drift_s": float(np.mean(drifts)) if drifts else None,
            "n_samples": len(drifts),
        })
    return result
