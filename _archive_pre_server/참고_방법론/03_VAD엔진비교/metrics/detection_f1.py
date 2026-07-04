"""
Speech/non-speech 검출 정확도 (프레임 단위 P/R/F1).
예측 VAD 구간과 GT 발화 구간을 동일 프레임 격자에 라벨링해 비교.

segments: .start/.end (초) 속성을 가진 객체 리스트 (SpeechSegment 등).
"""
from typing import List

import numpy as np


def _frame_labels(segments: List, duration: float, frame_s: float) -> np.ndarray:
    n = int(duration / frame_s) + 1
    labels = np.zeros(n, dtype=bool)
    for seg in segments:
        a = max(0, int(seg.start / frame_s))
        b = min(n, int(seg.end / frame_s))
        if b > a:
            labels[a:b] = True
    return labels


def detection_f1(
    pred_segments: List,
    gt_segments: List,
    duration: float,
    frame_s: float = 0.01,
) -> dict:
    pred = _frame_labels(pred_segments, duration, frame_s)
    gt = _frame_labels(gt_segments, duration, frame_s)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
