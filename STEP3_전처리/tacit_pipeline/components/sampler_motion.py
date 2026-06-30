"""
프레임 샘플링 — motion guided sampling (스펙 5.1).

핵심: 스마트글래스는 머리에 달려 ego-motion(고개 돌림)이 화면 전체를 움직인다.
→ 순수 픽셀 motion 금지. **YOLO가 잡은 hand/도구/부품 bbox의 움직임**을 motion으로 정의.

샘플 기준(합집합):
  (1) 손/객체 bbox 움직임이 임계 이상  OR
  (2) 해당 구간에 발화 존재(안 움직여도 중요한 정적 점검)  OR
  (3) hand-부품 bbox 근접/접촉(조작 중)

전략 교체 가능: UniformSampler / MotionGuidedSampler / (확장).
오늘은 디코드/모델 호출 안 함 — 파라미터·로직 골격만. 실제 프레임 디코드는 내일.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..constants import HAND_CLASSES, MOTION_TRACK_CLASSES, PART_CLASSES
from ..interfaces.detector import FrameRef
from ..schema.intermediate import BBox, FrameDetections, FrameMeta, Transcript


def _iou(a: BBox, b: BBox) -> float:
    ax2, ay2, bx2, by2 = a.x + a.w, a.y + a.h, b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _center_dist(a: BBox, b: BBox) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


class UniformSampler:
    """가장 단순한 baseline. registry 키: 'uniform'. 디버깅/비교용."""

    def __init__(self, every_sec: float = 2.0, **extra: Any):
        self.every_sec = every_sec

    def sample(self, meta: FrameMeta, coarse_detections=None, transcript=None) -> List[FrameRef]:
        step = max(1, int(self.every_sec * meta.fps))
        return [
            FrameRef(frame_idx=i, timestamp=meta.frame_to_sec(i))
            for i in range(0, max(1, meta.n_frames), step)
        ]


class MotionGuidedSampler:
    """hybrid motion-guided 샘플러. registry 키: 'motion_guided'.

    coarse_detections(성긴 YOLO 검출)와 transcript(발화 구간)를 받아 위 3기준 합집합으로 고른다.
    coarse_detections가 없으면 오케스트레이터가 먼저 성긴 검출을 만들어 넘겨야 한다.
    """

    def __init__(
        self,
        motion_thresh_px: float = 15.0,  # bbox 중심 이동 임계(픽셀)
        contact_iou: float = 0.05,  # hand-부품 접촉 판단 IoU
        contact_dist_px: float = 80.0,  # 근접 판단 거리
        speech_pad_sec: float = 0.5,  # 발화 구간 패딩
        min_gap_sec: float = 0.4,  # 너무 촘촘한 샘플 억제(중복 제거)
        **extra: Any,
    ):
        self.motion_thresh_px = motion_thresh_px
        self.contact_iou = contact_iou
        self.contact_dist_px = contact_dist_px
        self.speech_pad_sec = speech_pad_sec
        self.min_gap_sec = min_gap_sec

    def sample(
        self,
        meta: FrameMeta,
        coarse_detections: Optional[List[FrameDetections]] = None,
        transcript: Optional[Transcript] = None,
    ) -> List[FrameRef]:
        chosen: dict[int, float] = {}  # frame_idx -> timestamp

        if coarse_detections:
            chosen.update(self._motion_and_contact_frames(coarse_detections))

        if transcript:
            chosen.update(self._speech_frames(meta, transcript))

        # 시간순 정렬 + 최소 간격으로 thinning
        ordered = sorted(chosen.items(), key=lambda kv: kv[1])
        out: List[FrameRef] = []
        last_t = -1e9
        for fidx, t in ordered:
            if t - last_t >= self.min_gap_sec:
                out.append(FrameRef(frame_idx=fidx, timestamp=t))
                last_t = t
        return out

    # ── 기준 (1)+(3): 움직임 / 접촉 ────────────────────────────────────
    def _motion_and_contact_frames(self, dets: List[FrameDetections]) -> dict[int, float]:
        picked: dict[int, float] = {}
        prev_centers: dict[str, BBox] = {}
        for fd in dets:
            hands = [d for d in fd.detections if d.cls in HAND_CLASSES]
            parts = [d for d in fd.detections if d.cls in PART_CLASSES]

            # (3) hand-부품 접촉/근접 → 조작 중
            for h in hands:
                for p in parts:
                    if _iou(h.bbox, p.bbox) >= self.contact_iou or _center_dist(
                        h.bbox, p.bbox
                    ) <= self.contact_dist_px:
                        picked[fd.frame_idx] = fd.timestamp

            # (1) 추적 대상 bbox 중심 이동량(ego-motion 무시: 클래스 bbox 기준)
            for d in fd.detections:
                if d.cls not in MOTION_TRACK_CLASSES:
                    continue
                prev = prev_centers.get(d.cls)
                if prev is not None and _center_dist(prev, d.bbox) >= self.motion_thresh_px:
                    picked[fd.frame_idx] = fd.timestamp
                prev_centers[d.cls] = d.bbox
        return picked

    # ── 기준 (2): 발화 존재 구간 ──────────────────────────────────────
    def _speech_frames(self, meta: FrameMeta, transcript: Transcript) -> dict[int, float]:
        picked: dict[int, float] = {}
        for u in transcript.utterances:
            t0 = max(0.0, u.start - self.speech_pad_sec)
            t1 = u.end + self.speech_pad_sec
            # 발화 구간 양 끝 + 중앙을 대표 프레임으로(촘촘함은 min_gap_sec가 정리)
            for t in (t0, (t0 + t1) / 2, t1):
                fidx = meta.sec_to_frame(t)
                picked[fidx] = meta.frame_to_sec(fidx)
        return picked
