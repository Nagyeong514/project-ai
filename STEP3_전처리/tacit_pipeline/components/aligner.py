"""
타임스탬프 정렬 (스펙 5.6).

- YOLO/VLM(프레임 기반)과 STT(초 기반)를 공통 초 단위로 변환(이미 schema가 초로 통일).
- 발화·행동은 동시에 안 일어난다(말 먼저/행동 먼저/무언) → 1:1 매칭 금지.
- **윈도우(±N초)** 로 '구간' 단위로 후보를 묶어 LLM에 넘긴다.

이 모듈은 모델을 안 쓴다(순수 로직). 따라서 오늘 바로 테스트 가능한 부분.
"""

from __future__ import annotations

from typing import List

from ..schema.intermediate import (
    ActionDescription,
    AlignedWindow,
    Detection,
    Transcript,
    Utterance,
)


class WindowAligner:
    """행동 timestamp를 기준 앵커로, ±window_sec 안의 발화/검출을 묶는다. registry 키: 'window'.

    전략(앵커):
      - 행동(VLM action)이 있으면 그것을 앵커로 윈도우 생성.
      - 어느 행동 윈도우에도 안 걸린 발화는 'utterance_only' 윈도우로 별도 수거(스펙 5.7 케이스3).
    """

    def __init__(self, window_sec: float = 4.0, merge_overlapping: bool = True, **extra):
        self.window_sec = window_sec
        self.merge_overlapping = merge_overlapping

    def align(
        self,
        actions: List[ActionDescription],
        transcript: Transcript,
        detections: List[Detection] | None = None,
    ) -> List[AlignedWindow]:
        detections = detections or []
        utterances = sorted(transcript.utterances, key=lambda u: u.start)
        windows: List[AlignedWindow] = []
        used_utt: set[int] = set()

        # 1) 행동 앵커 윈도우
        for act in sorted(actions, key=lambda a: a.timestamp):
            t0 = act.timestamp - self.window_sec
            t1 = act.timestamp + self.window_sec
            w_utts = [
                u for i, u in enumerate(utterances) if self._overlaps(u.start, u.end, t0, t1)
            ]
            for i, u in enumerate(utterances):
                if self._overlaps(u.start, u.end, t0, t1):
                    used_utt.add(i)
            w_dets = [d for d in detections if t0 <= d.timestamp <= t1]
            windows.append(
                AlignedWindow(
                    window_start=t0,
                    window_end=t1,
                    actions=[act],
                    utterances=w_utts,
                    detections=w_dets,
                )
            )

        # 2) 어디에도 안 걸린 발화 → utterance_only 윈도우 (케이스3 보존)
        leftover = [u for i, u in enumerate(utterances) if i not in used_utt]
        for u in leftover:
            windows.append(
                AlignedWindow(
                    window_start=u.start,
                    window_end=u.end,
                    actions=[],
                    utterances=[u],
                    detections=[d for d in detections if u.start <= d.timestamp <= u.end],
                )
            )

        windows.sort(key=lambda w: w.window_start)
        if self.merge_overlapping:
            windows = self._merge(windows)
        return windows

    @staticmethod
    def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
        return a0 <= b1 and b0 <= a1

    def _merge(self, windows: List[AlignedWindow]) -> List[AlignedWindow]:
        """시간 겹치는 인접 윈도우 병합(중복 후보 난립 방지)."""
        if not windows:
            return windows
        merged = [windows[0]]
        for w in windows[1:]:
            last = merged[-1]
            if w.window_start <= last.window_end:
                last.window_end = max(last.window_end, w.window_end)
                last.actions.extend(w.actions)
                # 발화/검출은 timestamp로 중복 제거
                self._extend_unique_utt(last.utterances, w.utterances)
                self._extend_unique_det(last.detections, w.detections)
            else:
                merged.append(w)
        return merged

    @staticmethod
    def _extend_unique_utt(dst: List[Utterance], src: List[Utterance]) -> None:
        seen = {(u.start, u.end) for u in dst}
        for u in src:
            if (u.start, u.end) not in seen:
                dst.append(u)
                seen.add((u.start, u.end))

    @staticmethod
    def _extend_unique_det(dst: List[Detection], src: List[Detection]) -> None:
        seen = {(d.frame_idx, d.cls) for d in dst}
        for d in src:
            if (d.frame_idx, d.cls) not in seen:
                dst.append(d)
                seen.add((d.frame_idx, d.cls))
