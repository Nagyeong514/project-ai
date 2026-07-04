"""
STEP4 실행 본체 — YOLO 검출 + VLM 관찰. 영상 갈래 전용(STT/transcript 의존 없음 —
음성 갈래와의 합류는 STEP5에서 한 번만 일어난다, README의 두 갈래 다이어그램 참고).

입력: STEP3가 만든 output/_frames/<video_id>/frames_meta.json + jpg들.
산출물: output/detections/<video_id>.json, output/vlm_observations/<video_id>.observations.json.
"""

from __future__ import annotations

from typing import List

from tacit_common import artifacts
from tacit_common.config import PipelineConfig
from tacit_common.interfaces.detector import FrameRef
from tacit_common.schema.intermediate import Detection, FrameMeta
from step4_registry import build_detector, build_vlm


class Step4Runner:
    def __init__(self, cfg: PipelineConfig, detector=None, vlm=None):
        self.cfg = cfg
        self.detector = detector or build_detector(cfg.detector)
        self.vlm = vlm or build_vlm(cfg.vlm)
        self.frames_dir = cfg.paths.resolve(cfg.paths.step3_frames_dir)
        self.detections_dir = cfg.paths.resolve(cfg.paths.step4_detections_dir)
        self.observations_dir = cfg.paths.resolve(cfg.paths.step4_observations_dir)

    def _injected_parts(self, video_id: str, flat_dets: List[Detection]):
        detected = [d.cls for d in flat_dets]
        if hasattr(self.vlm, "_injected_parts_for"):
            return self.vlm._injected_parts_for(video_id, detected)
        return sorted(set(detected)) or None

    def run(self, video_id: str) -> None:
        fmeta = artifacts.load_frames_meta(self.frames_dir, video_id)
        frame_paths, times, fps = fmeta["frame_paths"], fmeta["times"], fmeta["fps"]
        # path=video_id는 자리채움용 — 이 스텝은 프레임 jpg만 다루고 원본 영상 파일은
        # 다시 열지 않는다(FrameRef.image가 항상 jpg 경로라 detector의 재디코드 폴백은 안 탐).
        meta = FrameMeta(video_id=video_id, path=video_id, fps=fps, n_frames=len(frame_paths))

        print(f"[STEP4][1/2] YOLO 검출: {video_id}")
        frame_refs = [FrameRef(frame_idx=i, timestamp=times[i], image=p)
                      for i, p in enumerate(frame_paths)]
        try:
            detections_by_frame = self.detector.detect(frame_refs, meta)
            flat_dets: List[Detection] = [d for fd in detections_by_frame for d in fd.detections]
        except Exception as e:  # YOLO 실패해도 진행(부품주입/위치힌트만 손해)
            print(f"      [WARN] YOLO 건너뜀: {e}")
            flat_dets = []
        artifacts.save_detections(self.detections_dir, video_id, flat_dets)
        if hasattr(self.detector, "unload"):
            self.detector.unload()

        print(f"[STEP4][2/2] VLM 관찰: {video_id}")
        injected = self._injected_parts(video_id, flat_dets)
        actions = self.vlm.observe_frames(frame_paths, times, injected_parts=injected)
        artifacts.save_observations(self.observations_dir, video_id, actions)
        if hasattr(self.vlm, "unload"):
            self.vlm.unload()

        print(f"      검출 {len(flat_dets)}건, 관찰 {len(actions)}건")
