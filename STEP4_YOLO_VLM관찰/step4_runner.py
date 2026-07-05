"""
STEP4 실행 본체 — YOLO 검출 + VLM 관찰. 영상 갈래 전용(STT/transcript 의존 없음 —
음성 갈래와의 합류는 STEP5에서 한 번만 일어난다, README의 두 갈래 다이어그램 참고).

입력: STEP3가 만든 output/_frames/<video_id>/frames_meta.json + jpg들.
산출물: output/detections/<video_id>.json, output/vlm_observations/<video_id>.observations.json.

**YOLO는 서브프로세스로 격리해서 돈다(2026-07-04).** ultralytics가 select_device() 안에서
os.environ["CUDA_VISIBLE_DEVICES"]를 무조건 덮어써서(`.predict()` 최초 호출 시점, 실측 확인),
같은 프로세스에서 이어서 VLM을 device_map="auto"로 띄우면 GPU 1장에 갇혀버린다(실측: CLIP1
99프레임 단일GPU 21.97GiB OOM). 자식 프로세스는 os.environ 복사본을 받으므로, YOLO를
step4_components/yolo_subprocess_entry.py로 완전히 분리하면 이 오염이 부모(=이후 VLM을
로드할 이 프로세스)로 새어나가지 않는다. `detector` 인자를 명시적으로 주입한 경우(테스트/목킹용)
에만 예외적으로 인프로세스 호출 — 그 외에는 항상 서브프로세스를 거친다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tacit_common import artifacts
from tacit_common.config import PipelineConfig
from tacit_common.schema.intermediate import Detection
from step4_registry import build_vlm


class Step4Runner:
    def __init__(self, cfg: PipelineConfig, config_path: Optional[str] = None, detector=None, vlm=None):
        self.cfg = cfg
        self.config_path = config_path  # YOLO 서브프로세스에 그대로 넘길 config 경로
        self._injected_detector = detector  # None이 아니면 서브프로세스 대신 인프로세스 호출(테스트용)
        self.vlm = vlm or build_vlm(cfg.vlm)
        self.frames_dir = cfg.paths.resolve(cfg.paths.step3_frames_dir)
        self.detections_dir = cfg.paths.resolve(cfg.paths.step4_detections_dir)
        self.observations_dir = cfg.paths.resolve(cfg.paths.step4_observations_dir)

    def _injected_parts(self, video_id: str, flat_dets: List[Detection]):
        detected = [d.cls for d in flat_dets]
        if hasattr(self.vlm, "_injected_parts_for"):
            return self.vlm._injected_parts_for(video_id, detected)
        return sorted(set(detected)) or None

    def _run_yolo_subprocess(self, video_id: str) -> None:
        if not self.config_path:
            raise ValueError(
                "Step4Runner(config_path=...) 없이는 YOLO 서브프로세스를 못 띄운다 — "
                "테스트/목킹 목적이면 detector=... 를 주입해서 인프로세스 경로를 쓸 것."
            )
        script = Path(__file__).resolve().parent / "step4_components" / "yolo_subprocess_entry.py"
        cmd = [sys.executable, str(script), "--config", self.config_path, "--video-id", video_id]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"YOLO 서브프로세스 실패(exit={result.returncode}): {video_id}")

    def run(self, video_id: str) -> None:
        fmeta = artifacts.load_frames_meta(self.frames_dir, video_id)
        frame_paths, times, fps = fmeta["frame_paths"], fmeta["times"], fmeta["fps"]

        print(f"[STEP4][1/2] YOLO 검출: {video_id}")
        if self._injected_detector is not None:
            # 테스트/목킹 전용 — 실제 파이프라인에서는 안 씀(CUDA_VISIBLE_DEVICES 오염 위험 감수).
            # 여기서도 예외를 삼키지 않는다(빈 결과로 조용히 계속하는 게 최악의 실패 모드).
            from tacit_common.interfaces.detector import FrameRef
            from tacit_common.schema.intermediate import FrameMeta
            meta = FrameMeta(video_id=video_id, path=video_id, fps=fps, n_frames=len(frame_paths))
            frame_refs = [FrameRef(frame_idx=i, timestamp=times[i], image=p)
                          for i, p in enumerate(frame_paths)]
            try:
                detections_by_frame = self._injected_detector.detect(frame_refs, meta)
                flat_dets: List[Detection] = [d for fd in detections_by_frame for d in fd.detections]
            finally:
                if hasattr(self._injected_detector, "unload"):
                    self._injected_detector.unload()
            artifacts.save_detections(self.detections_dir, video_id, flat_dets)
        else:
            self._run_yolo_subprocess(video_id)
            flat_dets = artifacts.load_detections(self.detections_dir, video_id)

        # 검출 0건 방어(2026-07-05): 손이 계속 나오는 수리 영상에서 전 프레임 0건은
        # "검출할 게 없음"이 아니라 사실상 항상 파이프라인/환경 고장이다(실제로 3일간
        # 4클립 전부 빈 detections로 돌았고, 그동안 부품주입이 통째로 꺼져 있었다).
        # noop은 사용자가 명시적으로 '검출 없이 가겠다'고 선택한 것이므로 예외.
        if not flat_dets and self.cfg.detector.impl != "noop":
            raise RuntimeError(
                f"YOLO 검출 0건: {video_id} (프레임 {len(frame_paths)}개). "
                "이 영상들에서 정상 실행이면 hand 등이 반드시 잡힌다 — 가중치 경로/디바이스/"
                "환경(GPU 노드 패키지)을 의심할 것. 정말 검출 없이 진행하려면 detector.impl=noop."
            )

        print(f"[STEP4][2/2] VLM 관찰: {video_id}")
        injected = self._injected_parts(video_id, flat_dets)
        # videos_map으로 부품을 손지정한 영상은 그 목록을 전 구간 고정 주입.
        # 그 외에는 YOLO 검출을 넘겨서 VLM 어댑터가 청크(시간구간)별로 실제 검출된
        # 클래스만 주입하게 한다(구간에 안 나오는 부품을 주입하면 오히려 환각 유도).
        in_map = video_id in getattr(self.vlm, "_videos_map", {})
        actions = self.vlm.observe_frames(
            frame_paths, times, injected_parts=injected,
            detections=None if in_map else flat_dets)
        artifacts.save_observations(self.observations_dir, video_id, actions)
        if hasattr(self.vlm, "unload"):
            self.vlm.unload()

        print(f"      검출 {len(flat_dets)}건, 관찰 {len(actions)}건")
