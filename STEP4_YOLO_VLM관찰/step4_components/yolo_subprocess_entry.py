"""
YOLO 검출 전용 서브프로세스 진입점.

step4_runner.py가 이 스크립트를 subprocess로 띄운다 — ultralytics가 select_device() 안에서
os.environ["CUDA_VISIBLE_DEVICES"]를 무조건 덮어쓰는 부작용(실측 확인: 2026-07-04,
ultralytics/utils/torch_utils.py:222, `.predict()` 최초 호출 시점에 발생, YOLO(weights_path)
생성자나 _load()에서는 안 터짐) 때문이다. 이게 메인 프로세스에서 그대로 일어나면, 뒤이어 VLM이
device_map="auto"로 GPU 2장을 쓰려 해도 YOLO가 이미 좁혀놓은 1장에 갇힌다(2026-07-04 실측:
CLIP1 99프레임에서 21.97GiB 단일 GPU OOM).

자식 프로세스는 부모 os.environ의 **복사본**을 받는다 — 여기서 CUDA_VISIBLE_DEVICES를 아무리
덮어써도 프로세스 종료와 함께 사라지고 부모(=이후 VLM을 로드할 프로세스)는 전혀 영향 안 받는다.
그래서 코드를 고치는 게 아니라 프로세스를 분리하는 것 자체가 근본 해결책이다.

산출물은 STEP4→STEP5 핸드오프용으로 이미 있는 output/detections/<video_id>.json 그대로 —
서브프로세스가 최종 파일에 바로 쓰므로 별도 임시 포맷이 필요 없다.

실패 정책(2026-07-05 변경): 검출 중 예외가 나면 **빈 파일을 쓰지 않고 exit!=0으로 죽는다.**
부모(step4_runner)가 returncode를 보고 즉시 RuntimeError를 낸다. "실패했지만 빈 결과로 계속"은
환경 문제(패키지 누락 등)를 '검출 0건'과 구분 불가능한 산출물로 둔갑시켰던 전력이 있다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent.parent  # project-ai 루트
sys.path.insert(0, str(ROOT / "STEP4_YOLO_VLM관찰"))
sys.path.insert(0, str(ROOT))

from tacit_common import artifacts
from tacit_common.config import PipelineConfig
from tacit_common.interfaces.detector import FrameRef
from tacit_common.schema.intermediate import Detection, FrameMeta
from step4_registry import build_detector


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO 검출 전용 서브프로세스(CUDA_VISIBLE_DEVICES 오염 격리용)"
    )
    ap.add_argument("--config", required=True, help="config.yaml 경로")
    ap.add_argument("--video-id", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    frames_dir = cfg.paths.resolve(cfg.paths.step3_frames_dir)
    detections_dir = cfg.paths.resolve(cfg.paths.step4_detections_dir)

    fmeta = artifacts.load_frames_meta(frames_dir, args.video_id)
    frame_paths, times, fps = fmeta["frame_paths"], fmeta["times"], fmeta["fps"]
    meta = FrameMeta(video_id=args.video_id, path=args.video_id, fps=fps, n_frames=len(frame_paths))

    detector = build_detector(cfg.detector)
    frame_refs = [FrameRef(frame_idx=i, timestamp=times[i], image=p)
                  for i, p in enumerate(frame_paths)]
    # 예외를 여기서 삼키지 않는다(2026-07-05). 예전엔 실패해도 빈 결과로 계속 진행했는데,
    # 그 결과 'No module named pyparsing' 같은 환경 문제가 detections=[] 빈 파일로 둔갑해
    # 3일간 4클립 전부 부품주입이 꺼진 채 돌았다(산출물만 봐서는 "검출 0건"과 구분 불가).
    # 실패는 traceback+exit!=0으로 부모(step4_runner)까지 그대로 올린다 — CLAUDE.md 원칙 5.
    try:
        detections_by_frame = detector.detect(frame_refs, meta)
        flat_dets: List[Detection] = [d for fd in detections_by_frame for d in fd.detections]
    finally:
        if hasattr(detector, "unload"):
            detector.unload()

    path = artifacts.save_detections(detections_dir, args.video_id, flat_dets)
    n_frames_hit = len({d.frame_idx for d in flat_dets})
    print(f"[OK] YOLO 서브프로세스 완료: {args.video_id} 검출 {len(flat_dets)}건 "
          f"(프레임 {n_frames_hit}/{len(frame_paths)}개에서) → {path}")


if __name__ == "__main__":
    main()
