"""
CLIP1의 60~190초 구간만 잘라서(이미 추출된 프레임 재사용) 독립된 짧은 영상처럼 VLM에 넣어본다.
목적: 60~190초의 반복 관찰이 VLM 아티팩트인지 실제 반복 동작인지 구분.
시각은 0초부터 시작하도록 재정규화(새 영상으로 보이게).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0, str(ROOT / "STEP3_전처리"))  # step3_components (혹시 필요시)
sys.path.insert(0, str(ROOT / "STEP4_YOLO_VLM관찰"))
sys.path.insert(0, str(ROOT))

from tacit_common.config import PipelineConfig
from step4_registry import build_vlm

CFG_PATH = str(ROOT / "파이프라인_통합실행" / "config.yaml")
FRAMES_META = str(ROOT / "STEP3_전처리/output/_frames/CLIP1_정상조립과정.mp4/frames_meta.json")


def main() -> None:
    with open(FRAMES_META, "r", encoding="utf-8") as f:
        d = json.load(f)
    frame_paths, times = d["frame_paths"], d["times"]

    sub = [(p, t) for p, t in zip(frame_paths, times) if 60.0 <= t <= 190.0]
    sub_paths = [p for p, _ in sub]
    sub_times = [t - 60.0 for _, t in sub]  # 0초부터 시작하도록 재정규화

    print(f"[SEGMENT] 프레임 {len(sub_paths)}개, 0~{sub_times[-1]:.1f}초(원본 60~190초)")

    cfg = PipelineConfig.load(CFG_PATH)
    vlm = build_vlm(cfg.vlm)
    actions = vlm.observe_frames(sub_paths, sub_times, injected_parts=None)

    print("=" * 80)
    print(f"[RESULT] 관찰 {len(actions)}건")
    for a in actions:
        print(f"  ts={a.timestamp:6.1f}s actor={a.actor!s:6} action={a.action}")
    print("=" * 80)


if __name__ == "__main__":
    main()
