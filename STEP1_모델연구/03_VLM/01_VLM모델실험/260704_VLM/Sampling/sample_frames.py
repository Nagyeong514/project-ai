"""
시나리오영상_0704 안 4개 클립에 대해 tacit_pipeline/components/frame_extract.py 와
동일한 방식(ffmpeg CLI, fps 균등 샘플링, 긴 변 480px 리사이즈)으로 프레임을 추출한다.

fps=0.5는 임의값이 아니라 vlm_qwen.py(QwenVLActionExtractor)의 fps_short/fps_long과
동일한 값이다 — VLM 관찰 0건 버그를 0.15→0.5로 올려 고친 이력이 있어(README 참고),
나중에 이 프레임을 그대로 VLM에 재사용하려면 같은 값을 써야 한다.

원본 파이프라인과의 차이점 딱 하나:
  frame_extract.extract_frames()는 out_dir을 영상마다 분리하지 않고 호출될 때마다
  그 폴더의 이전 frame_*.jpg를 지운다(원본은 "영상 1개 완전 처리 후 다음 영상"
  구조라 문제없음). 여기서는 4개 영상 결과를 동시에 보존해야 하므로,
  영상마다 별도의 out_dir(frames/<video_id>/)을 만들어 호출한다 — 그 외 로직은
  frame_extract.py 함수를 그대로 가져다 쓴다(재구현하지 않음).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

SAMPLING_DIR = Path(__file__).resolve().parent
MEMBER_DIR = SAMPLING_DIR.parent.parent  # .../오유빈
sys.path.insert(0, str(MEMBER_DIR))

from tacit_pipeline.components.frame_extract import extract_frames, probe_duration  # noqa: E402

VIDEO_DIR = MEMBER_DIR / "260704_VLM" / "scenario_videos_0704"
VIDEO_NAMES = [
    "CLIP1_normal_assembly.mp4",
    "CLIP2_0704.mp4",
    "CLIP3_before_reboot_0704.mp4",
    "CLIP4_reboot_bios.mp4",
]

# vlm_qwen.py QwenVLActionExtractor 기본값과 동일(현재 config: fps_short=fps_long=0.5)
FPS = 0.5
LONG_SIDE = 480

OUT_ROOT = SAMPLING_DIR / "frames"
MANIFEST_PATH = SAMPLING_DIR / "manifest.json"


def main() -> None:
    ffmpeg_bin = os.environ.get("FFMPEG_BIN")  # 이 서버엔 frame_extract.py의 기본 경로가 없어 override
    manifest: dict[str, dict] = {}

    for name in VIDEO_NAMES:
        video_path = VIDEO_DIR / name
        video_id = video_path.stem
        out_dir = OUT_ROOT / video_id

        dur = probe_duration(str(video_path), ffmpeg_bin=ffmpeg_bin)
        paths, times = extract_frames(
            str(video_path), FPS, str(out_dir), long_side=LONG_SIDE, ffmpeg_bin=ffmpeg_bin
        )

        # 실제 리사이즈된 프레임 크기(첫 장 기준 — 같은 영상 내 프레임은 전부 동일 크기).
        # bbox가 픽셀인지 정규화인지 헷갈리지 않도록 검출 결과 옆에 같이 남겨두기 위함.
        with Image.open(paths[0]) as im:
            frame_width, frame_height = im.size

        manifest[name] = {
            "video_id": video_id,
            "duration_sec": round(dur, 2),
            "fps": FPS,
            "long_side": LONG_SIDE,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "n_frames": len(paths),
            "frames": [
                {"path": str(Path(p).relative_to(SAMPLING_DIR)), "timestamp_sec": t}
                for p, t in zip(paths, times)
            ],
        }
        print(f"[OK] {name}: {dur:.1f}s -> {len(paths)}프레임 -> {out_dir}")

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] manifest 저장 -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
