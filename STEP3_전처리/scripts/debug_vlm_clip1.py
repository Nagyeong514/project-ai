"""
CLIP1(롱클립, 198초, fps_long=0.15 분기)에서 VLM이 실제로 뭘 뱉는지 raw 텍스트를 직접 찍어본다.
STT/YOLO/LLM 없이 VLM 단계만 실행 — 원인 진단용 임시 스크립트.
"""
from __future__ import annotations

import sys
from pathlib import Path

STEP3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STEP3_ROOT))

from tacit_pipeline.config import PipelineConfig
from tacit_pipeline.registry import build_vlm
from tacit_pipeline.components.frame_extract import extract_frames, probe_duration

VIDEO = "/home/ai_user/team_a2/members/안나경/master/master_videos/CLIP1_정상조립과정.mp4"


def main() -> None:
    cfg = PipelineConfig.load(str(STEP3_ROOT / "config.yaml"))
    vlm = build_vlm(cfg.vlm)

    dur = probe_duration(VIDEO, vlm.ffmpeg_bin)
    fps = vlm.fps_for_duration(dur)
    print(f"[DEBUG] duration={dur:.1f}s fps={fps} threshold={vlm.long_video_threshold_sec}")

    frame_paths, times = extract_frames(VIDEO, fps, vlm.frames_dir,
                                         long_side=vlm.long_side, ffmpeg_bin=vlm.ffmpeg_bin)
    print(f"[DEBUG] n_frames={len(frame_paths)} first_times={times[:5]} last_times={times[-5:]}")

    vlm._load()
    from PIL import Image
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    from tacit_pipeline.prompts.vlm_observation import build_video_observation_messages
    from tacit_pipeline.schema.intermediate import seconds_to_hhmmss
    base = build_video_observation_messages(None)
    system_text, user_text = base[0]["content"], base[1]["content"]
    user_text += (f"\n\n영상 길이는 약 {seconds_to_hhmmss(times[-1]) if times else '알 수 없음'}이다. "
                  "timestamp는 영상 시작(00:00:00) 기준 대략적인 초 단위로 적어라.")
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [{"type": "video", "video": frames},
                                     {"type": "text", "text": user_text}]},
    ]
    raw = vlm._generate_mm(messages)
    print("=" * 80)
    print("[RAW OUTPUT]")
    print(raw)
    print("=" * 80)
    parsed = vlm._parse_observations(raw)
    print(f"[PARSED] {len(parsed)}건")


if __name__ == "__main__":
    main()
