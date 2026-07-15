"""
STEP4 모션가이드 클립용 실행 엔트리포인트.
(2026-07-07 원본 STEP4의 복제본으로 시작 → 2026-07-13 STEP4_YOLO_VLM관찰_모션O 폴더를
 원본 STEP4로 병합하며 run_step4_motion.py 로 이관. 이제 원본과 동일한 step4_runner/
 step4_components(정본, 7/7 segment_bounds 포함)를 그대로 재사용한다.)

기본 run_step4.py(단일 --video-id 처리)와의 차이는 **입력뿐**이다:
  - run_step4.py: STEP3_전처리가 만들어둔 output/_frames/<video_id>/ 를 --video-id로 하나 처리
  - 이 스크립트: STEP3_전처리/output/motion_guided_clips/ 의 mp4 클립들을 전부 순회하며,
    클립마다 STEP3의 프레임 추출(step3_components.frame_extract — 코드 그대로 import)을 먼저
    돌리고 이어서 동일한 Step4Runner.run()을 호출. 프레임에 segments 키가 없으므로
    step4_runner의 hard_breaks 는 None(기존 동작과 완전 동일).

산출물은 config_motion.yaml 의 paths 대로 output_motion/ 아래로 격리 저장된다:
  output_motion/detections/<클립파일명>.json
  output_motion/vlm_observations/<클립파일명>.observations.json

사용:
    python run_step4_motion.py                    # 클립 15개 전부(관찰 산출물 있으면 skip)
    python run_step4_motion.py --limit 1          # 첫 클립 1개만(테스트)
    python run_step4_motion.py --no-skip-existing # 전부 재실행
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "STEP3_전처리"))  # frame_extract 재사용

import argparse
import traceback

from step4_preflight import run_preflight
from step4_runner import Step4Runner

CLIPS_DIR = "/home/ai_user/team_a2/members/안나경/project-ai/STEP3_전처리/output/motion_guided_clips"


def _extract_frames_for_clip(cfg, video_path: str, video_id: str, frames_dir: str) -> int:
    """STEP3의 프레임 추출을 클립 1개에 그대로 적용(step3_runner.run()의 프레임 부분과 동일 로직)."""
    from step3_components.frame_extract import extract_frames, probe_duration
    from tacit_common import artifacts

    fe = cfg.frame_extraction
    dur = probe_duration(video_path, fe.ffmpeg_bin)
    if fe.fps_override is not None:
        fps = fe.fps_override
    else:
        fps = fe.fps_long if dur >= fe.long_video_threshold_sec else fe.fps_short
    video_frames_dir = str(Path(frames_dir) / video_id)
    frame_paths, times = extract_frames(
        video_path, fps, video_frames_dir,
        long_side=fe.long_side, ffmpeg_bin=fe.ffmpeg_bin,
    )
    artifacts.save_frames_meta(frames_dir, video_id, frame_paths, times, fps, dur)
    print(f"      duration={dur:.0f}s fps={fps} → {len(frame_paths)}프레임 → {video_frames_dir}")
    return len(frame_paths)


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP4 모션O(모션가이드 클립 순회) 실행")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_motion.yaml"))
    ap.add_argument("--clips-dir", default=CLIPS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만(테스트용)")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="관찰 산출물이 이미 있어도 재실행(기본: 있으면 skip)")
    args = ap.parse_args()

    config_path = str(Path(args.config).resolve())  # YOLO 서브프로세스에 그대로 넘어감
    cfg = run_preflight(config_path)

    clips = sorted(Path(args.clips_dir).glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"클립 없음: {args.clips_dir}")
    if args.limit:
        clips = clips[: args.limit]
    print(f"[STEP4-모션O] 대상 클립 {len(clips)}개 (dir={args.clips_dir})")

    from tacit_common import artifacts

    frames_dir = cfg.paths.resolve(cfg.paths.step3_frames_dir)
    observations_dir = cfg.paths.resolve(cfg.paths.step4_observations_dir)
    runner = Step4Runner(cfg, config_path=config_path)

    total_frames = 0
    failed: list[str] = []
    for i, clip in enumerate(clips, 1):
        video_id = clip.name
        print(f"\n{'=' * 80}\n[모션O {i}/{len(clips)}] {video_id}\n{'=' * 80}")
        if (not args.no_skip_existing
                and artifacts.observations_path(observations_dir, video_id).exists()):
            n = len(artifacts.load_frames_meta(frames_dir, video_id)["frame_paths"])
            total_frames += n
            print(f"[SKIP] 관찰 산출물 이미 있음(프레임 {n}장)")
            continue
        try:
            total_frames += _extract_frames_for_clip(cfg, str(clip), video_id, frames_dir)
            runner.run(video_id)
            print(f"[OK] STEP4 완료: {video_id}")
        except Exception as e:
            print(f"[FAIL] {video_id}: {e}")
            traceback.print_exc()
            failed.append(video_id)

    print(f"\n[STEP4-모션O] 처리한 총 프레임 수: {total_frames}")
    if failed:
        print(f"[STEP4-모션O] 실패 {len(failed)}건: {failed}")
        sys.exit(1)
    print(f"[STEP4-모션O] 완료: {len(clips)}클립")


if __name__ == "__main__":
    main()
