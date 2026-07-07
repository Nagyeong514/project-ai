"""
EXP_motion_guided_vs_uniform — 모션가이드 샘플링 클립을 기존 YOLO→VLM 파이프라인에 통과시켜
클립별 행동(action) 리스트 JSON을 만든다. 균일 프레임 추출(대조군)과의 비교용 실험군 데이터.

기존 코드 재사용(수정 없음, sys.path import):
  - STEP3_전처리/step3_components/frame_extract.py  → ffmpeg 프레임 추출
  - STEP4_YOLO_VLM관찰/step4_runner.Step4Runner     → YOLO(서브프로세스)+VLM 관찰
  - tacit_common/artifacts                          → 중간 산출물 입출력

이 스크립트가 새로 하는 일은 딱 세 가지:
  1. 원본 긴 영상 대신 motion_guieded_sampling/ 의 짧은 클립들을 순회
  2. 클립당 최소 MIN_FRAMES 장 보장(짧은 클립에서 프레임이 거의 안 뽑히는 것 방지)
  3. 관찰 산출물을 "CLIP그룹 → clipNN → 행동 문자열 리스트"로 단순화해 통합 JSON 저장

실행(GPU 노드에서, 파이프라인_통합실행/.venv 사용):
    python run_experiment.py --limit 1          # 첫 클립 1개 스모크 테스트
    python run_experiment.py                    # 전체(이미 관찰 산출물이 있는 클립은 skip)
    python run_experiment.py --no-skip-existing # 전부 재실행
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
PROJECT_AI = Path("/home/ai_user/team_a2/members/안나경/project-ai")
for _p in ["STEP3_전처리", "STEP4_YOLO_VLM관찰"]:
    sys.path.insert(0, str(PROJECT_AI / _p))
sys.path.insert(0, str(PROJECT_AI))  # tacit_common

# 프리플라이트가 가장 먼저(무거운 모델 로드 위로는 5초 안에 죽는 검증만 — CLAUDE.md 원칙 1)
from step4_preflight import run_preflight

DEFAULT_CLIPS_DIR = "/home/ai_user/team_a2/members/안나경/master/motion_guieded_sampling"
MIN_FRAMES = 4  # 클립당 최소 프레임 장수(이보다 적게 뽑히면 fps를 올려 재추출)

_GROUP_RE = re.compile(r"^(CLIP\d+)")          # 파일명 앞부분 → 원본 영상 그룹
_PART_RE = re.compile(r"_(clip\d+)\.mp4$", re.IGNORECASE)  # 파일명 끝부분 → 클립 번호


def parse_clip_name(video_id: str) -> tuple[str, str]:
    """'CLIP1_정상조립과정_clip01.mp4' → ('CLIP1', 'clip01'). 패턴 불일치면 파일명 그대로."""
    g = _GROUP_RE.match(video_id)
    p = _PART_RE.search(video_id)
    group = g.group(1) if g else Path(video_id).stem
    part = p.group(1).lower() if p else Path(video_id).stem
    return group, part


def extract_clip_frames(cfg, video_path: str, video_id: str, frames_dir: str) -> int:
    """STEP3의 ffmpeg 추출을 그대로 사용하되 최소 장수를 보장. 추출된 프레임 수 반환."""
    from step3_components.frame_extract import extract_frames, probe_duration
    from tacit_common import artifacts

    fe = cfg.frame_extraction
    dur = probe_duration(video_path, fe.ffmpeg_bin)
    if fe.fps_override is not None:
        fps = fe.fps_override
    else:
        fps = fe.fps_long if dur >= fe.long_video_threshold_sec else fe.fps_short

    out_dir = str(Path(frames_dir) / video_id)
    paths, times = extract_frames(video_path, fps, out_dir,
                                  long_side=fe.long_side, ffmpeg_bin=fe.ffmpeg_bin)
    if len(paths) < MIN_FRAMES and dur > 0:
        fps = MIN_FRAMES / dur
        print(f"      프레임 {len(paths)}장 < 최소 {MIN_FRAMES}장 → fps={fps:.3f}로 재추출")
        paths, times = extract_frames(video_path, fps, out_dir,
                                      long_side=fe.long_side, ffmpeg_bin=fe.ffmpeg_bin)

    artifacts.save_frames_meta(frames_dir, video_id, paths, times, fps, dur)
    print(f"      duration={dur:.1f}s fps={fps} → {len(paths)}프레임 → {out_dir}")
    return len(paths)


def rebuild_actions_json(observations_dir: str, clip_names: list[str], out_path: Path) -> dict:
    """지금까지 관찰 산출물이 있는 클립 전부를 모아 단순화 JSON을 다시 쓴다(클립마다 갱신 —
    중간에 죽어도 완료분은 남게).

    형식: {"CLIP1": {"clip01": ["행동1", "행동2", ...], ...}, ...}
    타임스탬프/프레임번호 등 메타데이터는 전부 버리고 action 문자열만 남긴다
    (VLM 어댑터가 dedup을 이미 해두므로 여기선 순서 그대로 나열만 한다).
    """
    from tacit_common import artifacts

    result: dict[str, dict[str, list[str]]] = {}
    for video_id in clip_names:
        if not artifacts.observations_path(observations_dir, video_id).exists():
            continue
        obs = artifacts.load_observations(observations_dir, video_id)
        group, part = parse_clip_name(video_id)
        actions = [a.action for a in obs if a.action.strip()]
        result.setdefault(group, {})[part] = actions

    ordered = {g: dict(sorted(result[g].items())) for g in sorted(result)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser(description="모션가이드 클립 → YOLO+VLM → 행동 리스트 JSON")
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만(스모크 테스트용)")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="관찰 산출물이 이미 있어도 다시 돌린다(기본: 있으면 skip)")
    args = ap.parse_args()

    config_path = str(Path(args.config).resolve())  # YOLO 서브프로세스에 절대경로로 넘어가야 함
    cfg = run_preflight(config_path)

    clips = sorted(Path(args.clips_dir).glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"클립 없음: {args.clips_dir}")
    if args.limit:
        clips = clips[: args.limit]
    print(f"[EXP] 대상 클립 {len(clips)}개 (dir={args.clips_dir})")
    print(f"[EXP] python: {sys.executable}")

    from tacit_common import artifacts
    from step4_runner import Step4Runner

    frames_dir = cfg.paths.resolve(cfg.paths.step3_frames_dir)
    observations_dir = cfg.paths.resolve(cfg.paths.step4_observations_dir)
    actions_json_path = EXP_DIR / "output" / "actions_by_clip.json"
    summary_path = EXP_DIR / "output" / "run_summary.json"

    runner = Step4Runner(cfg, config_path=config_path)

    clip_names = [c.name for c in clips]
    summary: dict[str, dict] = {}
    total_frames = 0
    failed: list[str] = []

    for i, clip in enumerate(clips, 1):
        video_id = clip.name
        print(f"\n{'=' * 80}\n[EXP {i}/{len(clips)}] {video_id}\n{'=' * 80}")

        obs_exists = artifacts.observations_path(observations_dir, video_id).exists()
        if obs_exists and not args.no_skip_existing:
            fmeta = artifacts.load_frames_meta(frames_dir, video_id)
            n_frames = len(fmeta["frame_paths"])
            n_obs = len(artifacts.load_observations(observations_dir, video_id))
            total_frames += n_frames
            summary[video_id] = {"frames": n_frames, "observations": n_obs, "skipped": True}
            print(f"[SKIP] 관찰 산출물 이미 있음(프레임 {n_frames}, 관찰 {n_obs}건)")
            continue

        try:
            n_frames = extract_clip_frames(cfg, str(clip), video_id, frames_dir)
            total_frames += n_frames
            runner.run(video_id)
            n_obs = len(artifacts.load_observations(observations_dir, video_id))
            summary[video_id] = {"frames": n_frames, "observations": n_obs, "skipped": False}
            if n_obs == 0:
                print(f"[WARN] 관찰 0건: {video_id} (재시도 후에도 빈 청크였을 수 있음)")
        except Exception as e:
            print(f"[FAIL] {video_id}: {e}")
            traceback.print_exc()
            failed.append(video_id)
            summary[video_id] = {"frames": summary.get(video_id, {}).get("frames"),
                                 "error": str(e)}
            continue

        # 클립마다 통합 JSON 갱신(중간에 죽어도 완료분 보존)
        rebuild_actions_json(observations_dir, clip_names, actions_json_path)

    result = rebuild_actions_json(observations_dir, clip_names, actions_json_path)

    print(f"\n{'=' * 80}\n[EXP SUMMARY]\n{'=' * 80}")
    for vid, s in summary.items():
        print(f"  {vid}: {s}")
    print(f"\n[EXP] 처리한 총 프레임 수: {total_frames}  ← 대조군(균일 추출)과 비용 비교용")
    print(f"[EXP] 행동 JSON: {actions_json_path}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"total_frames": total_frames, "clips": summary}, f,
                  ensure_ascii=False, indent=2)

    # 스모크 검증(체크리스트 5: exit 0인데 결과가 텅 빈 게 최악의 실패) — 성공 클립이
    # 하나라도 있으면 통합 JSON에 행동이 실제로 담겨 있어야 한다.
    ok_clips = [v for v, s in summary.items() if not s.get("error")]
    if ok_clips:
        n_actions_total = sum(len(a) for g in result.values() for a in g.values())
        assert n_actions_total > 0, "성공 클립이 있는데 행동 리스트가 전부 비어 있음 — 파이프라인 점검 필요"

    if failed:
        print(f"[EXP] 실패 {len(failed)}건: {failed}")
        sys.exit(1)
    print("[EXP] 완료")


if __name__ == "__main__":
    main()
