"""
프레임 소스 — "영상 하나 → (frame_paths, times, fps, duration, segments)"를 만드는
STEP3 registry 컴포넌트(스위치: `sampling.impl` = "uniform" | "motion").

**메모(2026-07-07):** `docs/전처리_파이프라인_계획서.md` §2-1의 "모션 샘플링 사용 안 함
(0.5fps 균등 추출)" 결정은 `00_사전연구/03_전처리_모션가이드_vs_균일추출` 비교 실험 실측으로
갱신됨 — 4클립 전부 모션가이드가 우세(행동 방향성 정확도, 환각·노이즈 적음)로 나와 팀이
모션가이드 채택 쪽으로 방향을 바꿨다. 균일추출은 폐기가 아니라 기준선(8/8)으로 보존하고,
이 파일이 그 기준선(UniformFrameSource)과 신규 채택안(MotionSampledFrameSource)을 같은
인터페이스로 나란히 제공한다. 계획서 문서 자체 갱신은 별도로 진행 예정(이 메모는 그 근거
포인터).

주의(혼동 방지): `STEP4_YOLO_VLM관찰/step4_components/sampler_motion.py`의
`MotionGuidedSampler`(레지스트리 등록만 되고 미사용)는 이것과 무관하다 — 그건 YOLO
bbox(hand/부품) 움직임 기반 "프레임 단위" 샘플러이고, 여기 `MotionSampledFrameSource`는
팀원이 이미 만들어둔 "영상 레벨 clip-window 샘플링" 산출물(CSV+서브클립 mp4, absdiff 기반)을
소비하는 어댑터다. 서로 다른 것이니 섞어서 고치지 말 것.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from step3_components.frame_extract import extract_frames, probe_duration


def select_fps(fe_cfg: Any, duration_sec: float) -> float:
    """FrameExtractionConfig + 영상 길이 → 사용할 fps. Step3Runner._fps_for_duration과 동일 로직
    (두 FrameSource 구현이 공유하도록 여기 모듈 함수로 옮김)."""
    if fe_cfg.fps_override is not None:
        return fe_cfg.fps_override
    return fe_cfg.fps_long if duration_sec >= fe_cfg.long_video_threshold_sec else fe_cfg.fps_short


class UniformFrameSource:
    """기준선(baseline). registry 키: 'uniform'. 현재 Step3Runner.run()의 동작을 그대로
    감싼 것 — 산출물 바이트 단위로 동일(순수 리팩터링, 동작 변경 없음)."""

    def __init__(self, **extra: Any):
        self.extra = extra

    def extract(self, video_path: str, video_id: str, frames_dir: str, fe_cfg: Any) -> Dict[str, Any]:
        dur = probe_duration(video_path, fe_cfg.ffmpeg_bin)
        fps = select_fps(fe_cfg, dur)
        out_dir = str(Path(frames_dir) / video_id)
        frame_paths, times = extract_frames(
            video_path, fps, out_dir, long_side=fe_cfg.long_side, ffmpeg_bin=fe_cfg.ffmpeg_bin,
        )
        return {
            "frame_paths": frame_paths, "times": times,
            "fps": fps, "duration": dur, "segments": None,
        }


class MotionSampledFrameSource:
    """모션가이드 샘플링 어댑터. registry 키: 'motion'.

    팀원이 이미 만들어둔 산출물(CSV + 서브클립 mp4, `01_코드/*.py`는 이번엔 import하지
    않음)만 읽는다. 오프셋 보정의 단일 진실 공급원은 CSV 컬럼(start_sec 등) — 파일명 파싱
    금지(CLIP2/clip_03 반올림 불일치 실측 사례).
    """

    def __init__(
        self,
        motion_root: str = "00_사전연구/04_모션가이드샘플링_프로토타입/04_샘플링 영상",
        dedup_window_sec: float = 0.5,
        regenerate: bool = False,
        # 2026-07-08(STEP3 정식 이식, 계획서: docs/계획서_모션가이드_STEP3_이식.md):
        # CSV+서브클립이 아예 없는 신규 영상에 대해서만 팀원 코드(motion_sampling
        # 패키지)를 호출해 자동 생성한다. 기존에 검증 끝난 4클립(CLIP1~4)은 이 값을
        # False로 둬서 팀원 산출물을 그대로 읽던 기존 동작이 그대로 유지되게 한다
        # (재생성 시 A/B/C 전략이 다시 뽑혀 지금까지의 검증이 무효화되는 것 방지).
        auto_generate: bool = False,
        **extra: Any,
    ):
        self.motion_root = motion_root
        self.dedup_window_sec = dedup_window_sec
        self.regenerate = regenerate
        self.auto_generate = auto_generate
        self.extra = extra

    def _resolved_motion_root(self) -> Path:
        p = Path(self.motion_root)
        if p.is_absolute():
            return p
        # tacit_common.config.PROJECT_ROOT(project-ai 루트) 기준 상대경로로 해석.
        from tacit_common.config import PROJECT_ROOT
        return PROJECT_ROOT / p

    def _read_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["clip_no"] = int(r["clip_no"])
            r["start_sec"] = float(r["start_sec"])
            r["end_sec"] = float(r["end_sec"])
            r["duration_sec"] = float(r["duration_sec"])
        rows.sort(key=lambda r: r["clip_no"])
        return rows

    @staticmethod
    def _build_segments(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        segments = []
        for i, r in enumerate(rows):
            overlaps_next = False
            if i + 1 < len(rows):
                overlaps_next = rows[i + 1]["start_sec"] < r["end_sec"]
            segments.append({
                "clip_no": r["clip_no"], "filename": r["filename"],
                "start_sec": r["start_sec"], "end_sec": r["end_sec"],
                "overlaps_next": overlaps_next,
            })
        return segments

    def extract(self, video_path: str, video_id: str, frames_dir: str, fe_cfg: Any) -> Dict[str, Any]:
        if self.regenerate:
            raise NotImplementedError(
                "MotionSampledFrameSource(regenerate=True) 미구현 — 이번 범위는 팀원이 이미 "
                "만들어둔 CSV+서브클립 산출물을 읽는 경로만 지원한다. 새 영상에 대해 처음부터 "
                "모션 스코어를 계산하려면 01_코드/motion_score.py 등을 호출하는 재샘플링 경로를 "
                "별도로 구현해야 한다(나중 과제)."
            )

        stem = Path(video_path).stem
        clip_dir = self._resolved_motion_root() / stem
        csv_path = clip_dir / "clip_timestamps.csv"
        if not csv_path.exists():
            if not self.auto_generate:
                raise FileNotFoundError(
                    f"모션가이드 샘플링 산출물 없음: {csv_path}\n"
                    f"  → video_id({video_id})에 대응하는 '{stem}' 폴더/CSV가 있는지 확인하세요.\n"
                    f"  → auto_generate=true로 config를 바꾸면 자동 생성됩니다."
                )
            # 2026-07-08 이식: 팀원 코드(motion_sampling 패키지)를 직접 호출해 서브클립을
            # 새로 만든다. run_abc_pipeline()이 이미 files_dir/clips_dir을 인자로 받게
            # 설계돼 있어 알고리즘은 그대로 두고 출력 위치만 여기(clip_dir)로 지정한다.
            from step3_components.motion_sampling import run_abc_pipeline
            print(f"[MOTION-SAMPLING] {video_id}: 서브클립 산출물 없음 → 자동 생성 시작")
            run_abc_pipeline(
                video_path=video_path,
                files_dir=str(clip_dir),
                clips_dir=str(clip_dir),
            )
        rows = self._read_csv(csv_path)

        video_frames_dir = Path(frames_dir) / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        # (corrected_time, clip_no, local_path) 전체 수집
        collected: List[Tuple[float, int, str]] = []
        tmp_dirs: List[Path] = []
        for r in rows:
            subclip_path = clip_dir / r["filename"]
            tmp_out = video_frames_dir / f"_sub_{r['clip_no']:02d}"
            tmp_dirs.append(tmp_out)
            fps = select_fps(fe_cfg, r["duration_sec"])
            local_paths, local_times = extract_frames(
                str(subclip_path), fps, str(tmp_out),
                long_side=fe_cfg.long_side, ffmpeg_bin=fe_cfg.ffmpeg_bin,
            )
            for p, t in zip(local_paths, local_times):
                collected.append((r["start_sec"] + t, r["clip_no"], p))

        segments = self._build_segments(rows)

        # 중복 제거: 시간근접(±dedup_window_sec) 기준이 아니라 **구간 중첩** 기준.
        # (2026-07-07 수정) 처음엔 "정렬 후 인접 프레임 시간차 <= dedup_window_sec면 스킵"으로
        # 짰는데, 실측 CLIP4에서 clip_01[0,37]/clip_02[19,57] 18초 중첩 구간의 0.5fps 그리드가
        # 홀/짝으로 어긋나 있어(0,2,4,... vs 19,21,23,...) 인접 프레임 간격이 항상 1초 이상 —
        # 시간근접 기준으로는 사실상 하나도 안 걸러짐(98/99 프레임 실측). 대신 CSV의
        # overlaps_next로 이미 확정된 "이전 세그먼트가 이 시각대를 이미 커버하는가"를 직접
        # 판정: 각 clip_no 프레임 중 시각이 '바로 이전 clip_no의 end_sec + dedup_window_sec'
        # 이하면 버린다(이전 서브클립이 먼저 그 구간을 담당하므로). 이러면 그리드 정렬과
        # 무관하게 중첩 구간은 항상 앞쪽 서브클립 프레임만 남고, 서로 안 겹치지만 맞닿은
        # 세그먼트(예: CLIP4 clip04/05, end=start=153.0)의 경계 중복 1프레임도 그대로 잡힌다.
        prev_end_by_clip_no: Dict[int, float] = {
            segments[i]["clip_no"]: segments[i - 1]["end_sec"] for i in range(1, len(segments))
        }
        collected.sort(key=lambda x: (x[0], x[1]))
        kept: List[Tuple[float, int, str]] = []
        for t, clip_no, p in collected:
            pe = prev_end_by_clip_no.get(clip_no)
            if pe is not None and t <= pe + self.dedup_window_sec:
                continue  # 이전 서브클립이 이미 이 시각대를 커버함 — 이 프레임은 버림
            kept.append((t, clip_no, p))

        # 채택된 프레임을 원본 시각순으로 frame_%04d.jpg 이름 재부여해 최종 위치로 복사
        frame_paths: List[str] = []
        times: List[float] = []
        for i, (t, _clip_no, src_path) in enumerate(kept, start=1):
            dst_path = video_frames_dir / f"frame_{i:04d}.jpg"
            shutil.copyfile(src_path, dst_path)
            frame_paths.append(str(dst_path))
            times.append(t)

        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)

        # 검증(a)의 자동화 버전 — 조용한 오프셋 계산 실수 즉시 탐지
        for t in times:
            if not any(seg["start_sec"] - 1e-6 <= t <= seg["end_sec"] + 1e-6 for seg in segments):
                raise AssertionError(
                    f"오프셋 보정 오류: 시각 {t:.2f}s가 어떤 세그먼트 범위에도 안 들어감 "
                    f"(video_id={video_id}, segments={segments})"
                )

        duration = probe_duration(video_path, fe_cfg.ffmpeg_bin)
        fps_used = select_fps(fe_cfg, rows[0]["duration_sec"]) if rows else fe_cfg.fps_short

        return {
            "frame_paths": frame_paths, "times": times,
            "fps": fps_used, "duration": duration, "segments": segments,
        }
