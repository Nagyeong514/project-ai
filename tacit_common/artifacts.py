"""
STEP3→4→5 사이를 오가는 중간 산출물의 저장/로드를 한 곳에 고정한다.

단독 실행이든(각 STEP이 파일로 주고받음) 통합 실행이든(파이프라인_통합실행이 모델은
재사용하되 단계 사이는 항상 파일을 거치게 함, 두 실행경로 일치 원칙) 이 함수들만 거치면
저장 형식이 어긋날 일이 없다 — 한쪽 STEP만 형식을 바꾸고 다른 쪽이 못 읽는 사고 방지.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .schema.intermediate import ActionDescription, Detection, AlignedWindow, Transcript


# ── STEP3 → STEP4 : 프레임 메타(경로+시각) ─────────────────────────────────
def frames_meta_path(frames_dir: str, video_id: str) -> Path:
    return Path(frames_dir) / video_id / "frames_meta.json"


def save_frames_meta(
    frames_dir: str, video_id: str, frame_paths: List[str], times: List[float],
    fps: float, duration: float,
) -> str:
    path = frames_meta_path(frames_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id, "fps": fps, "duration": duration,
        "frame_paths": frame_paths, "times": times,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def load_frames_meta(frames_dir: str, video_id: str) -> Dict[str, Any]:
    path = frames_meta_path(frames_dir, video_id)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── STEP3 → STEP5 : transcript ──────────────────────────────────────────────
# STT 어댑터도 transcribe() 안에서 자체적으로 저장하지만, 그건 정제(정규화+반복감지) 전
# 원문 스냅샷이다. step3_runner가 refine() 이후 이 함수로 다시 저장해 파일에는 항상
# "정제까지 끝난" 버전이 남게 한다(STEP5는 파일만 읽으므로 정제 전 버전을 보면 안 됨).
def transcript_path(transcript_dir: str, video_id: str) -> Path:
    return Path(transcript_dir) / f"{video_id}.json"


def save_transcript(transcript_dir: str, transcript: Transcript) -> str:
    path = transcript_path(transcript_dir, transcript.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)
    return str(path)


def load_transcript(transcript_dir: str, video_id: str) -> Transcript:
    path = transcript_path(transcript_dir, video_id)
    with path.open("r", encoding="utf-8") as f:
        return Transcript.model_validate(json.load(f))


# ── STEP4 → STEP5 : YOLO 검출 ───────────────────────────────────────────────
def detections_path(detections_dir: str, video_id: str) -> Path:
    return Path(detections_dir) / f"{video_id}.json"


def save_detections(detections_dir: str, video_id: str, detections: List[Detection]) -> str:
    path = detections_path(detections_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video_id": video_id, "detections": [d.model_dump() for d in detections]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def load_detections(detections_dir: str, video_id: str) -> List[Detection]:
    path = detections_path(detections_dir, video_id)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Detection.model_validate(d) for d in raw["detections"]]


# ── STEP4 → STEP5 : VLM 관찰 로그 ───────────────────────────────────────────
def observations_path(observations_dir: str, video_id: str) -> Path:
    return Path(observations_dir) / f"{video_id}.observations.json"


def save_observations(observations_dir: str, video_id: str, actions: List[ActionDescription],
                      raw_by_chunk: Dict[str, str] | None = None) -> str:
    """raw_by_chunk: VLM 원출력을 청크당 1건만 파일 최상위에 보존(디버깅용).

    예전엔 ActionDescription.raw 로 관찰마다 청크 원문 전체를 복제 저장했는데(실측
    CLIP4: 관찰 40건 × 같은 청크 JSON 중복), 관찰 객체에서 빼고 여기로 옮겼다(2026-07-05).
    """
    path = observations_path(observations_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id, "n_observations": len(actions),
        "observations": [a.model_dump() for a in actions],
    }
    if raw_by_chunk:
        payload["raw_by_chunk"] = raw_by_chunk
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def load_observations(observations_dir: str, video_id: str) -> List[ActionDescription]:
    path = observations_path(observations_dir, video_id)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ActionDescription.model_validate(a) for a in raw["observations"]]


# ── STEP5 내부(정렬 결과, LLM 융합 입력) ────────────────────────────────────
def aligned_windows_path(windows_dir: str, video_id: str) -> Path:
    return Path(windows_dir) / f"{video_id}.windows.json"


def save_aligned_windows(windows_dir: str, video_id: str, windows: List[AlignedWindow]) -> str:
    path = aligned_windows_path(windows_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video_id": video_id, "windows": [w.model_dump() for w in windows]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def load_aligned_windows(windows_dir: str, video_id: str) -> List[AlignedWindow]:
    path = aligned_windows_path(windows_dir, video_id)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [AlignedWindow.model_validate(w) for w in raw["windows"]]
