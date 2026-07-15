"""
하위 클립(CLIPn_..._clipNN) 산출물을 CLIP 그룹당 하나의 JSON으로 병합 (2026-07-07).

run_step4_motion.py는 클립 파일 하나당 산출물 하나를 만든다. 이 스크립트는 같은 그룹의
하위 클립들을 clip 번호 순서로 이어붙여, 그룹을 "하나의 가상 영상"으로 보는 산출물을
추가 생성한다(하위 클립별 산출물은 그대로 둠 — 병합본은 별도 폴더).

시간축: 하위 클립들은 원본 영상의 서로 다른 지점에서 잘려 나왔지만 원본 기준 오프셋
정보가 없으므로, "클립들을 순서대로 이어붙인 가상 영상" 기준으로 재부여한다 —
clip01의 duration(frames_meta.json)만큼 clip02의 모든 시각을 밀고, 그 다음도 누적.
frame_idx도 같은 방식으로 누적 프레임 수만큼 민다. 형식은 원본 STEP4와 동일.

사용:
    python scripts/merge_by_group.py            # output_motion/*_by_group/ 에 그룹당 1파일 생성
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# STEP4_YOLO_VLM관찰 폴더(scripts/의 부모). 모션 산출물은 그 아래 output_motion/ 에 격리.
MO_ROOT = Path(__file__).resolve().parent.parent
_CLIP_SUFFIX = re.compile(r"_clip(\d+)(?=\.mp4$)", re.IGNORECASE)


def group_of(video_id: str):
    """'CLIP1_정상조립과정_clip01.mp4' → ('CLIP1_정상조립과정.mp4', 1). 미일치면 None."""
    m = _CLIP_SUFFIX.search(video_id)
    if not m:
        return None
    return _CLIP_SUFFIX.sub("", video_id), int(m.group(1))


def _offset_chunk(chunk: str | None, off: float) -> str | None:
    if not chunk:
        return chunk
    try:
        a, b = chunk.split("-")
        return f"{float(a) + off:.0f}-{float(b) + off:.0f}"
    except ValueError:
        return chunk


def main() -> None:
    frames_dir = MO_ROOT / "output_motion" / "_frames"
    obs_dir = MO_ROOT / "output_motion" / "vlm_observations"
    det_dir = MO_ROOT / "output_motion" / "detections"
    obs_out = MO_ROOT / "output_motion" / "vlm_observations_by_group"
    det_out = MO_ROOT / "output_motion" / "detections_by_group"
    obs_out.mkdir(parents=True, exist_ok=True)
    det_out.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for p in sorted(obs_dir.glob("*.observations.json")):
        video_id = p.name[: -len(".observations.json")]
        g = group_of(video_id)
        if g is None:
            print(f"[WARN] 클립 패턴 미일치, 건너뜀: {video_id}")
            continue
        groups[g[0]].append((g[1], video_id))

    if not groups:
        sys.exit("병합할 산출물이 없음 — run_step4_motion.py 를 먼저 돌릴 것")

    for gid in sorted(groups):
        subs = sorted(groups[gid])  # clip 번호 순
        merged_obs, merged_raw, merged_dets = [], {}, []
        t_off, f_off = 0.0, 0
        for num, video_id in subs:
            fmeta = json.load(open(frames_dir / video_id / "frames_meta.json", encoding="utf-8"))
            obs = json.load(open(obs_dir / f"{video_id}.observations.json", encoding="utf-8"))
            for o in obs["observations"]:
                o = dict(o)
                o["timestamp"] = o["timestamp"] + t_off
                if o.get("end_timestamp") is not None:
                    o["end_timestamp"] = o["end_timestamp"] + t_off
                o["chunk"] = _offset_chunk(o.get("chunk"), t_off)
                merged_obs.append(o)
            for k, v in (obs.get("raw_by_chunk") or {}).items():
                merged_raw[_offset_chunk(k, t_off) or k] = v
            det_path = det_dir / f"{video_id}.json"
            if det_path.exists():
                for d in json.load(open(det_path, encoding="utf-8"))["detections"]:
                    d = dict(d)
                    d["timestamp"] = d["timestamp"] + t_off
                    d["frame_idx"] = d["frame_idx"] + f_off
                    merged_dets.append(d)
            t_off += fmeta["duration"]
            f_off += len(fmeta["frame_paths"])

        obs_payload = {"video_id": gid, "n_observations": len(merged_obs),
                       "observations": merged_obs}
        if merged_raw:
            obs_payload["raw_by_chunk"] = merged_raw
        with open(obs_out / f"{gid}.observations.json", "w", encoding="utf-8") as f:
            json.dump(obs_payload, f, ensure_ascii=False, indent=2)
        with open(det_out / f"{gid}.json", "w", encoding="utf-8") as f:
            json.dump({"video_id": gid, "detections": merged_dets}, f,
                      ensure_ascii=False, indent=2)
        print(f"[OK] {gid}: 하위클립 {len(subs)}개 → 관찰 {len(merged_obs)}건, "
              f"검출 {len(merged_dets)}건 (가상 길이 {t_off:.0f}s, {f_off}프레임)")

    print(f"[OK] 병합 완료 → {obs_out}, {det_out}")


if __name__ == "__main__":
    main()
