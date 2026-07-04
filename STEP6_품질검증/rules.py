# -*- coding: utf-8 -*-
"""
[게이트 C - 탈락조건] timestamp_validity
[게이트 C - 점수항목] utterance_signal 의 규칙기반(마커 탐지) 부분

문서 6-2 설계원칙 4: action_only(발화 없는) 스텝은 발화를 전제하는 검사에서 면제한다.
"""
import re
from typing import List, Tuple, Dict, Any, Optional

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


def parse_timestamp(ts: str) -> Optional[int]:
    """'HH:MM:SS' -> 총 초. 형식이 잘못되면 None."""
    if not ts:
        return None
    m = TIME_RE.match(ts.strip())
    if not m:
        return None
    h, mi, s = map(int, m.groups())
    return h * 3600 + mi * 60 + s


def find_transcript_segment(ts_sec: int, transcript: List[dict], window: int = 5) -> Optional[dict]:
    """timestamp 초 값에 해당하는 transcript 세그먼트를 찾는다."""
    best = None
    best_dist = None
    for seg in transcript:
        seg_start = parse_timestamp(seg.get("timestamp", ""))
        if seg_start is None:
            continue
        seg_end_raw = seg.get("end")
        seg_end = parse_timestamp(seg_end_raw) if seg_end_raw else seg_start + window
        if seg_start <= ts_sec <= seg_end:
            return seg
        dist = min(abs(ts_sec - seg_start), abs(ts_sec - seg_end))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = seg
    # 정확히 창 안에 든 세그먼트가 없으면, window 초 이내로 가장 가까운 것을 fallback 으로 반환
    if best is not None and best_dist is not None and best_dist <= window:
        return best
    return None


def text_roughly_matches(a: str, b: str) -> bool:
    """공백 제거 후 부분일치 여부로 '실재하는가'를 판단 (완전 재생성 방지를 위해 느슨하게)."""
    if not a or not b:
        return False
    na = re.sub(r"\s+", "", a)
    nb = re.sub(r"\s+", "", b)
    return na in nb or nb in na


def check_timestamp_validity(candidate: dict, transcript: List[dict]) -> Tuple[bool, Dict[str, Any]]:
    """
    [탈락조건] 모든 source가 clip_start~clip_end 내에 있고 transcript 에 실재하는가.
    evidence == "utterance" 인 항목에만 적용, action_only는 면제 (설계원칙 4).
    """
    details: Dict[str, Any] = {"checks": [], "passed": True}

    src = candidate["metadata"]["source"]
    clip_start = parse_timestamp(src["clip_start"])
    clip_end = parse_timestamp(src["clip_end"])
    if clip_start is None or clip_end is None:
        details["passed"] = False
        details["checks"].append({
            "item": "clip_range_format", "ok": False,
            "note": f"clip_start/clip_end 형식 오류: {src.get('clip_start')} ~ {src.get('clip_end')}"
        })
        return False, details

    def _check_one(label: str, ts: str, expected_text: Optional[str], require_text_match: bool):
        ts_sec = parse_timestamp(ts) if ts else None
        entry = {"item": label, "timestamp": ts, "ok": True, "note": ""}
        if ts_sec is None:
            entry["ok"] = False
            entry["note"] = "timestamp 형식 오류 또는 누락"
            details["checks"].append(entry)
            details["passed"] = False
            return
        if not (clip_start <= ts_sec <= clip_end):
            entry["ok"] = False
            entry["note"] = f"clip 범위({src['clip_start']}~{src['clip_end']}) 밖"
            details["checks"].append(entry)
            details["passed"] = False
            return
        if require_text_match:
            if not transcript:
                entry["ok"] = False
                entry["note"] = "transcript 파일이 없어 실재 여부 확인 불가"
                details["checks"].append(entry)
                details["passed"] = False
                return
            seg = find_transcript_segment(ts_sec, transcript)
            if seg is None:
                entry["ok"] = False
                entry["note"] = "해당 timestamp 근처 transcript 세그먼트를 찾을 수 없음"
                details["checks"].append(entry)
                details["passed"] = False
                return
            if expected_text and not text_roughly_matches(expected_text, seg["text"]):
                entry["ok"] = False
                entry["note"] = f"transcript 원문과 불일치. transcript='{seg['text']}'"
                details["checks"].append(entry)
                details["passed"] = False
                return
        entry["note"] = "OK"
        details["checks"].append(entry)

    know = candidate["knowledge"]

    # situation_source: 발화 존재를 전제 (텍스트 원문 대조는 situation 문장이 요약이라 생략, 범위/실재만 확인)
    for ts in know.get("situation_source", []):
        _check_one("situation_source", ts, None, require_text_match=True)

    # reasoning: origin이 utterance일 때만 실재 대조
    for ts in know.get("reasoning_source", []):
        require_match = know.get("reasoning_origin") == "utterance"
        _check_one("reasoning_source", ts, None, require_text_match=require_match)

    # diagnostic_steps: evidence == "utterance" 인 것만 검사 (action_only는 면제)
    for step in know.get("diagnostic_steps", []):
        if step.get("evidence") == "utterance":
            _check_one(
                f"diagnostic_step[{step.get('order')}]",
                step.get("timestamp"),
                step.get("source_utterance"),
                require_text_match=True,
            )
        else:
            details["checks"].append({
                "item": f"diagnostic_step[{step.get('order')}]",
                "timestamp": step.get("timestamp"),
                "ok": True,
                "note": "action_only -> 발화 실재 검사 면제 (설계원칙 4)",
            })

    return details["passed"], details


def utterance_signal_rule_score(candidate: dict, config) -> Tuple[float, Dict[str, Any]]:
    """
    utterance_signal 의 규칙기반 부분: 근거 발화에 인과/이탈/주의/부정 마커가 실제로 포함된 비율.
    action_only 스텝 비율만큼은 애초에 발화가 없으므로 분모에서 제외하고,
    '발화가 있는 스텝 대비 마커 포함 비율'로 측정 (문서 6-3: action_only 비율 대비 signal로 측정).
    """
    steps = candidate["knowledge"].get("diagnostic_steps", [])
    utterance_steps = [s for s in steps if s.get("evidence") == "utterance" and s.get("source_utterance")]
    action_only_steps = [s for s in steps if s.get("evidence") == "action_only"]

    all_markers = (
        config.causal_markers + config.deviation_markers + config.caution_markers + config.negation_markers
    )

    hit_count = 0
    per_step = []
    for s in utterance_steps:
        text = s["source_utterance"]
        hits = [m for m in all_markers if m.replace(" ", "") in text.replace(" ", "")]
        if hits:
            hit_count += 1
        per_step.append({"order": s.get("order"), "text": text, "matched_markers": hits})

    # 발화가 있는 스텝이 하나도 없다면(전부 action_only) 규칙기반 신호는 0으로 두고,
    # 이후 LLM judge 쪽에서 action_only 비율을 감안해 최종 점수를 보정한다.
    rule_score = hit_count / len(utterance_steps) if utterance_steps else 0.0

    detail = {
        "total_steps": len(steps),
        "utterance_steps": len(utterance_steps),
        "action_only_steps": len(action_only_steps),
        "marker_hit_steps": hit_count,
        "rule_score": rule_score,
        "per_step": per_step,
    }
    return rule_score, detail
