"""
STEP5 -> STEP6 입력 형식 어댑터.

배경: STEP6_품질검증(팀원 코드)은 (1) "JSON 파일 하나 = 후보 하나" 구조와
(2) transcript가 {"segments":[{"timestamp":"HH:MM:SS","text":...}]} 형식이라고
전제하고 짜여 있다. 반면 STEP5 실제 출력은 (1) 영상 하나당 {"candidates":[...]}
배열이고 (2) STEP3 transcript는 {"utterances":[{"start":float,"raw_text":...}]}
형식이다. 이 두 스키마 차이를 STEP6 코드는 한 줄도 안 건드리고 우리 쪽에서
메꾸는 게 이 스크립트의 목적이다(frame_source.py의 UniformFrameSource/
MotionSampledFrameSource와 같은 "어댑터" 패턴).

★ 원본은 절대 건드리지 않는다 — STEP5 tacit_json, STEP3 transcript 둘 다 읽기만
  한다. 이 스크립트의 산출물은 전부 새 폴더(output/step6_adapter_input/)에만 쓴다.

★ 변환 검증이 핵심이다 — STEP6의 0단계(timestamp_validity)는 이 변환된
  transcript로 "후보가 인용한 발화가 실재하는가"를 판정한다. 변환 과정에서
  발화가 하나라도 빠지거나 시각이 어긋나면, 멀쩡한 후보가 "발화를 지어낸 것"으로
  오판정(reject)당한다. 그래서 변환 직후 원본과 건수/내용을 전수 대조하고,
  하나라도 안 맞으면 조용히 넘어가지 않고 바로 예외로 죽는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TACIT_DIR = PROJECT_ROOT / "STEP5_LLM으로_암묵지_후보생성" / "output" / "tacit_json"
TRANSCRIPT_DIR = PROJECT_ROOT / "STEP3_전처리" / "transcripts"


def seconds_to_hhmmss(sec: float) -> str:
    """STEP6 rules.py의 TIME_RE(`^\\d{1,2}:\\d{2}:\\d{2}$`, 정수초만)에 맞춰
    반올림한 "HH:MM:SS"로 변환. STEP5 자체 출력 timestamp도 이미 정수초
    포맷이라(예: "00:04:30") 같은 granularity로 맞추는 것뿐, 정밀도 손실 없음
    (원본 STT도 초 단위 발화 경계라 소수점 자리는 애초에 유의미하지 않음)."""
    total = round(sec)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def convert_transcript(transcript_path: Path) -> List[Dict[str, Any]]:
    """STEP3 {"utterances":[{"start","end","raw_text",...}]}
    -> STEP6 {"segments":[{"timestamp","end","text"}]} 로 변환.
    발화를 하나도 빠뜨리지 않고 순서 그대로 전부 옮긴다(hallucination 플래그
    포함 — STEP6은 그 필드를 모르므로 STEP6 입장에선 그냥 발화 원문 하나다)."""
    src = json.loads(transcript_path.read_text(encoding="utf-8"))
    utterances = src.get("utterances", [])
    segments = []
    for u in utterances:
        segments.append({
            "timestamp": seconds_to_hhmmss(u["start"]),
            "end": seconds_to_hhmmss(u["end"]),
            "text": u["raw_text"],
        })

    # ── 경계 겹침 제거(2026-07-08) ──────────────────────────────────
    # STT 발화는 빈틈없이 이어져서 반올림 후 앞 세그먼트 end == 뒤 세그먼트 start가 된다.
    # STEP6 rules.find_transcript_segment는 '경계 포함(≤) + 최초 매칭'이라, 뒤 발화의
    # 시작 시각을 조회하면 앞 세그먼트가 먼저 잡혀 원문 불일치로 false reject가 났다
    # (CLIP1 실측: 0단계 FAIL 4건 전부 이 오탐). STEP6 무수정 원칙에 따라 어댑터 쪽에서
    # end를 다음 start와 안 겹치게 깎는다. end는 어차피 STEP6이 구간 판정에만 쓰는 값이라
    # 발화 원문/시작 시각(검증 대상)은 그대로다.
    for i in range(len(segments) - 1):
        next_start = _hhmmss_to_seconds(segments[i + 1]["timestamp"])
        cur_start = _hhmmss_to_seconds(segments[i]["timestamp"])
        cur_end = _hhmmss_to_seconds(segments[i]["end"])
        if cur_end >= next_start:
            # 다음 시작 직전 초로 깎되, start보다 작아지지는 않게(0초 발화 방어).
            # 두 발화가 같은 초에 시작하는 극단 케이스는 겹침이 남지만, 그땐 어느 쪽이
            # 잡혀도 같은 초의 발화라 timestamp 자체는 유효 — 원문 불일치만 안 나면 된다.
            segments[i]["end"] = seconds_to_hhmmss(max(cur_start, next_start - 1))

    # ── 변환 검증(필수) ──────────────────────────────────────────────
    if len(segments) != len(utterances):
        raise AssertionError(
            f"transcript 변환 건수 불일치: 원본 {len(utterances)}건 -> 변환 {len(segments)}건 "
            f"({transcript_path})"
        )
    for i, (seg, u) in enumerate(zip(segments, utterances)):
        if seg["text"] != u["raw_text"]:
            raise AssertionError(
                f"transcript 변환 내용 불일치[{i}]: 원본={u['raw_text']!r} 변환={seg['text']!r} "
                f"({transcript_path})"
            )
        back_start = _hhmmss_to_seconds(seg["timestamp"])
        if back_start != round(u["start"]):
            raise AssertionError(
                f"transcript 변환 시각 불일치[{i}]: 원본 start={u['start']} "
                f"(반올림 {round(u['start'])}) vs 변환후 역산={back_start} ({transcript_path})"
            )
    return segments


def _hhmmss_to_seconds(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def split_candidates(tacit_path: Path, transcript_filename: str) -> List[Dict[str, Any]]:
    """STEP5 {"video_id","candidates":[...]} 파일 하나를 읽어, 후보마다
    STEP6이 기대하는 "파일 하나 = 후보 하나" 형태의 dict로 분리한다.
    각 후보의 metadata.source.transcript_ref만 어댑터 출력 폴더 기준
    상대경로로 덮어쓴다(원본 STEP5 파일 자체는 안 건드림 — 여긴 읽은 걸
    메모리에서 복사해 고치는 것뿐)."""
    doc = json.loads(tacit_path.read_text(encoding="utf-8"))
    out = []
    for cand in doc.get("candidates", []):
        cand = json.loads(json.dumps(cand))  # deep copy
        cand.setdefault("metadata", {}).setdefault("source", {})["transcript_ref"] = \
            f"transcripts/{transcript_filename}"
        out.append(cand)
    return out


def run(output_root: Path) -> Dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_root / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    stats = {"videos": 0, "candidates": 0, "transcript_utterances": 0}
    tacit_files = sorted(TACIT_DIR.glob("*.tacit.json"))
    if not tacit_files:
        raise FileNotFoundError(f"STEP5 tacit_json 산출물 없음: {TACIT_DIR}")

    for tacit_path in tacit_files:
        video_id = tacit_path.name.replace(".tacit.json", "")
        transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
        if not transcript_path.exists():
            raise FileNotFoundError(f"{video_id}에 대응하는 STEP3 transcript 없음: {transcript_path}")

        # 1) transcript 변환 + 검증
        segments = convert_transcript(transcript_path)
        seg_filename = f"{video_id}_segments.json"
        seg_out_path = transcripts_dir / seg_filename
        seg_out_path.write_text(
            json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 파일로 다시 읽어서 왕복 검증(쓰기 과정 자체의 손실도 배제)
        reloaded = json.loads(seg_out_path.read_text(encoding="utf-8"))["segments"]
        if reloaded != segments:
            raise AssertionError(f"transcript 파일 왕복(write→read) 불일치: {seg_out_path}")
        stats["transcript_utterances"] += len(segments)

        # 2) 후보 분리 + 개별 파일 저장 + 검증
        candidates = split_candidates(tacit_path, seg_filename)
        for cand in candidates:
            cand_id = cand.get("id")
            if not cand_id:
                raise ValueError(f"{video_id}: id 없는 후보 발견 — {cand}")
            cand_path = output_root / f"{cand_id}.json"
            cand_path.write_text(json.dumps(cand, ensure_ascii=False, indent=2), encoding="utf-8")
            reloaded_cand = json.loads(cand_path.read_text(encoding="utf-8"))
            if reloaded_cand != cand:
                raise AssertionError(f"후보 파일 왕복(write→read) 불일치: {cand_path}")
            stats["candidates"] += 1

        stats["videos"] += 1
        print(f"[step6_adapter] {video_id}: 후보 {len(candidates)}건, 발화 {len(segments)}건 변환·검증 통과")

    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="STEP5 출력을 STEP6 입력 형식으로 변환(원본 무변경)")
    ap.add_argument(
        "--output", default=str(PROJECT_ROOT / "STEP5_LLM으로_암묵지_후보생성" / "output" /
                                 "step6_adapter_input" / "motion_run1"),
        help="STEP6 --input 에 그대로 넘길 폴더 경로"
    )
    args = ap.parse_args()
    out_dir = Path(args.output)
    stats = run(out_dir)
    print(f"\n[step6_adapter] 완료 — 영상 {stats['videos']}개, 후보 {stats['candidates']}건, "
          f"발화 {stats['transcript_utterances']}건 전부 변환·검증 통과")
    print(f"[step6_adapter] STEP6 실행 시 --input \"{out_dir}\" 로 지정하면 됨")
