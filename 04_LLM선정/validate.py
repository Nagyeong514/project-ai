"""validate.py — 골든셋 형식 점검 (계획서 3.2).

generate.py / judge.py 가 실제로 읽는 규칙과 동일한 기준으로 골든셋을 검사한다.
"여기서 통과 = 파이프라인이 정상으로 읽음" 이 되도록 별칭 규칙을 맞췄다.

검사 항목:
- JSON 파싱 깨짐(콤마/따옴표 오류) → 줄·열 위치 + 주변 스니펫 표시
- id(또는 sample_id) 누락 / 중복
- input.vlm_result, input.transcript 빈 값(중첩/ top-level 모두 허용)
- reference(또는 ideal_tacit_candidates / input.reference) 누락
- 전체 통과/실패 개수 + 문제 레코드 번호 요약

사용:
  python validate.py                 # config.yaml 의 golden_set 경로 검사
  python validate.py 다른파일.json    # 임의 파일 검사
종료코드: 모두 통과=0, 하나라도 실패=1 (스크립트/CI 에서 활용)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def resolve_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).resolve()
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return (cfg_path.parent / cfg["paths"]["golden_set"]).resolve()


def error_snippet(text: str, pos: int, width: int = 40) -> str:
    """파싱 실패 위치 주변 텍스트를 ^ 표시와 함께 반환."""
    start = max(0, pos - width)
    end = min(len(text), pos + width)
    frag = text[start:end].replace("\n", "⏎")
    caret = " " * (pos - start) + "^"
    return f"    …{frag}…\n     {caret}"


def record_id(rec: dict):
    """id 별칭: id / sample_id (generate.py·judge.py 와 동일)."""
    return rec.get("id") or rec.get("sample_id")


def has_reference(rec: dict) -> bool:
    ref = (rec.get("reference")
           or rec.get("ideal_tacit_candidates")
           or rec.get("input", {}).get("reference"))
    if ref is None:
        return False
    if isinstance(ref, (list, dict, str)):
        return len(ref) > 0
    return True


def main() -> int:
    path = resolve_path(sys.argv)
    if not path.exists():
        print(f"[오류] 파일 없음: {path}")
        return 1

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"[오류] 파일이 비어 있음: {path}")
        return 1

    # ── 1) JSON 파싱 (배열/JSONL 자동 인식) + 깨짐 위치 표시 ──
    records: list[dict] = []
    if text[0] == "[":  # JSON 배열
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[JSON 파싱 실패] {path.name}")
            print(f"    위치: {e.lineno}행 {e.colno}열 — {e.msg}")
            print(error_snippet(text, e.pos))
            print("\n→ 보통 직전 항목 끝의 콤마 누락/과다, 따옴표 짝 안 맞음이 원인입니다.")
            return 1
        if not isinstance(data, list):
            print("[오류] 최상위가 배열이 아닙니다(JSON 배열 또는 JSONL 이어야 함).")
            return 1
        records = data
    else:  # JSONL
        for ln, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[JSON 파싱 실패] {path.name} — {ln}행")
                print(f"    {e.msg} (열 {e.colno})")
                print(error_snippet(line, e.colno - 1))
                return 1

    if not records:
        print("[오류] 레코드가 0개입니다.")
        return 1

    # ── 2) 레코드별 내용 검사 ──
    problems: list[str] = []   # "N번(id): 사유"
    failed_idx: set[int] = set()
    seen_ids: dict[str, int] = {}

    def fail(i: int, sid, reason: str):
        failed_idx.add(i)
        problems.append(f"  #{i} (id={sid!r}): {reason}")

    for i, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            fail(i, None, "레코드가 객체(dict)가 아님")
            continue

        sid = record_id(rec)
        if sid in (None, ""):
            fail(i, sid, "id/sample_id 누락")
        else:
            sid = str(sid)
            if sid in seen_ids:
                fail(i, sid, f"id 중복 (먼저 #{seen_ids[sid]} 에 등장)")
            else:
                seen_ids[sid] = i

        inp = rec.get("input", rec)  # 중첩/ top-level 모두 허용
        if not str(inp.get("vlm_result", "")).strip():
            fail(i, sid, "input.vlm_result 비어 있음")
        if not str(inp.get("transcript", "")).strip():
            fail(i, sid, "input.transcript 비어 있음")

        if not has_reference(rec):
            fail(i, sid, "reference/ideal_tacit_candidates 누락(또는 빈 값)")

    # ── 3) 요약 ──
    total = len(records)
    n_fail = len(failed_idx)
    n_pass = total - n_fail
    print(f"검사 파일: {path}")
    print(f"형식: {'JSON 배열' if text[0] == '[' else 'JSONL'}  |  레코드 {total}개\n")

    if problems:
        print(f"❌ 문제 {len(problems)}건 (실패 레코드 {n_fail}개):")
        for p in problems:
            print(p)
        print(f"\n요약: 통과 {n_pass}/{total}, 실패 {n_fail}/{total} "
              f"→ 실패 번호: {sorted(failed_idx)}")
        return 1

    print(f"✅ 통과 {n_pass}/{total} — 모든 레코드 형식 정상. (id 중복 없음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
