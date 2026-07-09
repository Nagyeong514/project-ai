"""Pass 2(응집) 스모크 판정기 — CLIP4 기준 (a)~(d) 자동 검사 (2026-07-09).

입력: STEP5 산출물(tacit.json + pass1.json + windows.json). 사람 판단이 필요한
서술 품질은 표로만 출력하고, 기계 판정 가능한 (a)~(d)는 PASS/FAIL을 박는다.

기준 개정(2026-07-09, 사용자 확정): 구 (a) "GT A8 구간 90% 스팬 커버"는 비인접 반려
게이트 하에서 구조적 불가(최대 62% — W10과 W12 사이 W11이 다른 목적) → 기준 재정의,
게이트는 유지. W12는 단독 유지 허용(발화 자체에 A8 회수 재료 포함).
  (a) 수치확인 파편 병합: 한 병합 그룹의 window_ids가 {W09,W10}을 포함
      + 진입 계열 응집: 한 그룹의 window_ids가 {W05,W06,W07}을 포함
  (b) {W09,W10} 병합 그룹의 reasoning_origin=utterance (접지 멤버 reasoning 상속).
      진입 계열 그룹은 origin 무관(발화 없으면 model_inferred가 정상).
  (c) 1건 그룹들의 서술(knowledge+metadata LLM몫+case)이 Pass 1과 바이트 동일
  (d) 귀속 게이트: 윈도우 전수 귀속·무중복 + 병합 그룹 인접성 + 전체 뭉침 없음

Pass 1 품질 확인(2026-07-09 방침 변경 — Qwen3-14B 전면 확정, run2 통제비교 포기):
  (e) Pass 1 후보 서술·diagnostic_steps의 시각이 전부 그 후보 윈도우 범위(±5s) 안
      (존재하지 않는 시각 인용 = FAIL)
  (f) reasoning_origin=utterance 후보의 reasoning이 실제 그 윈도우 발화에 근거
      (인용문이 발화에 실존하거나 정규화 공통부분 존재; 발화 0건 윈도우의
       utterance 주장 = 신고율 비정상 FAIL)
  (g) Pass 1 후보 수 상식 범위: CLIP4 기준 4~24건 (3건 이하 뭉침 / 25건 이상 파편화 = FAIL)

(a)~(g) 전부 PASS면 4클립 전체 → 채점 → STEP6 재판정까지 자동 진행(중단 없음).

사용: python3 check_pass2_smoke.py [--video-id CLIP4_재부팅및BIOS.mp4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/ai_user/team_a2/members/안나경/project-ai")
OUT = ROOT / "STEP5_LLM으로_암묵지_후보생성" / "output"

def narrative_key(cand: dict) -> tuple:
    """1건 그룹 바이트 동일성 대조 키 — 코드가 어차피 재부여하는 id는 제외하고,
    LLM 서술 몫 전체(knowledge 통째) + metadata의 LLM 몫 + 결박/유도 필드."""
    m = cand.get("metadata", {})
    return (
        tuple(cand.get("window_ids", [])),
        cand.get("case"),
        json.dumps(cand.get("knowledge", {}), ensure_ascii=False, sort_keys=True),
        m.get("task"), tuple(m.get("keywords", [])), m.get("scenario_title"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", default="CLIP4_재부팅및BIOS.mp4")
    args = ap.parse_args()
    vid = args.video_id

    final = json.loads((OUT / "tacit_json" / f"{vid}.tacit.json").read_text(encoding="utf-8"))
    pass1 = json.loads((OUT / "tacit_json" / f"{vid}.pass1.json").read_text(encoding="utf-8"))
    wins = json.loads((OUT / "aligned_windows" / f"{vid}.windows.json").read_text(encoding="utf-8"))["windows"]
    n_win = len(wins)
    wtime = {f"W{i:02d}": (w["window_start"], w["window_end"]) for i, w in enumerate(wins, 1)}

    fc, pc = final["candidates"], pass1["candidates"]
    print(f"[smoke] {vid}: 윈도우 {n_win}개, Pass1 {len(pc)}건 → 최종 {len(fc)}건\n")

    # ── 병합/단독 식별: Pass1 서술과 완전 동일하면 단독(객체 재사용), 아니면 병합 ──
    p1_keys = {narrative_key(c): c for c in pc}
    singles, merged = [], []
    for c in fc:
        (singles if narrative_key(c) in p1_keys else merged).append(c)

    # 병합 그룹의 멤버 = 그 그룹 window_ids에 포함되는 Pass1 후보들
    def members_of(g):
        gw = set(g["window_ids"])
        return [p for p in pc if set(p["window_ids"]) <= gw and narrative_key(p) not in
                {narrative_key(s) for s in singles}]

    print("── 병합 그룹 ──")
    for g in merged:
        mem = members_of(g)
        span = (min(wtime[w][0] for w in g["window_ids"]),
                max(wtime[w][1] for w in g["window_ids"]))
        print(f"  {g['id']} {g['window_ids']} ({span[0]:.0f}~{span[1]:.0f}s) "
              f"origin={g['knowledge']['reasoning_origin']} conflict={g['knowledge']['conflict']} "
              f"case={g.get('case')}")
        print(f"    멤버 {len(mem)}건: {[p['id'].rsplit('_',1)[-1] + str(p['window_ids']) for p in mem]}")
        print(f"    insight: {g['knowledge']['tacit_insight']}")
        print(f"    reasoning: {g['knowledge']['reasoning']}")
    if not merged:
        print("  (없음)")

    results = {}

    # (d) 귀속 게이트 재검증
    seen: dict = {}
    for c in fc:
        for w in c["window_ids"]:
            seen[w] = seen.get(w, 0) + 1
    all_ids = [f"W{i:02d}" for i in range(1, n_win + 1)]
    missing = [w for w in all_ids if w not in seen]
    dup = [w for w, n in seen.items() if n > 1]
    nonadj = []
    for c in fc:
        idxs = sorted(int(w[1:]) for w in c["window_ids"])
        if any(b - a != 1 for a, b in zip(idxs, idxs[1:])):
            nonadj.append(c["id"])
    whole = [c["id"] for c in fc if len(c["window_ids"]) >= n_win]
    results["(d) 귀속 게이트"] = (not missing and not dup and not nonadj and not whole,
                             f"누락{missing} 중복{dup} 비인접{nonadj} 전체뭉침{whole}")

    # (c) 1건 그룹 바이트 동일 + 전수 회계
    n_members = sum(len(members_of(g)) for g in merged)
    account_ok = len(singles) + n_members == len(pc) and \
        len(fc) == len(pc) - (n_members - len(merged))
    results["(c) 1건 그룹 Pass1 바이트 동일"] = (
        account_ok,
        f"단독 {len(singles)}건 전부 Pass1과 동일서술, 병합멤버 {n_members}건 "
        f"(회계 {'일치' if account_ok else '불일치!'})")

    # (a)(b) [2026-07-09 3차 결정(사용자): 판정 제외 — 정보 표시만] Pass 2 병합은
    # "되면 보너스"로 격하. Qwen3가 6/6 시행에서 비인접 {W09,W10,W12}를 고집하고
    # 반려되면 병합을 포기해, 완화된 기준도 run 간 복불복이라 게이트로 부적합.
    # 응집 품질의 최종 판정은 채점 회수표(A8 미회수 시 중단)와 STEP6 판정 분포가 한다.
    VERIFY_WINS = {"W09", "W10", "W12", "W13"}
    verify_g = next((g for g in merged
                     if len(set(g["window_ids"]) & VERIFY_WINS) >= 2), None)
    info_only = {}
    info_only["(a·정보) 수치확인 계열 병합 그룹"] = (
        verify_g is not None,
        f"그룹: {verify_g['window_ids'] if verify_g else '없음'} (계열={sorted(VERIFY_WINS)})")

    # (b·정보) 병합 그룹이 있으면 utterance 상속 여부도 정보로만 표시
    if verify_g is not None:
        ro = verify_g["knowledge"]["reasoning_origin"]
        info_only["(b·정보) 수치확인 그룹 origin"] = (ro == "utterance", f"origin={ro}")

    # ── (e)(f)(g) Pass 1 품질 확인 [2026-07-09 추가 — 새 모델의 기존 규칙 준수] ──
    import re

    def hhmmss_to_sec(h, m, s):
        return int(h) * 3600 + int(m) * 60 + int(s)

    def norm(s):
        return re.sub(r"[\s\W]+", "", s or "")

    # 윈도우별 발화 원문(환각 제외)과 시간 경계
    win_utts = {f"W{i:02d}": [u["raw_text"] for u in w.get("utterances", [])
                              if not u.get("repeat_hallucination")]
                for i, w in enumerate(wins, 1)}

    # (e) 시각 접지 — 서술 속 HH:MM:SS/MM:SS 표기 + diagnostic_steps.timestamp 전부 검사
    ts_bad = []
    TS = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)")
    for c in pc:
        lo = min(wtime[w][0] for w in c["window_ids"]) - 5
        hi = max(wtime[w][1] for w in c["window_ids"]) + 5
        k = c["knowledge"]
        narr = " ".join([k.get("situation", ""), k.get("tacit_insight", ""),
                         k.get("reasoning") or ""])
        for m in TS.finditer(narr):
            sec = (hhmmss_to_sec(m.group(1), m.group(2), m.group(3)) if m.group(3)
                   else int(m.group(1)) * 60 + int(m.group(2)))
            if not (lo <= sec <= hi):
                ts_bad.append(f"{c['id']}: 서술 속 {m.group(0)}({sec}s) ∉ [{lo:.0f},{hi:.0f}]")
        for s in k.get("diagnostic_steps", []):
            if s.get("timestamp"):
                sec = hhmmss_to_sec(*s["timestamp"].split(":"))
                if not (lo <= sec <= hi):
                    ts_bad.append(f"{c['id']} step{s['order']}: {s['timestamp']} ∉ 윈도우 범위")
    results["(e) Pass1 시각 접지(윈도우 범위 내)"] = (
        not ts_bad, "; ".join(ts_bad[:3]) if ts_bad else f"후보 {len(pc)}건 전부 범위 내")

    # (f) utterance 신고 검증 — 발화 실존 + reasoning 내용 접지
    f_bad = []
    n_utt_origin = 0
    for c in pc:
        k = c["knowledge"]
        if k.get("reasoning_origin") != "utterance":
            continue
        n_utt_origin += 1
        utts = [u for w in c["window_ids"] for u in win_utts.get(w, [])]
        if not utts:
            f_bad.append(f"{c['id']}: 발화 0건 윈도우인데 utterance 주장(신고율 비정상)")
            continue
        rn = norm(k.get("reasoning") or "")
        quotes = re.findall(r"[‘'\"]([^‘'\"]{4,60})[’'\"]", k.get("reasoning") or "")
        grounded = False
        for q in quotes:
            qn = norm(q)[:12]
            if qn and any(qn in norm(u) for u in utts):
                grounded = True
                break
        if not grounded:  # 인용 부호가 없으면 정규화 6자 공통부분으로 폴백
            for u in utts:
                un = norm(u)
                if any(un[i:i + 6] in rn for i in range(0, max(1, len(un) - 6), 3)):
                    grounded = True
                    break
        if not grounded:
            f_bad.append(f"{c['id']}: reasoning이 윈도우 발화와 무접점(위장 의심)")
    results["(f) utterance 신고의 발화 접지"] = (
        not f_bad,
        "; ".join(f_bad[:3]) if f_bad else f"utterance 후보 {n_utt_origin}건 전부 접지 확인")

    # (g) Pass 1 후보 수 상식 범위 (CLIP4 기준 run2 = 12~13건)
    n_p1 = len(pc)
    results["(g) Pass1 후보 수 4~24건"] = (
        4 <= n_p1 <= 24, f"{n_p1}건 (3 이하=뭉침, 25 이상=파편화)")

    print("\n── 판정 (c)~(g) ──")
    all_ok = True
    for k, (ok, detail) in results.items():
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {k} — {detail}")
    print("\n── 정보 (판정 제외 — 병합은 보너스) ──")
    for k, (ok, detail) in info_only.items():
        print(f"  [{'○' if ok else '×'}] {k} — {detail}")
    print(f"\n[smoke] 종합: {'PASS — 4클립→채점→STEP6 자동 진행' if all_ok else 'FAIL — 중단 보고'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
