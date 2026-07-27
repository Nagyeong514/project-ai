"""tacit2/eval/score.py 방식의 채점기 — STEP5 산출물(tacit_json) 대상 이식판(2026-07-06).

판정 로직은 tacit2 원본과 동일: any_of 키워드 그룹 중 하나라도 그룹 안 키워드가 '전부'
후보 텍스트(situation/tacit_insight/reasoning/diagnostic_steps)에 있으면 회수.
차이: (1) 경로가 STEP5 출력 구조, (2) 정답지는 S1 들여쓰기만 고친 사본(answer_key_fixed.yaml),
(3) S1 특수검사에 conflict 뿐 아니라 reasoning_origin=utterance 검사도 함께 출력(5-B 검증 항목),
(4) --trace 로 누락 항목의 스테이지 역추적(VLM 관찰 → 윈도우 → LLM)까지 자동 수행.
★ 평가 전용 — 정답지 내용을 프롬프트에 절대 주입 금지.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path("/home/ai_user/team_a2/members/안나경/project-ai")
TACIT_DIR = ROOT / "STEP5_LLM_암묵지_후보생성/output/tacit_json"
WINDOWS_DIR = ROOT / "STEP5_LLM_암묵지_후보생성/output/aligned_windows"
OBS_DIR = ROOT / "STEP4_YOLO_VLM관찰/output/vlm_observations"
TRANSCRIPT_DIR = ROOT / "STEP3_전처리/transcripts"


def candidate_text(cand: dict) -> str:
    k = cand.get("knowledge", {})
    parts = [k.get("situation", ""), k.get("tacit_insight", ""), k.get("reasoning", "") or ""]
    for s in k.get("diagnostic_steps", []) or []:
        parts += [s.get("action", ""), s.get("source_utterance") or ""]
    return " ".join(p for p in parts if p)


# ── 동의어 보정(2026-07-06) ──────────────────────────────────────────────
# 채점 관대화가 아니라 '표현 차이 보정': 의미가 실제로 같은 표기만 등록한다.
# 억지 매칭 금지 — 이 표 자체가 검토 대상이므로 추가할 때마다 근거를 주석으로.
# 매칭 판정은 여전히 "그룹 안 모든 개념이 존재해야 회수"로 동일하다(완화 아님).
SYNONYMS: dict[str, list[str]] = {
    "색": ["색상", "컬러"],          # '색'은 '색상'의 부분문자열이라 사실상 표기 통일용
    "라벨": ["레이블", "라벨링"],
    "클릭": ["딸깍"],               # 래치 잠금 소리의 두 표기(A7) — 동일 청각 신호
    "두": ["2", "두 번", "두번", "2회", "두번째"],
    "세척": ["닦"],                 # '닦아준다/닦음' — 동일 행위(A6)
    "BIOS": ["바이오스"],
    "여섯": ["6"],
    "커넥터": ["컨넥터"],           # 표기 흔들림
    "파악": ["확인", "식별"],        # '구성 파악'≈'확인 후 진행'(A1/A8 desc 계열)
}


def _terms_for(kw: str) -> list[str]:
    return [kw] + SYNONYMS.get(kw, [])


def _find_term(text: str, kw: str):
    """kw 또는 등록된 동의어가 text에 있으면 (실제 매칭된 표기, 위치) 반환."""
    for term in _terms_for(kw):
        pos = text.find(term)
        if pos >= 0:
            return term, pos
    return None, -1


def _snippet(text: str, pos: int, width: int = 28) -> str:
    s, e = max(0, pos - width), min(len(text), pos + width)
    return ("…" if s > 0 else "") + text[s:e].replace("\n", " ") + ("…" if e < len(text) else "")


def group_hits(text: str, any_of) -> list:
    """동의어 보정 매칭. 반환: [(그룹, {kw: (매칭표기, 근거 스니펫)}), ...].
    그룹 안 '모든' 키워드(또는 그 동의어)가 존재해야 회수 — 원본과 동일한 엄격도."""
    hits = []
    for g in any_of:
        matched = {}
        for kw in g:
            term, pos = _find_term(text, kw)
            if term is None:
                matched = None
                break
            matched[kw] = (term, _snippet(text, pos))
        if matched is not None:
            hits.append((g, matched))
    return hits


def load_docs(tacit_dir: Path = TACIT_DIR):
    docs = {}
    for p in sorted(tacit_dir.glob("*.tacit.json")):
        cid = p.name.replace(".tacit.json", "")
        docs[cid] = json.loads(p.read_text(encoding="utf-8"))
    return docs


def trace_miss(item, cid_list):
    """누락 항목이 어느 스테이지에서 빠졌는지 역추적.
    (1) VLM 관찰 텍스트 (2) 발화 텍스트 (3) 정렬 윈도우 직렬화 대상 (4) LLM 출력 — 순서로
    키워드 그룹 존재 여부를 본다. 어디까지 살아있었는지가 곧 문제 스테이지."""
    out = []
    for cid in cid_list:
        obs_p = OBS_DIR / f"{cid}.observations.json"
        win_p = WINDOWS_DIR / f"{cid}.windows.json"
        tr_p = TRANSCRIPT_DIR / f"{cid}.json"
        obs_text = ""
        if obs_p.exists():
            d = json.loads(obs_p.read_text(encoding="utf-8"))
            obs_text = " ".join(o.get("action", "") for o in d.get("observations", []))
        utt_text = ""
        if tr_p.exists():
            t = json.loads(tr_p.read_text(encoding="utf-8"))
            utt_text = " ".join(u.get("raw_text", "") + " " + u.get("normalized_text", "")
                                for u in t.get("utterances", []))
        # 윈도우는 action 과 utterance 를 분리 검사 — 합쳐 검사하면 서로 다른 요소의
        # 단어가 우연히 동시출현해 오탐(실측: '커넥터'는 발화에만, '손'은 action 에만
        # 있는데 합본에서 ['커넥터','손'] 그룹이 존재로 잘못 판정됨).
        win_act, win_utt = "", ""
        if win_p.exists():
            w = json.loads(win_p.read_text(encoding="utf-8"))
            for win in w.get("windows", []):
                win_act += " " + " ".join(a.get("action", "") for a in win.get("actions", []))
                win_utt += " " + " ".join(u.get("raw_text", "") for u in win.get("utterances", []))
        stages = [("VLM관찰", obs_text), ("STT발화", utt_text),
                  ("윈도우(action)", win_act), ("윈도우(발화)", win_utt)]
        for name, text in stages:
            hits = group_hits(text, item["any_of"])
            mark = f"키워드그룹 {hits[0][0]} 존재" if hits else "없음"
            out.append(f"      [{cid}] {name}: {mark}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=str(Path(__file__).parent / "answer_key_fixed.yaml"))
    ap.add_argument("--trace", action="store_true", help="누락 항목 스테이지 역추적")
    # 2026-07-08(A/B 비교용): 균일추출 기준선(백업 폴더)을 같은 새 GT로 재채점하기 위한
    # override. 안 주면 기존 그대로 라이브 STEP5 출력(TACIT_DIR)을 본다 — 채점 로직/키워드
    # 매칭 방식은 전혀 안 바뀜, 어느 tacit_json 묶음을 볼지만 바꾼다.
    ap.add_argument("--tacit-dir", default=None, help="tacit_json 폴더 override(A/B 비교용)")
    args = ap.parse_args()

    key = yaml.safe_load(Path(args.key).read_text(encoding="utf-8"))
    tacit_dir = Path(args.tacit_dir) if args.tacit_dir else TACIT_DIR
    docs = load_docs(tacit_dir)
    if not docs:
        print("[eval] tacit_json 산출물 없음 — 파이프라인 먼저 실행")
        raise SystemExit(1)
    print(f"[eval] 대상 문서: {list(docs)}\n")

    def docs_for(label):
        return [(cid, d) for cid, d in docs.items() if label in cid]

    hit = miss = silent_hit = silent_total = 0
    missed_items = []
    print(f"{'ID':4s} {'클립':7s} {'판정':6s} 항목")
    print("-" * 78)
    for item in key["items"]:
        matched_by = None
        for cid, doc in docs_for(item["clip"]):
            for cand in doc.get("candidates", []):
                hits = group_hits(candidate_text(cand), item["any_of"])
                if hits:
                    matched_by = (cand.get("id", "?"), hits[0])
                    break
            if matched_by:
                break
        is_silent = item.get("silent", False)
        silent_total += 1 if is_silent else 0
        s_mark = "[침묵]" if is_silent else "      "
        if matched_by:
            hit += 1
            silent_hit += 1 if is_silent else 0
            cand_id, (group, matched) = matched_by
            tier = "정확일치" if all(term == kw for kw, (term, _) in matched.items()) else "동의어보정"
            print(f"{item['id']:4s} {item['clip']:7s} 회수✓  {s_mark}{item['desc'][:40]}")
            print(f"      [{tier}] ← {cand_id} (그룹 {group})")
            print(f"      정답지 문구: \"{item['desc']}\"")
            for kw, (term, snip) in matched.items():
                syn = "" if term == kw else f" (동의어 '{term}'로 매칭)"
                print(f"      산출물 근거[{kw}{syn}]: \"{snip}\"")
        else:
            miss += 1
            missed_items.append(item)
            extra = "  ← 침묵 암묵지(action_only 경로 점검)" if is_silent else ""
            print(f"{item['id']:4s} {item['clip']:7s} 누락✗  {s_mark}{item['desc'][:40]}{extra}")

    print("-" * 78)
    print(f"[eval] 회수율: {hit}/{hit + miss}  (침묵 항목: {silent_hit}/{silent_total})")

    # S1 특수 검사: conflict + reasoning_origin=utterance(LED 발화)
    for sp in key.get("special", []):
        if sp.get("check") != "conflict_true":
            continue
        conflict_found = False
        led_utterance = []
        for cid, d in docs_for(sp["clip"]):
            for c in d.get("candidates", []):
                k = c.get("knowledge", {})
                if k.get("conflict") is True:
                    conflict_found = True
                text = candidate_text(c)
                if re.search(r"황색|백색|LED|깜빡|점멸", text):
                    led_utterance.append((c.get("id"), k.get("reasoning_origin"),
                                          k.get("conflict")))
        print(f"\n[eval] S1 {sp['clip']}:")
        print(f"  conflict=true 존재: {'예' if conflict_found else '아니오'}"
              f" (VLM 관찰에 LED 횟수 없으면 false 가 정상 — 정답지 주석)")
        for cid_, ro, cf in led_utterance:
            ok = "✓ 정답" if ro == "utterance" else f"✗ (기대 utterance, 실제 {ro})"
            print(f"  LED 관련 후보 {cid_}: reasoning_origin={ro} {ok}, conflict={cf}")
        if not led_utterance:
            print("  LED 관련 후보 자체가 없음 — A3 누락과 같은 원인일 것")

    if args.trace and missed_items:
        print("\n[trace] 누락 항목 역추적 (어느 스테이지까지 살아있었나):")
        for item in missed_items:
            cids = [cid for cid in docs if item["clip"] in cid]
            print(f"  {item['id']} ({item['clip']}) — {item['desc'][:40]}")
            for line in trace_miss(item, cids):
                print(line)


if __name__ == "__main__":
    main()
