"""
STEP5 Pass 2(지식 단위 응집) 미니 검증 하네스 (2026-07-09).

승급 규칙(기존 하네스와 동일): **이 하네스 먼저 → 통과 시 CLIP4 스모크 → 통과 시 4클립.**

W04+W07+W10 축소 페이로드(mini_fusion_harness와 같은 3윈도우, 미니 내 재번호
W01~W03)에 '가짜 파편 후보 세트'를 만들어 Pass 2를 단독 검증한다:
  - c01(W01, 발화有): 단독 지식 — 1건 그룹으로 남아야 정상.
  - c02+c03(W02, 발화有): 같은 지식(GPU 탈거 요령)을 둘로 쪼갠 파편 —
    병합 + c02(origin=utterance)의 reasoning 상속이 기대 동작.
    c03에 가짜 conflict=true를 심어 conflict OR 승계도 함께 검증.
  - c04(W03, 무발화): 단독 지식(쿨링팬 코드 분리) — 1건 그룹 기대.

Part A (오프라인, GPU/모델 불필요 — 로그인 노드에서 즉시 실행 가능):
  귀속 게이트 6종 + 1건 그룹 스키마 + 병합 그룹 상속 규칙을 합성 GroupingDraft로
  단위검증. LLM 없이 코드 경로만 검사한다.
Part B (GPU): 실제 융합 LLM에 응집 프롬프트를 물려 trials회 호출 —
  1차 통과율 / 기대 병합({c02,c03}) 성사율 / utterance 상속 여부를 기록.

사용:
  Part A만(로그인 노드):  python3 mini_grouping_harness.py --offline-only
  전체(GPU 노드):        python3 mini_grouping_harness.py --config ../파이프라인_통합실행/config.yaml --trials 3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import argparse
import json
import time
from typing import Any, Dict, List

VIDEO_ID_DEFAULT = "CLIP1_정상조립과정.mp4"
INDICES_DEFAULT = "4,7,10"


# ── 가짜 파편 후보 세트 ────────────────────────────────────────────────

def build_fake_doc(comp, windows, video_id: str):
    """미니 윈도우 3개(W01~W03) 위에 파편 후보 4건을 합성해 Pass 1 문서 형태로 만든다.

    서술은 합성이지만 문서 조립은 실제 경로(_rebuild_from_draft)를 그대로 태워
    diagnostic_steps/situation_source/case 등이 실전과 동일한 형태가 되게 한다.
    """
    from step5_schema.tacit_schema import DraftCandidate, DraftKnowledge, FusionDraft, Metadata

    def meta(task, kws, title):
        return Metadata(task=task, keywords=kws, scenario_title=title)

    draft = FusionDraft(candidates=[
        DraftCandidate(
            window_ids=["W01"],
            metadata=meta("RAM 장착", ["RAM", "슬롯", "홈"], "홈에 맞춘 RAM 삽입"),
            knowledge=DraftKnowledge(
                situation="정비공이 RAM 모듈을 슬롯 위치에 맞춰 삽입하고 있다.",
                tacit_insight="RAM 모듈을 슬롯 홈 위치에 맞춰 방향을 확인한 뒤 삽입한다.",
                reasoning="정비공이 홈에 맞춰 넣어야 한다고 말함 — 방향이 어긋나면 핀이 손상될 수 있다.",
                reasoning_origin="utterance")),
        DraftCandidate(  # ── 파편 A: c02 (병합 대상, utterance 접지 멤버)
            window_ids=["W02"],
            metadata=meta("GPU 탈거", ["GPU", "슬롯", "흔들기"], "GPU를 흔들어 느슨하게"),
            knowledge=DraftKnowledge(
                situation="정비공이 GPU를 좌우로 흔들며 슬롯에서 느슨하게 만들고 있다.",
                tacit_insight="GPU를 뺄 때는 먼저 좌우로 흔들어 슬롯에서 느슨하게 만든다.",
                reasoning="정비공이 흔들어서 빼야 슬롯이 상하지 않는다고 말함 — 한 번에 당기면 슬롯이 파손될 수 있다.",
                reasoning_origin="utterance")),
        DraftCandidate(  # ── 파편 B: c03 (병합 대상, model_inferred + 가짜 conflict)
            window_ids=["W02"],
            metadata=meta("GPU 탈거", ["GPU", "당기기"], "GPU를 당겨 꺼내기"),
            knowledge=DraftKnowledge(
                situation="정비공이 느슨해진 GPU를 당겨 슬롯에서 꺼내고 있다.",
                tacit_insight="느슨해진 GPU를 수직으로 당겨 슬롯에서 꺼낸다.",
                reasoning="기울여 당기면 슬롯 접점이 긁힐 수 있다.",
                reasoning_origin="model_inferred",
                conflict=True,
                conflict_detail="(하네스 합성) 관찰은 좌우로 당김인데 발화는 수직으로 뽑는다고 말함")),
        DraftCandidate(
            window_ids=["W03"],
            metadata=meta("GPU 탈거", ["쿨링팬", "연결 코드"], "쿨링팬 코드 분리"),
            knowledge=DraftKnowledge(
                situation="정비공이 GPU 쿨링팬 연결 코드를 아래쪽으로 당겨 분리하고 있다.",
                tacit_insight="GPU를 들어내기 전에 쿨링팬 연결 코드를 먼저 분리한다.",
                reasoning="코드를 연결한 채 들어올리면 커넥터가 뜯어질 수 있다.",
                reasoning_origin="model_inferred")),
    ])
    doc = comp._rebuild_from_draft(draft, windows, video_id)
    comp._assign_ids(doc, video_id)
    return doc


# ── Part A: 오프라인 단위검증 ─────────────────────────────────────────

def expect_raise(name: str, fn, *needles: str) -> Dict[str, Any]:
    try:
        fn()
    except Exception as e:
        msg = str(e)
        ok = all(n in msg for n in needles)
        return {"test": name, "ok": ok,
                "detail": f"{'기대 위반 검출' if ok else '다른 오류'}: {msg[:110]}"}
    return {"test": name, "ok": False, "detail": "위반인데 예외가 안 남(게이트 구멍)"}


def run_offline_tests(comp, windows, video_id: str) -> List[Dict[str, Any]]:
    from step5_schema.tacit_schema import GroupDraft, GroupingDraft

    doc = build_fake_doc(comp, windows, video_id)
    assert len(doc.candidates) == 4, f"합성 문서 후보 4건이어야 함(실제 {len(doc.candidates)})"
    cand_wids = {f"c{j:02d}": list(c.window_ids) for j, c in enumerate(doc.candidates, 1)}
    n_win = len(windows)
    results: List[Dict[str, Any]] = []

    def gd(groups):
        return GroupingDraft.model_validate({"groups": groups})

    merged_ok = {
        "member_ids": ["c02", "c03"],
        "metadata": {"task": "GPU 탈거", "keywords": ["GPU", "슬롯", "흔들기"],
                     "scenario_title": "흔들어 느슨하게 만든 뒤 당겨 꺼내기"},
        "knowledge": {
            "situation": "정비공이 GPU를 좌우로 흔들어 느슨하게 만든 뒤 당겨 슬롯에서 꺼내고 있다.",
            "tacit_insight": "GPU를 뺄 때는 좌우로 흔들어 느슨하게 만든 뒤 당겨 꺼낸다.",
            "reasoning": "정비공이 흔들어서 빼야 슬롯이 상하지 않는다고 말함 — 한 번에 당기면 슬롯이 파손될 수 있다.",
            "reasoning_origin": "utterance"}}

    # A1. 정상 응집: 전수 귀속 + 병합 1건 → 게이트 통과, 적용 결과 3건
    try:
        draft = gd([{"member_ids": ["c01"]}, merged_ok, {"member_ids": ["c04"]}])
        comp._check_grouping(draft, cand_wids, n_win)
        new_doc = comp._apply_grouping(draft, doc, windows, video_id)
        m = next(c for c in new_doc.candidates if len(c.window_ids) == 1 and "W02" in c.window_ids)
        checks = {
            "그룹 3건": len(new_doc.candidates) == 3,
            "병합 window_ids=[W02]": m.window_ids == ["W02"],
            "conflict OR 승계": m.knowledge.conflict is True and bool(m.knowledge.conflict_detail),
            "utterance 상속 유지": m.knowledge.reasoning_origin.value == "utterance",
            "병합 steps 재생성": len(m.knowledge.diagnostic_steps) > 0,
            "case 유도됨": all(c.case in ("both", "utterance", "action") for c in new_doc.candidates),
        }
        # 1건 그룹 바이트 동일: 같은 '객체'를 재사용해야 한다(id는 재부여되므로 제외하고 대조)
        singles_same_obj = (
            any(c is doc.candidates[0] for c in new_doc.candidates)
            and any(c is doc.candidates[3] for c in new_doc.candidates))
        checks["1건 그룹 객체 재사용(바이트 동일)"] = singles_same_obj
        ok = all(checks.values())
        results.append({"test": "A1 정상 응집+상속", "ok": ok,
                        "detail": "; ".join(f"{k}={'O' if v else 'X'}" for k, v in checks.items())})
    except Exception as e:
        results.append({"test": "A1 정상 응집+상속", "ok": False,
                        "detail": f"예외: {type(e).__name__}: {str(e)[:110]}"})

    # A2~A6. 게이트 위반 검출
    results.append(expect_raise(
        "A2 누락 검출(c04 없음)",
        lambda: comp._check_grouping(gd([{"member_ids": ["c01"]}, merged_ok]), cand_wids, n_win),
        "누락"))
    results.append(expect_raise(
        "A3 중복 검출(c01 두 그룹)",
        lambda: comp._check_grouping(gd([{"member_ids": ["c01"]}, {"member_ids": ["c01", "c02"],
                                          "metadata": merged_ok["metadata"], "knowledge": merged_ok["knowledge"]},
                                         {"member_ids": ["c03"]}, {"member_ids": ["c04"]}]), cand_wids, n_win),
        "중복"))
    results.append(expect_raise(
        "A4 유령 cand_id 즉사(c99)",
        lambda: comp._check_grouping(gd([{"member_ids": ["c99"]}, {"member_ids": ["c01"]},
                                         {"member_ids": ["c02"]}, {"member_ids": ["c03"]},
                                         {"member_ids": ["c04"]}]), cand_wids, n_win),
        "존재하지 않는"))
    results.append(expect_raise(
        "A5 비인접 병합 반려(c01+c04, W01+W03)",
        lambda: comp._check_grouping(gd([{"member_ids": ["c01", "c04"],
                                          "metadata": merged_ok["metadata"], "knowledge": merged_ok["knowledge"]},
                                         {"member_ids": ["c02"]}, {"member_ids": ["c03"]}]), cand_wids, n_win),
        "비인접"))
    results.append(expect_raise(
        "A6 과잉병합 반려(전체 윈도우 한 그룹)",
        lambda: comp._check_grouping(gd([{"member_ids": ["c01", "c02", "c03", "c04"],
                                          "metadata": merged_ok["metadata"], "knowledge": merged_ok["knowledge"]}]),
                                     cand_wids, n_win),
        "과잉병합"))

    # A7. 1건 그룹 스키마: knowledge를 echo해도 무시하고 Pass 1 원문 재사용
    try:
        draft = gd([{"member_ids": ["c01"],
                     "knowledge": {"situation": "(무시돼야 할 재서술)", "tacit_insight": "(무시)",
                                   "reasoning": "(무시)", "reasoning_origin": "model_inferred"}},
                    merged_ok, {"member_ids": ["c04"]}])
        comp._check_grouping(draft, cand_wids, n_win)
        new_doc = comp._apply_grouping(draft, doc, windows, video_id)
        c1 = next(c for c in new_doc.candidates if c.window_ids == ["W01"])
        ok = (c1 is doc.candidates[0]
              and c1.knowledge.situation == "정비공이 RAM 모듈을 슬롯 위치에 맞춰 삽입하고 있다.")
        results.append({"test": "A7 1건 그룹 echo 무시(원문 재사용)", "ok": ok,
                        "detail": "Pass 1 객체 그대로" if ok else "재서술이 반영됨(규칙 위반)"})
    except Exception as e:
        results.append({"test": "A7 1건 그룹 echo 무시(원문 재사용)", "ok": False,
                        "detail": f"예외: {type(e).__name__}: {str(e)[:110]}"})

    # A8. utterance 위장 교정: utterance 멤버가 없는 병합 그룹(c03+c04)을 utterance 태깅
    try:
        masq = {"member_ids": ["c03", "c04"],
                "metadata": merged_ok["metadata"],
                "knowledge": dict(merged_ok["knowledge"], reasoning_origin="utterance")}
        draft = gd([{"member_ids": ["c01"]}, {"member_ids": ["c02"]}, masq])
        comp._check_grouping(draft, cand_wids, n_win)
        new_doc = comp._apply_grouping(draft, doc, windows, video_id)
        m = next(c for c in new_doc.candidates if c.window_ids == ["W02", "W03"])
        ok = m.knowledge.reasoning_origin.value == "model_inferred"
        results.append({"test": "A8 utterance 위장 강제 교정", "ok": ok,
                        "detail": f"최종 origin={m.knowledge.reasoning_origin.value}"})
    except Exception as e:
        results.append({"test": "A8 utterance 위장 강제 교정", "ok": False,
                        "detail": f"예외: {type(e).__name__}: {str(e)[:110]}"})

    # A9. 병합 그룹 knowledge 누락 → 스키마 단계에서 잡혀야 함
    results.append(expect_raise(
        "A9 병합 그룹 knowledge 누락 검출",
        lambda: gd([{"member_ids": ["c01"]}, {"member_ids": ["c02", "c03"]},
                    {"member_ids": ["c04"]}]),
        "knowledge"))
    return results


# ── Part B: GPU 실호출 검증 ───────────────────────────────────────────

def judge_grouping_attempt(comp, raw: str, doc, windows, video_id: str,
                           cand_wids: Dict[str, List[str]]) -> Dict[str, Any]:
    r: Dict[str, Any] = {"first_pass": False, "fail_stage": None, "n_groups": None,
                         "merged": [], "expected_merge": False, "inherit_ok": None,
                         "flags": []}
    bad = comp._looks_derailed(raw)
    if bad:
        r["fail_stage"] = f"derail({''.join(bad[:5])!r})"
        return r
    try:
        draft = comp._parse_grouping(raw)
    except Exception as e:
        r["fail_stage"] = f"schema({type(e).__name__}: {str(e)[:110]})"
        return r
    try:
        comp._check_grouping(draft, cand_wids, len(windows))
    except Exception as e:
        r["fail_stage"] = f"gate({str(e)[:110]})"
        return r
    try:
        new_doc = comp._apply_grouping(draft, doc, windows, video_id)
    except Exception as e:
        r["fail_stage"] = f"apply({type(e).__name__}: {str(e)[:110]})"
        return r
    r["first_pass"] = True
    r["n_groups"] = len(new_doc.candidates)
    for g in draft.groups:
        if len(g.member_ids) >= 2:
            r["merged"].append(sorted(g.member_ids))
    r["expected_merge"] = sorted(["c02", "c03"]) in r["merged"]
    if r["expected_merge"]:
        m = next(c for c in new_doc.candidates
                 if c.window_ids == ["W02"] or set(c.window_ids) >= {"W02"})
        r["inherit_ok"] = m.knowledge.reasoning_origin.value == "utterance"
        if not r["inherit_ok"]:
            r["flags"].append(f"utterance 멤버(c02)가 있는데 그룹 origin={m.knowledge.reasoning_origin.value}")
    for mg in r["merged"]:
        if mg != ["c02", "c03"]:
            r["flags"].append(f"기대 밖 병합 {mg}(목적이 다른 후보끼리 묶임 의심)")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP5 Pass 2 응집 미니 하네스")
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    ap.add_argument("--video-id", default=VIDEO_ID_DEFAULT)
    ap.add_argument("--indices", default=INDICES_DEFAULT,
                    help="원본 aligned_windows의 윈도우 순번(1-base). 기본 4,7,10")
    ap.add_argument("--trials", type=int, default=3, help="Part B 호출 횟수")
    ap.add_argument("--offline-only", action="store_true",
                    help="Part A(게이트 단위검증)만 — GPU/모델 불필요")
    args = ap.parse_args()

    from tacit_common import artifacts
    from tacit_common.config import PipelineConfig
    from step5_components.llm_fusion import QwenLLMFusion
    from step5_prompts.llm_grouping_prompt import build_grouping_messages

    cfg = PipelineConfig.load(args.config)  # Part A는 경로만 필요 — preflight는 Part B에서
    windows_dir = cfg.paths.resolve(cfg.paths.step5_aligned_windows_dir)
    all_windows = artifacts.load_aligned_windows(windows_dir, args.video_id)
    indices = sorted(int(x) for x in args.indices.split(",") if x.strip())
    windows = [all_windows[i - 1] for i in indices]
    print(f"[G-HARNESS] 축소 페이로드: 원본 {indices} → 미니 W01~W{len(windows):02d}")

    # Part A는 모델이 필요 없으므로 기본 생성자(로딩 안 함)로 게이트 코드만 태운다.
    comp_offline = QwenLLMFusion()
    offline = run_offline_tests(comp_offline, windows, args.video_id)
    print("\n===== Part A: 오프라인 게이트/상속 단위검증 =====")
    n_ok = 0
    for t in offline:
        n_ok += 1 if t["ok"] else 0
        print(f"  [{'PASS' if t['ok'] else 'FAIL'}] {t['test']} — {t['detail']}")
    print(f"  → {n_ok}/{len(offline)} 통과")
    if n_ok < len(offline):
        raise SystemExit("[G-HARNESS] Part A 실패 — 코드 게이트부터 고쳐야 한다(Part B 진행 금지)")

    out: Dict[str, Any] = {"video_id": args.video_id, "indices": indices,
                           "offline": offline, "gpu_trials": []}
    if not args.offline_only:
        from step5_preflight import run_preflight
        from step5_registry import build_llm
        cfg = run_preflight(args.config)
        comp = build_llm(cfg.llm)
        comp._load()
        doc = build_fake_doc(comp, windows, args.video_id)
        payload = comp._serialize_candidates(doc, windows)
        cand_wids = {p["cand_id"]: list(p["window_ids"]) for p in payload}
        messages = build_grouping_messages(args.video_id, payload)
        print(f"\n===== Part B: GPU 실호출 ({args.trials}회, T=0.2 고정) =====")
        for trial in range(1, args.trials + 1):
            t0 = time.time()
            try:
                raw = comp._infer(messages, temperature=0.2)
            except Exception as e:
                out["gpu_trials"].append({"trial": trial, "first_pass": False,
                                          "fail_stage": f"infer({type(e).__name__})"})
                print(f"  #{trial}: 생성 자체 실패 {type(e).__name__}: {str(e)[:100]}")
                continue
            r = judge_grouping_attempt(comp, raw, doc, windows, args.video_id, cand_wids)
            r.update({"trial": trial, "elapsed_s": round(time.time() - t0, 1), "raw_head": raw[:400]})
            out["gpu_trials"].append(r)
            print(f"  #{trial} ({r['elapsed_s']}s): "
                  f"{'PASS' if r['first_pass'] else 'FAIL:' + str(r['fail_stage'])} "
                  f"그룹={r['n_groups']} 병합={r['merged']} "
                  f"기대병합(c02+c03)={'O' if r['expected_merge'] else 'X'} "
                  f"상속={'O' if r.get('inherit_ok') else ('X' if r.get('inherit_ok') is False else '-')}")
            for fl in r["flags"]:
                print(f"     플래그: {fl}")
        passes = [r for r in out["gpu_trials"] if r.get("first_pass")]
        exp = [r for r in passes if r.get("expected_merge")]
        inh = [r for r in exp if r.get("inherit_ok")]
        print(f"\n  요약: 1차통과 {len(passes)}/{len(out['gpu_trials'])}, "
              f"기대병합 {len(exp)}, utterance 상속 {len(inh)}")
        comp.unload()
        assert out["gpu_trials"], "GPU 시도 0건 — 하네스가 헛돌았다"

    out_dir = Path("output/mini_harness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grouping_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 결과 저장 → {out_path}")


if __name__ == "__main__":
    main()
