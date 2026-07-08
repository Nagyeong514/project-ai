"""
STEP5 융합 미니 검증 하네스 (2026-07-08).

목적: 프롬프트/온도 실험의 피드백 루프를 25분(CLIP1 전체) → 2~3분(윈도우 2~3개 1회 호출)으로.
실험 승급 규칙: **이 하네스 먼저 → 통과 시 CLIP1 → 통과 시 4클립.**

하는 일:
  1. 저장된 aligned_windows에서 지정 윈도우만 뽑아 축소 페이로드 구성(정렬 재실행 없음).
  2. 융합 LLM을 온도별 × N회 호출하고, 매 호출을 fuse()의 재시도 루프 '없이' 1회분만
     즉석 판정: 탈선문자 / 스키마(FusionDraft) / 윈도우 귀속 게이트 / 발화 커버리지 / 접지율.
  3. 온도별 1차 통과율 표 + 서술 품질 눈검사용 원문 덤프 + 결과 JSON 저장.

결정화 실험(같은 날 지시): temperature를 단계적으로 내려(0.3 → 0.2(현행) → 0.1 → 0.0 greedy)
"1차 통과가 기본값"이 되는 최저 온도를 찾는다. 단 접지가 무너지거나 서술이 부자연스러워지는
온도는 탈락 — 그래서 접지율과 서술 덤프를 같이 본다.

사용(GPU 노드):
  python3 mini_fusion_harness.py --config ../파이프라인_통합실행/config.yaml \
      --video-id CLIP1_정상조립과정.mp4 --indices 4,7,10 --temps 0.3,0.2,0.1,0.0 --trials 3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common

import argparse
import json
import re
import time
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from step5_preflight import run_preflight


class HarnessConfig(BaseModel):
    """argparse 원시 문자열을 여기서 검증한다(체크리스트 3단계 — 손타이핑 오류 즉사)."""

    config_path: str
    video_id: str
    indices: List[int]  # aligned_windows에서 뽑을 윈도우 순번(1-base, 원본 기준)
    temps: List[float]
    trials: int

    @field_validator("indices")
    @classmethod
    def _v_indices(cls, v: List[int]) -> List[int]:
        if not (2 <= len(v) <= 4):
            raise ValueError(f"미니 하네스는 윈도우 2~4개용이다(받은 값 {len(v)}개) — 그 이상은 CLIP1로 승급하라")
        if any(i < 1 for i in v):
            raise ValueError("윈도우 순번은 1-base")
        return v

    @field_validator("temps")
    @classmethod
    def _v_temps(cls, v: List[float]) -> List[float]:
        if any(not (0.0 <= t <= 1.0) for t in v):
            raise ValueError(f"temperature 범위 밖: {v}")
        return v

    @field_validator("trials")
    @classmethod
    def _v_trials(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError("trials는 1~10")
        return v


# ── 1회 호출 즉석 판정 ────────────────────────────────────────────────

def judge_one_attempt(comp, raw: str, windows, video_id: str,
                      input_utts: List[str]) -> Dict[str, Any]:
    """fuse() 내부 검증을 재시도 없이 1회분만 그대로 실행해 단계별 합/불을 기록.
    검증 순서는 fuse()와 동일(탈선문자 → 스키마 → 재조립 → 귀속 게이트 → 커버리지)."""
    r: Dict[str, Any] = {
        "derailed": False, "schema_ok": False, "binding_ok": False,
        "coverage_ok": False, "first_pass": False, "fail_stage": None,
        "n_candidates": None, "with_utt": None, "grounded": None,
        "grounding_ratio": None, "quality_flags": [], "texts": [],
    }
    bad = comp._looks_derailed(raw)
    if bad:
        r["derailed"] = True
        r["fail_stage"] = f"derail({''.join(bad[:5])!r})"
        return r
    try:
        draft = comp._parse_draft(raw)
        doc = comp._rebuild_from_draft(draft, windows, video_id)
        r["schema_ok"] = True
    except Exception as e:
        r["fail_stage"] = f"schema({type(e).__name__}: {str(e)[:120]})"
        return r
    r["n_candidates"] = len(doc.candidates)
    try:
        comp._check_window_binding(doc, len(windows))
        r["binding_ok"] = True
    except Exception as e:
        r["fail_stage"] = f"binding({str(e)[:120]})"
    missing = comp._missing_utterances(doc, input_utts)
    cov_fail = bool(input_utts) and len(missing) > len(input_utts) * (1 - comp.min_utterance_coverage)
    r["coverage_ok"] = not cov_fail
    if cov_fail and r["fail_stage"] is None:
        r["fail_stage"] = f"coverage(누락 {len(missing)}/{len(input_utts)})"

    # 접지율(발화 있는 후보 중 utterance 태깅) — 게이트 실패여도 관찰은 남긴다
    with_utt = [c for c in doc.candidates if c.knowledge.situation_source]
    grounded = [c for c in with_utt if c.knowledge.reasoning_origin.value == "utterance"]
    r["with_utt"], r["grounded"] = len(with_utt), len(grounded)
    r["grounding_ratio"] = (len(grounded) / len(with_utt)) if with_utt else None

    # 지시 이행률 체크리스트(2026-07-09 확장): 프롬프트의 기계검증 가능 규칙들.
    # ① placeholder 미출력 ② 금지 필드 미출력(raw에서 검사 — pydantic이 무시하므로)
    # ③ 무발화 후보 origin=model_inferred ④ insight≠reasoning ⑤ 한국어
    forbidden = [k for k in ("diagnostic_steps", "source_utterance", "\"timestamp\"",
                             "\"id\"", "scenario_id", "equipment") if k in raw]
    if forbidden:
        r["quality_flags"].append(f"금지 필드 출력(경고 무시): {forbidden}")
    for c in doc.candidates:
        k = c.knowledge
        texts = k.tacit_insight + k.reasoning + k.situation
        if "..." in texts or "<작업유형>" in texts or "핵심어" in str(c.metadata.keywords):
            r["quality_flags"].append(f"{c.window_ids}: placeholder 복사")
        if not c.knowledge.situation_source and k.reasoning_origin.value == "utterance":
            # _rebuild가 이미 강제 교정하지만, 교정 '전' 위반은 print 로그로만 남아
            # 놓치기 쉬우므로 여기서도 플래그(모델의 원출력 이행률 측정 목적).
            r["quality_flags"].append(f"{c.window_ids}: 무발화인데 utterance 태깅(교정됨)")
        if re.sub(r"\s+", "", k.tacit_insight) == re.sub(r"\s+", "", k.reasoning):
            r["quality_flags"].append(f"{c.window_ids}: insight==reasoning 복붙")
        hangul = len(re.findall(r"[가-힣]", k.tacit_insight + k.reasoning))
        if hangul < 10:
            r["quality_flags"].append(f"{c.window_ids}: 한국어 아님({hangul}자)")
        r["texts"].append({
            "window_ids": c.window_ids,
            "origin": k.reasoning_origin.value,
            "tacit_insight": k.tacit_insight,
            "reasoning": k.reasoning,
        })
    r["first_pass"] = r["schema_ok"] and r["binding_ok"] and r["coverage_ok"]
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP5 융합 미니 검증 하네스")
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    ap.add_argument("--video-id", default="CLIP1_정상조립과정.mp4")
    ap.add_argument("--indices", default="4,7,10",
                    help="원본 aligned_windows의 윈도우 순번(1-base). 기본: W04(발화3)+W07(발화2,GT A2)+W10(무발화)")
    ap.add_argument("--temps", default="0.3,0.2,0.1,0.0", help="실험 온도(내림차순 권장). 0.0=greedy")
    ap.add_argument("--trials", type=int, default=3, help="온도당 반복 횟수")
    args = ap.parse_args()

    hcfg = HarnessConfig(
        config_path=args.config, video_id=args.video_id,
        indices=[int(x) for x in args.indices.split(",") if x.strip()],
        temps=[float(x) for x in args.temps.split(",") if x.strip()],
        trials=args.trials,
    )

    cfg = run_preflight(hcfg.config_path)  # 프리플라이트 통과 후에만 무거운 import/로딩

    from tacit_common import artifacts
    from step5_registry import build_llm
    from step5_prompts.llm_fusion_prompt import build_fusion_messages

    windows_dir = cfg.paths.resolve(cfg.paths.step5_aligned_windows_dir)
    all_windows = artifacts.load_aligned_windows(windows_dir, hcfg.video_id)
    for i in hcfg.indices:
        if i > len(all_windows):
            raise SystemExit(f"윈도우 순번 {i} > 총 {len(all_windows)}개")
    windows = [all_windows[i - 1] for i in sorted(hcfg.indices)]
    input_utts = [u.raw_text for w in windows for u in w.utterances if not u.repeat_hallucination]
    print(f"[HARNESS] 축소 페이로드: 원본 {sorted(hcfg.indices)} → 윈도우 {len(windows)}개, "
          f"발화 {len(input_utts)}건 (미니 내 재번호 W01~W{len(windows):02d})")

    comp = build_llm(cfg.llm)
    comp._load()  # 한 번만 로드하고 온도×횟수 전부 재사용 — 이게 시간 절약의 본체
    payload = comp._serialize(windows)
    messages = build_fusion_messages(hcfg.video_id, payload)

    results: List[Dict[str, Any]] = []
    for temp in hcfg.temps:
        for trial in range(1, hcfg.trials + 1):
            t0 = time.time()
            try:
                raw = comp._infer(messages, temperature=temp)
            except Exception as e:
                results.append({"temp": temp, "trial": trial, "elapsed_s": round(time.time() - t0, 1),
                                "first_pass": False, "fail_stage": f"infer({type(e).__name__})"})
                print(f"[HARNESS] T={temp} #{trial}: 생성 자체 실패 {type(e).__name__}: {str(e)[:100]}")
                continue
            r = judge_one_attempt(comp, raw, windows, hcfg.video_id, input_utts)
            r.update({"temp": temp, "trial": trial, "elapsed_s": round(time.time() - t0, 1)})
            results.append(r)
            g = f"{r['grounded']}/{r['with_utt']}" if r["with_utt"] is not None else "-"
            print(f"[HARNESS] T={temp} #{trial} ({r['elapsed_s']}s): "
                  f"{'PASS' if r['first_pass'] else 'FAIL:' + str(r['fail_stage'])} "
                  f"후보={r['n_candidates']} 접지={g} 품질플래그={len(r['quality_flags'])}")
            for q in r["quality_flags"]:
                print(f"[HARNESS]   품질: {q}")

    # ── 온도별 요약표 ──
    print("\n===== 온도별 요약 =====")
    print(f"{'temp':>5} | {'1차통과':>7} | {'접지율(평균)':>10} | 실패 내역")
    summary = []
    for temp in hcfg.temps:
        rs = [r for r in results if r["temp"] == temp]
        n_pass = sum(1 for r in rs if r.get("first_pass"))
        gr = [r["grounding_ratio"] for r in rs if r.get("grounding_ratio") is not None]
        avg_gr = sum(gr) / len(gr) if gr else None
        fails = [str(r.get("fail_stage")) for r in rs if not r.get("first_pass")]
        print(f"{temp:>5} | {n_pass}/{len(rs):>5} | "
              f"{f'{avg_gr:.2f}' if avg_gr is not None else '   - '}     | {'; '.join(fails) or '-'}")
        summary.append({"temp": temp, "pass": n_pass, "total": len(rs), "avg_grounding": avg_gr,
                        "fails": fails})

    # ── 서술 덤프(눈검사용) — 온도별 마지막 통과 시도의 전문 ──
    print("\n===== 서술 눈검사 덤프(온도별 마지막 PASS 시도) =====")
    for temp in hcfg.temps:
        passed = [r for r in results if r["temp"] == temp and r.get("first_pass")]
        if not passed:
            print(f"\n--- T={temp}: PASS 없음 ---")
            continue
        print(f"\n--- T={temp} ---")
        for t in passed[-1]["texts"]:
            print(f"  {t['window_ids']} [{t['origin']}]")
            print(f"    insight : {t['tacit_insight']}")
            print(f"    reasoning: {t['reasoning']}")

    out_dir = Path("output/mini_harness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"harness_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"video_id": hcfg.video_id, "indices": hcfg.indices, "trials": hcfg.trials,
                   "summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    # 스모크 테스트 원칙: 결과가 비어있지 않은지까지가 검증이다
    assert results, "결과 0건 — 하네스 자체가 헛돌았다"
    print(f"\n[OK] 결과 {len(results)}건 저장 → {out_path}")

    comp.unload()


if __name__ == "__main__":
    main()
