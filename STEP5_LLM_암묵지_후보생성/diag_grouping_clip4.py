"""
Pass 2 응집 진단 — CLIP4 실전 후보로 그루핑만 N회 반복 (2026-07-09).

배경: CLIP4 스모크에서 Pass 2가 병합 0건(전부 1건 그룹)으로 끝나 기준 (a)(b) 실패.
미니 하네스(합성 파편)는 3/3 병합 성공했으므로, 실전 서술에서의 판단이 문제다.
원시 출력을 파이프라인이 안 남기므로 여기서 raw 전문을 저장해 근거를 확보한다.

하는 일: 저장된 pass1 스냅샷(응집 전 12건) + aligned_windows 로드 → 모델 1회 로드 →
grouping 메시지 N회 추론(T=0.2 고정, 재시도 없음) → 회차별 raw/병합셋/게이트 판정 저장.
파이프라인 산출물은 건드리지 않는다(진단 전용 출력 폴더).

사용(GPU): python3 diag_grouping_clip4.py --config ../파이프라인_통합실행/config.yaml --trials 5

--model-override (2026-07-09 추가): 프롬프트·입력을 그대로 두고 모델만 교체하는 A/B 진단.
개정 프롬프트(v1.1 보조선)로도 Qwen2.5-14B가 병합 0/5(출력 바이트 동일 — 결정적 거부)라,
'프롬프트 문제 vs 모델 능력 문제'를 가르기 위한 것. Qwen3 계열은 thinking을 끈다
(브랜드벤치 patch_thinking_off와 동일 패턴 — 인스턴스 래핑, 클래스 무수정).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

from step5_preflight import run_preflight

VIDEO_ID = "CLIP4_재부팅및BIOS.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--model-override", default=None,
                    help="HF repo id — 프롬프트/입력 동일, 모델만 교체(A/B 진단용)")
    ap.add_argument("--use-retry-loop", action="store_true",
                    help="1회 추론 대신 실전 _group()(피드백 재시도 포함)를 trials회 실행 — "
                         "형식 위반(서술 누락/비인접)이 피드백으로 수렴하는지 검증")
    args = ap.parse_args()

    cfg = run_preflight(args.config)

    from tacit_common import artifacts
    from step5_registry import build_llm
    from step5_prompts.llm_grouping_prompt import build_grouping_messages
    from step5_schema.tacit_schema import TacitKnowledgeDocument

    windows_dir = cfg.paths.resolve(cfg.paths.step5_aligned_windows_dir)
    tacit_dir = Path(cfg.paths.resolve(cfg.paths.step5_tacit_json_dir))
    windows = artifacts.load_aligned_windows(windows_dir, VIDEO_ID)
    doc = TacitKnowledgeDocument.model_validate(
        json.loads((tacit_dir / f"{VIDEO_ID}.pass1.json").read_text(encoding="utf-8")))
    print(f"[DIAG] pass1 후보 {len(doc.candidates)}건, 윈도우 {len(windows)}개")

    if args.model_override:
        cfg.llm.params["model_name"] = args.model_override
        print(f"[DIAG] 모델 오버라이드: {args.model_override}")
    comp = build_llm(cfg.llm)
    comp._load()
    if args.model_override and "qwen3" in args.model_override.lower():
        # Qwen3 hybrid thinking 비활성(브랜드벤치와 동일 — <think> 블록의 JSON 오염 방지)
        orig = comp._tok.apply_chat_template

        def wrapped(*a, **k):
            k.setdefault("enable_thinking", False)
            return orig(*a, **k)

        comp._tok.apply_chat_template = wrapped
        print("[DIAG] qwen3: enable_thinking=False 주입")
    payload = comp._serialize_candidates(doc, windows)
    cand_wids = {p["cand_id"]: list(p["window_ids"]) for p in payload}
    messages = build_grouping_messages(VIDEO_ID, payload)
    sys_toks = len(messages[0]["content"]) // 3
    print(f"[DIAG] 입력: system ~{sys_toks}자/3, user {len(messages[1]['content'])}자")

    if args.use_retry_loop:
        # 실전 경로 그대로: _group()은 스키마·게이트 위반 시 짧은 피드백으로 재시도하고,
        # 전부 실패하면 Pass 1 문서를 그대로 반환한다(응집 0건과 구분해 기록).
        infer_calls = {"n": 0}
        orig_infer = comp._infer

        def counted(*a, **k):
            infer_calls["n"] += 1
            return orig_infer(*a, **k)

        comp._infer = counted
        results = []
        for t in range(1, args.trials + 1):
            infer_calls["n"] = 0
            t0 = time.time()
            new_doc = comp._group(doc, windows, VIDEO_ID)
            merged = [list(c.window_ids) for c in new_doc.candidates if len(c.window_ids) > 1]
            pass2_merged = [w for w in merged if w not in
                            [list(c.window_ids) for c in doc.candidates]]
            rec = {"trial": t, "elapsed_s": round(time.time() - t0, 1),
                   "attempts": infer_calls["n"],
                   "n_final": len(new_doc.candidates),
                   "converged": len(new_doc.candidates) != len(doc.candidates),
                   "pass2_merged_windows": pass2_merged,
                   "origins": {"|".join(c.window_ids): c.knowledge.reasoning_origin.value
                               for c in new_doc.candidates if len(c.window_ids) > 1}}
            results.append(rec)
            print(f"[DIAG-RETRY] #{t} ({rec['elapsed_s']}s, 시도 {rec['attempts']}회) "
                  f"후보 {len(doc.candidates)}→{rec['n_final']} 병합={pass2_merged} "
                  f"origin={rec['origins']}")
        out = Path("output/mini_harness")
        out.mkdir(parents=True, exist_ok=True)
        tag = "_retry" + ("_" + args.model_override.split("/")[-1].replace("-", "").replace(".", "")[:16]
                          if args.model_override else "")
        p = out / f"diag_grouping_clip4{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        p.write_text(json.dumps({"video_id": VIDEO_ID, "model": cfg.llm.params.get("model_name"),
                                 "results": results}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print(f"[OK] 저장 → {p}")
        comp.unload()
        return

    results = []
    for t in range(1, args.trials + 1):
        t0 = time.time()
        rec = {"trial": t}
        try:
            raw = comp._infer(messages, temperature=0.2)
            rec["elapsed_s"] = round(time.time() - t0, 1)
            rec["raw"] = raw
            try:
                draft = comp._parse_grouping(raw)
                rec["groups"] = [sorted(g.member_ids) for g in draft.groups]
                rec["merged"] = [g for g in rec["groups"] if len(g) >= 2]
                try:
                    comp._check_grouping(draft, cand_wids, len(windows))
                    rec["gate"] = "PASS"
                except Exception as e:
                    rec["gate"] = f"FAIL: {str(e)[:150]}"
            except Exception as e:
                rec["parse"] = f"FAIL: {type(e).__name__}: {str(e)[:150]}"
        except Exception as e:
            rec["infer_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        results.append(rec)
        print(f"[DIAG] #{t} ({rec.get('elapsed_s')}s) 병합={rec.get('merged')} "
              f"게이트={rec.get('gate')} err={rec.get('infer_error') or rec.get('parse')}")

    out = Path("output/mini_harness")
    out.mkdir(parents=True, exist_ok=True)
    tag = ""
    if args.model_override:
        tag = "_" + args.model_override.split("/")[-1].replace("-", "").replace(".", "")[:16]
    p = out / f"diag_grouping_clip4{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps({"video_id": VIDEO_ID, "model": cfg.llm.params.get("model_name"),
                             "payload": payload, "results": results},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 저장 → {p}")
    comp.unload()


if __name__ == "__main__":
    main()
