"""
STEP5 융합 LLM 브랜드 비교 벤치 — 클립2 (2026-07-08).

Qwen3-14B / Mistral Nemo 12B 를 동일 조건에서 비교한다.
(Gemma 3 12B 는 후보였으나 제외 확정 — gated 모델인데 HF_TOKEN 없음 + Gemma 계열
 fp16 오버플로 전례(bf16 전제 학습)로 이 클러스터(Turing, bf16 불가)와 궁합 나쁨.
 토큰 확보 시 MODELS 에 ("gemma", "google/gemma-3-12b-it", False) 한 줄 추가로 재편입.)
QwenLLMFusion 클래스는 수정하지 않는다 — 인스턴스 3개를 파라미터만 다르게 생성.

통제 조건(하나라도 어기면 비교 무효):
  1. 입력 고정 — output/llm_brand_bench/clip2_windows_frozen.json 을 세 모델이 공유.
     (최초 실행 시 aligned_windows 의 CLIP2 를 1회 스냅샷; 이후 절대 재생성 안 함.
      모델마다 STEP3/4/aligner 재실행 금지 — aligner 비결정성이 입력을 오염시킴.)
  2. 양자화 통일 — 전 모델 backend=hf_transformers, quantization=nf4, dtype=half.
  3. 생성 통일 — temperature=0.2, max_new_tokens=2048, 프롬프트(build_fusion_messages
     현재 버전) 동일. 모델명만 다름. max_retries=2(총 3회)로 통일해 시간 상한.
  4. 각 모델 2회(샘플링 편차 확인). 회차별 저장.

함정 방어:
  - 스모크 테스트: 본 실행 전 윈도우 1개 더미 생성으로 커널(sdpa EFFICIENT_ATTENTION은
    Qwen/Turing에서만 검증됨)+nf4+device_map 첫 forward 통과 확인. 죽는 모델은
    "실행불가(사유 기록)" 처리하고 나머지는 계속.
  - Qwen3 thinking 함정: 인스턴스의 tokenizer.apply_chat_template 에 enable_thinking=False
    를 주입(클래스 무수정 — 인스턴스 레벨 래핑). 미주입 시 <think> 블록이 JSON 파싱 오염.
  - peak VRAM 을 모델·회차별 기록(단독 실행 기준).

출력(기존 motion_run2 는 한 파일도 안 건드림):
  - bench_outputs/motion_run3_{qwen3,gemma,nemo}/run{1,2}/CLIP2.mp4.tacit.json
  - llm_brand_bench/llm_compare_clip2.md — 윈도우별 3모델 나란히 (핵심 산출물)
  - llm_brand_bench/metrics.json + 요약표(md 상단)

★ 2026-07-09 사전연구 이관: STEP5_LLM으로_암묵지_후보생성/ → 00_사전연구/
  06_LLM브랜드벤치_STEP5융합/ (결론: 교체 실익 없음, Qwen2.5-14B 유지 — README 참고).
  출력 경로는 스크립트 위치 기준으로 고쳐 재실행 가능하나, STEP5 모듈
  (step5_preflight/step5_registry/step5_prompts)을 sys.path 로 끌어오므로
  STEP5 폴더 구조가 바뀌면 같이 깨진다.

사용(GPU 노드): python3 llm_brand_bench_clip2.py --config <project-ai>/파이프라인_통합실행/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# 사전연구 이관(2026-07-09) 후 경로: project-ai 루트(tacit_common)와 STEP5 폴더
# (step5_preflight/step5_registry/step5_prompts) 둘 다 sys.path 에 필요하다.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "STEP5_LLM으로_암묵지_후보생성"))

import argparse
import json
import shutil
import time
from typing import Any, Dict, List, Optional

from step5_preflight import run_preflight

BENCH_DIR = Path(__file__).resolve().parent / "llm_brand_bench"
FROZEN = BENCH_DIR / "clip2_windows_frozen.json"
VIDEO_ID = "CLIP2.mp4"

# 정확한 HF repo id — 로그에 남긴다. 버전 갱신 시 여기만 수정.
MODELS = [
    # (brand, repo_id, thinking_patch)
    ("qwen3", "Qwen/Qwen3-14B", True),
    ("nemo", "mistralai/Mistral-Nemo-Instruct-2407", False),
]
N_RUNS = 2


def freeze_input(cfg) -> None:
    """최초 1회만 aligned_windows→스냅샷. 이미 있으면 절대 덮지 않는다(입력 고정)."""
    if FROZEN.exists():
        print(f"[BENCH] 고정 입력 재사용: {FROZEN}")
        return
    src = Path(cfg.paths.resolve(cfg.paths.step5_aligned_windows_dir)) / f"{VIDEO_ID}.windows.json"
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, FROZEN)
    print(f"[BENCH] 입력 스냅샷 생성: {src} → {FROZEN}")


def load_frozen_windows():
    from tacit_common.schema.intermediate import AlignedWindow
    raw = json.loads(FROZEN.read_text(encoding="utf-8"))
    return [AlignedWindow.model_validate(w) for w in raw["windows"]]


def build_component(cfg, repo_id: str):
    """생산 config의 llm params 를 기반으로, 통제 조건만 강제 덮어쓴 인스턴스 생성."""
    from step5_components.llm_fusion import QwenLLMFusion
    params = dict(cfg.llm.params or {})
    params.update(
        model_name=repo_id,
        backend="hf_transformers",
        quantization="nf4",       # 기본값 awq 금지 — 양자화 통일
        dtype="half",             # Turing: bf16 금지, fp16 compute 통일
        temperature=0.2,
        max_new_tokens=2048,
        max_retries=2,            # 총 3회 상한(시간 통제) — 시도 수는 지표로 기록
    )
    return QwenLLMFusion(**params)


def patch_thinking_off(comp) -> None:
    """Qwen3 hybrid thinking 비활성 — 클래스 무수정, 로드된 tokenizer 인스턴스만 래핑."""
    orig = comp._tok.apply_chat_template

    def wrapped(*a, **k):
        k.setdefault("enable_thinking", False)
        return orig(*a, **k)

    comp._tok.apply_chat_template = wrapped
    print("[BENCH] qwen3: enable_thinking=False 주입")


def count_infer_calls(comp, counter: Dict[str, int]) -> None:
    """재시도 횟수 계측 — 인스턴스 메서드 래핑(클래스 무수정)."""
    orig = comp._infer

    def wrapped(*a, **k):
        counter["calls"] += 1
        return orig(*a, **k)

    comp._infer = wrapped


def gpu_peak_gib() -> Dict[int, float]:
    import torch
    return {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
            for i in range(torch.cuda.device_count())}


def smoke_test(comp, windows) -> Dict[str, Any]:
    """윈도우 1개 더미 생성으로 커널/템플릿/양자화 첫 forward 검증. 5분 내 판가름."""
    from step5_prompts.llm_fusion_prompt import build_fusion_messages
    saved = comp.max_new_tokens
    comp.max_new_tokens = 256
    t0 = time.time()
    try:
        payload = comp._serialize(windows[:1])
        raw = comp._infer(build_fusion_messages(VIDEO_ID, payload), temperature=0.2)
        ok = bool(raw and raw.strip())
        derail = comp._looks_derailed(raw)
        return {"ok": ok and not derail, "elapsed_s": round(time.time() - t0, 1),
                "derail_chars": "".join(derail[:10]) or None,
                "head": (raw or "")[:200]}
    except Exception as e:
        return {"ok": False, "elapsed_s": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        comp.max_new_tokens = saved


def fusion_metrics(doc, windows) -> Dict[str, Any]:
    """자동 지표: 후보 수 / 융합 성공률 / utterance 접지율.
    융합 성공 = fusion 케이스(발화+행동) 윈도우를 가진 후보가 (i) 스텝에 두 evidence 를
    모두 담고 (ii) reasoning_origin=utterance 일 때. (i)는 재조립이 자동 보장하므로
    실질 판별은 (ii) — 발화를 근거로 엮었는가."""
    fusion_wids = {f"W{i:02d}" for i, w in enumerate(windows, 1) if w.case == "fusion"}
    n_fused = 0
    n_on_fusion = 0
    n_utt_origin = 0
    for c in doc.candidates:
        k = c.knowledge
        if k.reasoning_origin.value == "utterance":
            n_utt_origin += 1
        if set(c.window_ids) & fusion_wids:
            n_on_fusion += 1
            evid = {s.evidence.value for s in k.diagnostic_steps}
            if {"utterance", "action_only"} <= evid and k.reasoning_origin.value == "utterance":
                n_fused += 1
    return {
        "n_candidates": len(doc.candidates),
        "fusion_windows": len(fusion_wids),
        "candidates_on_fusion_windows": n_on_fusion,
        "fused_ok": n_fused,
        "fusion_rate": round(n_fused / n_on_fusion, 2) if n_on_fusion else None,
        "utterance_origin_ratio": round(n_utt_origin / len(doc.candidates), 2)
        if doc.candidates else None,
    }


def run_bench(cfg) -> Dict[str, Any]:
    import gc
    import torch
    from tacit_common.schema.intermediate import FrameMeta

    windows = load_frozen_windows()
    meta = FrameMeta(video_id=VIDEO_ID, path=VIDEO_ID, fps=1.0, n_frames=1)
    print(f"[BENCH] 고정 입력: 윈도우 {len(windows)}개 "
          f"(fusion 케이스 {sum(1 for w in windows if w.case == 'fusion')}개)")

    report: Dict[str, Any] = {"video_id": VIDEO_ID, "models": {}}
    for brand, repo_id, thinking_patch in MODELS:
        print(f"\n===== [{brand}] {repo_id} =====")
        entry: Dict[str, Any] = {"repo_id": repo_id, "runs": []}
        report["models"][brand] = entry
        comp = build_component(cfg, repo_id)
        try:
            t0 = time.time()
            comp._load()
            entry["load_s"] = round(time.time() - t0, 1)
        except Exception as e:
            entry["status"] = "실행불가(로드 실패)"
            entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"[BENCH] {brand} 로드 실패 → 실행불가: {entry['error']}")
            continue
        if thinking_patch:
            patch_thinking_off(comp)

        torch.cuda.reset_peak_memory_stats()
        entry["smoke"] = smoke_test(comp, windows)
        entry["smoke"]["peak_vram_gib"] = gpu_peak_gib()
        print(f"[BENCH] {brand} smoke: {entry['smoke']}")
        if not entry["smoke"]["ok"]:
            entry["status"] = "실행불가(스모크 실패)"
            comp.unload(); gc.collect(); torch.cuda.empty_cache()
            continue

        for n in range(1, N_RUNS + 1):
            counter = {"calls": 0}
            count_infer_calls(comp, counter)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            rec: Dict[str, Any] = {"run": n}
            try:
                doc = comp.fuse(windows, meta)
                rec["elapsed_s"] = round(time.time() - t0, 1)
                rec["attempts"] = counter["calls"]
                rec["peak_vram_gib"] = gpu_peak_gib()
                rec.update(fusion_metrics(doc, windows))
                out_dir = Path(__file__).resolve().parent / f"bench_outputs/motion_run3_{brand}/run{n}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{VIDEO_ID}.tacit.json"
                out.write_text(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2),
                               encoding="utf-8")
                rec["output"] = str(out)
                print(f"[BENCH] {brand} run{n}: 후보 {rec['n_candidates']}건, "
                      f"시도 {rec['attempts']}회, {rec['elapsed_s']}s, VRAM {rec['peak_vram_gib']}")
            except Exception as e:
                rec["elapsed_s"] = round(time.time() - t0, 1)
                rec["attempts"] = counter["calls"]
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"[BENCH] {brand} run{n} 실패: {rec['error']}")
            entry["runs"].append(rec)
        entry["status"] = "완료" if any("output" in r for r in entry["runs"]) else "전 회차 실패"
        comp.unload(); gc.collect(); torch.cuda.empty_cache()

    return report


def write_compare_view(report: Dict[str, Any]) -> None:
    """핵심 산출물 — 윈도우별로 세 모델의 insight/reasoning/origin 을 나란히."""
    windows = load_frozen_windows()
    lines: List[str] = ["# STEP5 융합 LLM 3종 비교 — CLIP2 (입력 고정)", ""]

    # 자동 지표 요약표
    lines += ["## 자동 지표", "",
              "| 모델 | 회차 | 후보 | 융합성공 | utt접지율 | 시도 | 시간(s) | peakVRAM(GiB) |",
              "|---|---|---|---|---|---|---|---|"]
    for brand, e in report["models"].items():
        if not e.get("runs"):
            lines.append(f"| {brand} | - | {e.get('status')} — {e.get('error', '')[:80]} | | | | | |")
            continue
        for r in e["runs"]:
            if "error" in r:
                lines.append(f"| {brand} | {r['run']} | 실패: {r['error'][:60]} | | | "
                             f"{r.get('attempts', '-')} | {r.get('elapsed_s', '-')} | |")
            else:
                vram = max(r["peak_vram_gib"].values())
                lines.append(f"| {brand} | {r['run']} | {r['n_candidates']} | "
                             f"{r['fused_ok']}/{r['candidates_on_fusion_windows']} | "
                             f"{r['utterance_origin_ratio']} | {r['attempts']} | "
                             f"{r['elapsed_s']} | {vram} |")
    lines += ["", "융합성공 = 발화+행동 윈도우의 후보가 두 증거를 스텝에 담고 "
              "reasoning_origin=utterance 인 비율(분모=해당 윈도우 후보 수).", ""]

    # 회차별 문서 로드
    docs: Dict[str, Dict[int, Any]] = {}
    for brand, e in report["models"].items():
        docs[brand] = {}
        for r in e.get("runs", []):
            if "output" in r:
                docs[brand][r["run"]] = json.loads(Path(r["output"]).read_text(encoding="utf-8"))

    def cands_for(brand: str, run: int, wid: str) -> List[dict]:
        doc = docs.get(brand, {}).get(run)
        if not doc:
            return []
        return [c for c in doc["candidates"] if wid in c["window_ids"]]

    lines.append("## 윈도우별 3모델 대조 (판단 재료 — 유창성·일관성은 사람이 평가)")
    for i, w in enumerate(windows, 1):
        wid = f"W{i:02d}"
        lines += ["", f"### {wid} ({w.case}) {w.window_start:.0f}~{w.window_end:.0f}s", "",
                  "**입력**"]
        for a in w.actions[:3]:
            lines.append(f"- 행동 @{a.timestamp:.0f}s: {a.action[:80]}")
        if len(w.actions) > 3:
            lines.append(f"- (행동 {len(w.actions) - 3}건 더)")
        for u in w.utterances:
            if not u.repeat_hallucination:
                lines.append(f"- 발화 @{u.start:.0f}s: “{u.raw_text}”")
        for brand, _, _ in MODELS:
            e = report["models"].get(brand, {})
            lines += ["", f"**{brand}** ({e.get('repo_id', '')})"]
            if not docs.get(brand):
                lines.append(f"- {e.get('status', '결과 없음')}")
                continue
            for run in sorted(docs[brand]):
                cs = cands_for(brand, run, wid)
                if not cs:
                    lines.append(f"- run{run}: (이 윈도우를 담은 후보 없음)")
                for c in cs:
                    k = c["knowledge"]
                    merged = "+".join(c["window_ids"])
                    lines += [f"- run{run} [{merged}] origin={k['reasoning_origin']}",
                              f"  - insight: {k['tacit_insight']}",
                              f"  - reasoning: {k['reasoning']}"]

    out = BENCH_DIR / "llm_compare_clip2.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[BENCH] 비교 뷰 저장 → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP5 융합 LLM 3종 비교(클립2)")
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml")
    args = ap.parse_args()

    cfg = run_preflight(args.config)
    freeze_input(cfg)
    report = run_bench(cfg)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_compare_view(report)

    ran = [b for b, e in report["models"].items() if e.get("status") == "완료"]
    assert ran, "세 모델 전부 실행불가 — metrics.json 의 사유를 볼 것"
    print(f"\n[OK] 벤치 완료 — 실행 성공 모델: {ran}")


if __name__ == "__main__":
    main()
