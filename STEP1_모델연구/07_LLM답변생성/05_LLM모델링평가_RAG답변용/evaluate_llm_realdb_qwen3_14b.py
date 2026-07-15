# -*- coding: utf-8 -*-
"""
실DB 벤치마크 — Qwen3-14B(Ollama Q4_K_M) 단독 재실행 (2026-07-11).

evaluate_llm_realdb.py(3종 비교 원본)는 무수정 보존 — 이 파일이 Qwen3-14B 전용 변형이다.
목적: STEP5 융합에 채택된 Qwen3-14B를 '답변 생성(음성 전달)' 역할로도 동일 벤치에 태워
기존 3종(qwen3-vl:8b-instruct / llama3.1:8b / gemma3:4b) 결과와 비교 가능한 수치를 얻는다.
동일 조건: 같은 SYSTEM_PROMPT, 같은 realdb_contexts.json(STEP7 실DB 검색 컨텍스트),
같은 Ollama Q4_K_M 서빙 스택, 같은 측정 방식(Latency/VRAM/Throughput).

Qwen3-14B 특이사항: hybrid thinking 모델 — Ollama chat에 think:false를 보내 사고 모드를
끈다(STEP5 transformers 경로의 enable_thinking=False와 동일 취지). think 파라미터가
거부되는 구버전 서버 대비, 실패 시 think 없이 재호출하고 <think> 블록을 제거하는
폴백을 둔다(폴백 사용 여부는 결과에 기록 — latency 해석 시 구분).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import defaultdict

import requests

OLLAMA_API = "http://localhost:11434/api/chat"
MODELS = ["qwen3:14b"]
MODEL_LABELS = {"qwen3:14b": "Qwen3-14B Q4_K_M"}
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXTS_FILE = os.path.join(EVAL_DIR, "realdb_contexts.json")
OUTPUT_FILE = os.path.join(EVAL_DIR, "llm_benchmark_results_realdb_qwen3_14b.json")
API_TIMEOUT = 300  # 14B + 첫 로드 여유(측정은 성공 호출만 집계, 로드는 워밍업이 흡수)
WARMUP_QUERY = "안녕하세요, 준비됐나요? 한 문장으로만 답하세요."

# SYSTEM_PROMPT — evaluate_llm_realdb.py와 바이트 동일(비교 가능성 유지)
SYSTEM_PROMPT = """
너는 조립 현장의 신입 작업자를 돕는 음성 작업 보조 AI다.
아래 [검색된 암묵지]만 근거로 답하고, 근거가 부족하면 솔직히 부족하다고 말하라.
규칙:
- 첫 1~2문장에 핵심 답을 먼저 말한다 (음성으로 읽힐 것을 가정, 간결한 존댓말)
- 절차가 있으면 번호 단계로 나열
- 마지막 줄에 "출처: 영상ID (시작~끝 타임스탬프)" 표기
- 마크다운 강조 기호 사용 금지, 일반 텍스트로만
"""

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def get_contexts():
    with open(CONTEXTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_vram_used_gib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return sum(int(x) for x in out.splitlines() if x.strip()) / 1024
    except Exception as e:
        print(f"[WARN] nvidia-smi 조회 실패: {e}")
        return None


def call_ollama(model, prompt):
    """(answer, latency_seconds, metrics, think_fallback) 반환. 1차: think=false.
    400/거부 시 think 없이 재호출 + <think> 블록 제거(think_fallback=True로 기록)."""
    base = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    for attempt, use_think in ((1, True), (2, False)):
        payload = dict(base)
        if use_think:
            payload["think"] = False
        start = time.time()
        try:
            r = requests.post(OLLAMA_API, json=payload, timeout=API_TIMEOUT)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            if use_think:
                print(f"  [THINK] think:false 거부({e}) — think 미지정 폴백 재호출")
                continue
            print(f"API Error for model {model}: {e}")
            return None, 0, {}, not use_think
        lat = time.time() - start
        body = r.json()
        answer = body.get("message", {}).get("content", "")
        if not use_think:
            answer = THINK_RE.sub("", answer)
        metrics = {k: body.get(k) for k in (
            "eval_count", "eval_duration", "prompt_eval_count",
            "prompt_eval_duration", "total_duration", "load_duration")}
        return answer, lat, metrics, not use_think
    return None, 0, {}, True


def tokens_per_sec(metrics):
    count, dur_ns = metrics.get("eval_count"), metrics.get("eval_duration")
    if not count or not dur_ns:
        return None
    return count / (dur_ns / 1e9)


def warmup_model(model):
    print(f"  [WARMUP] {model} 로딩 중...")
    ans, lat, _, fb = call_ollama(model, WARMUP_QUERY)
    if ans is None:
        print(f"  [WARMUP] 경고: {model} 워밍업 실패")
    else:
        print(f"  [WARMUP] 완료 ({lat:.1f}초, think_fallback={fb})")


def print_quantitative_report(results, vram_by_model):
    print("\n" + "=" * 100)
    print(" [자동 산출] 실DB 기반 LLM 벤치마크 — Qwen3-14B 단독 (Latency / VRAM / Throughput)")
    print("=" * 100)
    agg = defaultdict(lambda: {"latency_sum": 0.0, "latency_count": 0,
                               "tps_sum": 0.0, "tps_count": 0, "total": 0})
    for res in results:
        m = agg[res["model"]]
        m["total"] += 1
        if res["success"]:
            m["latency_sum"] += res["latency_seconds"]
            m["latency_count"] += 1
            if res["tokens_per_sec"] is not None:
                m["tps_sum"] += res["tokens_per_sec"]
                m["tps_count"] += 1
    for mod in MODELS:
        d = agg[mod]
        n_ok = d["latency_count"]
        avg_lat = d["latency_sum"] / n_ok if n_ok else 0
        avg_tps = d["tps_sum"] / d["tps_count"] if d["tps_count"] else None
        vram = vram_by_model.get(mod)
        print(f"{MODEL_LABELS[mod]:<24} | 평균 {avg_lat:.2f}초 ({n_ok}/{d['total']}건) | "
              f"VRAM {vram:.1f} GiB | {avg_tps:.1f} tok/s" if avg_tps else "측정 이상")
    print(f"* 원본 데이터: {OUTPUT_FILE}")
    print("=" * 100 + "\n")


def run_benchmark():
    contexts = get_contexts()
    assert contexts, f"컨텍스트 파일 비어 있음: {CONTEXTS_FILE}"

    results = []
    vram_by_model = {}
    for model in MODELS:
        print(f"\nBenchmarking model: {model}")
        warmup_model(model)
        v = get_vram_used_gib()
        vram_by_model[model] = v if v is not None else 0.0

        for entry in contexts:
            prompt = f"[검색된 암묵지]:\n{entry['context_text']}\n\n질문: {entry['query']}"
            ans, lat, metrics, fb = call_ollama(model, prompt)
            ok = ans is not None
            tps = tokens_per_sec(metrics) if ok else None
            results.append({
                "model": model,
                "query_id": entry["query_id"],
                "query": entry["query"],
                "retrieved": entry["retrieved"],
                "injected_context": entry["context_text"],
                "generated_answer": ans,
                "latency_seconds": round(lat, 2),
                "success": ok,
                "tokens_per_sec": round(tps, 2) if tps is not None else None,
                "think_fallback": fb,
                "ollama_metrics": metrics,
            })
            print(f"    - {entry['query_id']}: {lat:.1f}초, "
                  f"{f'{tps:.1f} tok/s' if tps else 'n/a'}, fb={fb}")
            v = get_vram_used_gib()
            if v is not None:
                vram_by_model[model] = max(vram_by_model[model], v)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {OUTPUT_FILE}")
    print_quantitative_report(results, vram_by_model)


if __name__ == "__main__":
    run_benchmark()
