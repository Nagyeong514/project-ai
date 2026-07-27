# -*- coding: utf-8 -*-
"""
실DB 벤치마크 — Gemma-3-12B(Ollama Q4_K_M) 단독 재실행 (2026-07-12).

evaluate_llm_realdb.py(3종 비교 원본)·evaluate_llm_realdb_qwen3_14b.py(Qwen 단독판)는
무수정 보존 — 이 파일이 Gemma-3-12B 전용 변형이다(Qwen 단독판과 동일 구조).
동일 조건: 같은 SYSTEM_PROMPT, 같은 realdb_contexts.json, 같은 Ollama Q4_K_M 스택,
같은 측정 방식(Latency/VRAM/Throughput). Gemma는 thinking 모델이 아니므로 think
파라미터를 보내지 않는다(Qwen 단독판과의 유일한 차이).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict

import requests

OLLAMA_API = "http://localhost:11434/api/chat"
MODELS = ["gemma3:12b"]
MODEL_LABELS = {"gemma3:12b": "Gemma-3-12B Q4_K_M"}
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXTS_FILE = os.path.join(EVAL_DIR, "realdb_contexts.json")
OUTPUT_FILE = os.path.join(EVAL_DIR, "llm_benchmark_results_realdb_gemma3_12b.json")
API_TIMEOUT = 300
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
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    start = time.time()
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API Error for model {model}: {e}")
        return None, 0, {}
    lat = time.time() - start
    body = r.json()
    answer = body.get("message", {}).get("content", "")
    metrics = {k: body.get(k) for k in (
        "eval_count", "eval_duration", "prompt_eval_count",
        "prompt_eval_duration", "total_duration", "load_duration")}
    return answer, lat, metrics


def tokens_per_sec(metrics):
    count, dur_ns = metrics.get("eval_count"), metrics.get("eval_duration")
    if not count or not dur_ns:
        return None
    return count / (dur_ns / 1e9)


def warmup_model(model):
    print(f"  [WARMUP] {model} 로딩 중...")
    ans, lat, _ = call_ollama(model, WARMUP_QUERY)
    print(f"  [WARMUP] {'완료 (%.1f초)' % lat if ans else '경고: 실패'}")


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
            ans, lat, metrics = call_ollama(model, prompt)
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
                "ollama_metrics": metrics,
            })
            print(f"    - {entry['query_id']}: {lat:.1f}초, "
                  f"{f'{tps:.1f} tok/s' if tps else 'n/a'}")
            v = get_vram_used_gib()
            if v is not None:
                vram_by_model[model] = max(vram_by_model[model], v)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {OUTPUT_FILE}")

    agg = defaultdict(lambda: {"lat": [], "tps": []})
    for res in results:
        if res["success"]:
            agg[res["model"]]["lat"].append(res["latency_seconds"])
            if res["tokens_per_sec"]:
                agg[res["model"]]["tps"].append(res["tokens_per_sec"])
    for mod in MODELS:
        d = agg[mod]
        print(f"\n{MODEL_LABELS[mod]} | 평균 {sum(d['lat'])/len(d['lat']):.2f}초 "
              f"({len(d['lat'])}/{len(contexts)}건) | VRAM {vram_by_model[mod]:.1f} GiB | "
              f"{sum(d['tps'])/len(d['tps']):.1f} tok/s")


if __name__ == "__main__":
    run_benchmark()
