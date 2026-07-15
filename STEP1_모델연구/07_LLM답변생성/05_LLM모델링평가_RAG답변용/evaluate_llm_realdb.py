# -*- coding: utf-8 -*-
"""
실DB 벤치마크 2단계 — STEP7 실제 벡터 DB 검색 결과(realdb_contexts.json)를 컨텍스트로
3종 모델(Ollama Q4_K_M)을 재벤치마크한다. evaluate_llm.py(합성 골든셋 버전)는 원본
보존을 위해 무수정 — 이 파일이 실DB 버전 전용이다.

실행 전 필수:
1. 로그인 노드에서 retrieve_contexts_realdb.py 먼저 실행 → realdb_contexts.json 생성
2. GPU 노드(n5)에서 Ollama 서버 기동 후 이 스크립트 실행:
   srun -p RTX6000 -w n5 --gres=gpu:1 bash run_benchmark_realdb_n5.sh

측정 지표(자동, 이번 실험은 이 3개만):
- 평균 Latency(round-trip, 성공 호출만 집계) — 기존 방식 유지
- VRAM 점유(모델 전환 시 keep_alive=0 언로드 → 단독 점유, nvidia-smi 폴링 최댓값) — 기존 방식 유지
- Throughput(tokens/sec) — 신규: Ollama 응답의 eval_count / eval_duration(ns) 기반,
  케이스별 측정 후 모델별 평균

이번 실험에서 채점하지 않는 것(설계 결정 2026-07-11): 지시 이행률, 사실적 충실도 —
휴리스틱 채점 전부 스킵. 대신 generated_answer 원문과 검색 메타(retrieved id/score)를
JSON에 전부 저장하므로 정성 평가는 사람이 직접 한다.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict

import requests

# 설정
OLLAMA_API = "http://localhost:11434/api/chat"
# ★ 태그 함정(2026-07-07 실측): qwen3-vl 기본 태그 "qwen3-vl:8b"는 Thinking 변형 —
# 반드시 -instruct 명시(evaluate_llm.py 상단 주석 참고).
MODELS = ["qwen3-vl:8b-instruct", "llama3.1:8b", "gemma3:4b"]
MODEL_LABELS = {
    "qwen3-vl:8b-instruct": "Qwen3-VL-8B-Instruct Q4_K_M",
    "llama3.1:8b": "Llama-3.1-8B Q4_K_M",
    "gemma3:4b": "Gemma-3-4B Q4_K_M",
}
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXTS_FILE = os.path.join(EVAL_DIR, "realdb_contexts.json")
OUTPUT_FILE = os.path.join(EVAL_DIR, "llm_benchmark_results_realdb.json")
API_TIMEOUT = 180
WARMUP_QUERY = "안녕하세요, 준비됐나요? 한 문장으로만 답하세요."  # 로드 트리거 전용 — 집계 제외

# SYSTEM_PROMPT는 기존 실험과 동일(비교 가능성 유지). 컨텍스트가 [핵심/이유/절차/출처]
# 템플릿 조립 결과라는 점만 다르다.
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
    """전체 GPU memory.used 합계(GiB) 스냅샷 — evaluate_llm.py와 동일 방식."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        used_mib = sum(int(x) for x in out.splitlines() if x.strip())
        return used_mib / 1024
    except Exception as e:
        print(f"[WARN] nvidia-smi 조회 실패(로그인 노드에서 실행 중이거나 GPU 없음?): {e}")
        return None


def call_ollama(model, prompt):
    """반환: (answer, latency_seconds, metrics). metrics는 Ollama 응답의 토큰/시간 원시값
    (eval_count/eval_duration 등, duration 단위는 전부 나노초)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API Error for model {model}: {e}")
        return None, 0, {}
    end_time = time.time()

    body = response.json()
    answer = body.get("message", {}).get("content", "")
    metrics = {k: body.get(k) for k in (
        "eval_count", "eval_duration",
        "prompt_eval_count", "prompt_eval_duration",
        "total_duration", "load_duration",
    )}
    return answer, end_time - start_time, metrics


def tokens_per_sec(metrics):
    """생성 처리량(tokens/sec) = eval_count / (eval_duration ns → s). 값 없으면 None."""
    count = metrics.get("eval_count")
    dur_ns = metrics.get("eval_duration")
    if not count or not dur_ns:
        return None
    return count / (dur_ns / 1e9)


def unload_model(model):
    """모델 전환 전 keep_alive=0 언로드 — VRAM 누적 측정 방지(evaluate_llm.py 버그수정 유지)."""
    try:
        requests.post(OLLAMA_API, json={"model": model, "messages": [], "keep_alive": 0}, timeout=30)
        time.sleep(3)
    except requests.exceptions.RequestException as e:
        print(f"  [UNLOAD] 경고: {model} 언로드 실패(VRAM 측정이 누적될 수 있음): {e}")


def warmup_model(model):
    """로딩(디스크→VRAM) 시간이 첫 케이스 latency에 섞이는 걸 막는 워밍업 — 집계 제외."""
    print(f"  [WARMUP] {model} 로딩 중...")
    ans, lat, _ = call_ollama(model, WARMUP_QUERY)
    if ans is None:
        print(f"  [WARMUP] 경고: {model} 워밍업 실패 — 이후 요청도 실패할 가능성 있음")
    else:
        print(f"  [WARMUP] 완료 ({lat:.1f}초)")


def print_quantitative_report(results, vram_by_model):
    print("\n" + "=" * 100)
    print(" [자동 산출] 실DB 기반 LLM 벤치마크 — Latency / VRAM / Throughput")
    print(" (지시 이행률/충실도 채점은 이번 실험에서 스킵 — 답변 원문은 JSON에 저장됨, 정성 평가용)")
    print("=" * 100)

    agg = defaultdict(lambda: {
        "latency_sum": 0.0, "latency_count": 0,
        "tps_sum": 0.0, "tps_count": 0,
        "total": 0,
    })
    for res in results:
        m = agg[res["model"]]
        m["total"] += 1
        if res["success"]:
            m["latency_sum"] += res["latency_seconds"]
            m["latency_count"] += 1
            if res["tokens_per_sec"] is not None:
                m["tps_sum"] += res["tokens_per_sec"]
                m["tps_count"] += 1

    header = f"{'모델명':<28} | {'평균 Latency':<14} | {'VRAM':<10} | {'평균 Throughput'}"
    print(header)
    print("-" * 100)
    for mod in MODELS:
        label = MODEL_LABELS.get(mod, mod)
        if mod in agg:
            data = agg[mod]
            n_ok = data["latency_count"]
            avg_lat = data["latency_sum"] / n_ok if n_ok else 0
            avg_tps = data["tps_sum"] / data["tps_count"] if data["tps_count"] else None
            n_fail = data["total"] - n_ok
            vram = vram_by_model.get(mod)
            vram_str = f"{vram:.1f} GiB" if vram is not None else "측정불가"
            tps_str = f"{avg_tps:.1f} tok/s ({data['tps_count']}건)" if avg_tps is not None else "측정불가"
            fail_note = f" (실패 {n_fail}건 제외)" if n_fail else ""
            print(f"{label:<28} | {avg_lat:.2f}초{fail_note:<10} | {vram_str:<10} | {tps_str}")
        else:
            print(f"{label:<28} | 측정 실패 | 측정 실패 | 측정 실패")
    print("-" * 100)
    print("* 컨텍스트 = STEP7 실데이터 벡터 DB(모션 run6 accept 6건) 검색 결과(top_k=3, sim>=0.40).")
    print("* 정성 평가는 저장된 JSON의 generated_answer를 직접 읽고 판단할 것.")
    print(f"* 원본 데이터: {OUTPUT_FILE}")
    print("=" * 100 + "\n")


def run_benchmark():
    contexts = get_contexts()
    if not contexts:
        print(f"경고: 컨텍스트 파일이 비어 있음 — retrieve_contexts_realdb.py를 먼저 실행하세요: {CONTEXTS_FILE}")
        return

    results = []
    vram_by_model = {}

    prev_model = None
    for model in MODELS:
        print(f"\nBenchmarking model: {model}")

        if prev_model:
            unload_model(prev_model)
        prev_model = model

        warmup_model(model)
        v = get_vram_used_gib()
        vram_by_model[model] = v if v is not None else 0.0

        def _bump_vram():
            v = get_vram_used_gib()
            if v is not None:
                vram_by_model[model] = max(vram_by_model[model], v)

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
                "ollama_metrics": metrics,  # eval_count/eval_duration 등 원시값(ns)
            })
            tps_str = f"{tps:.1f} tok/s" if tps is not None else "n/a"
            print(f"    - {entry['query_id']}: {lat:.1f}초, {tps_str}")
            _bump_vram()

        time.sleep(3)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Benchmark 데이터 저장 완료: {OUTPUT_FILE}")

    print_quantitative_report(results, vram_by_model)


if __name__ == "__main__":
    run_benchmark()
