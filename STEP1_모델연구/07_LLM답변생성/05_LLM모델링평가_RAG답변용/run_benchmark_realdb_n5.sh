#!/bin/bash
# 실DB 벤치마크 — GPU 노드(n5)에서 Ollama 서버 기동 → evaluate_llm_realdb.py 실행 → 결과 검증.
# run_benchmark_n5.sh의 복사본(원본 보존). 차이: 컨텍스트 JSON 프리플라이트 추가,
# 실행 대상이 evaluate_llm_realdb.py, 산출물 검증 기준이 3모델 × 질의 7건 = 21건.
#
# 사전 조건: 로그인 노드에서 retrieve_contexts_realdb.py 먼저 실행(realdb_contexts.json 생성).
# 로그인 노드에서: srun -p RTX6000 -w n5 --gres=gpu:1 bash run_benchmark_realdb_n5.sh
# (--pty 금지: 에이전트/자동화에서는 인터랙티브 셸을 못 쓴다 — CLAUDE.md §2)
set -uo pipefail

EVAL_DIR="/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용"
export OLLAMA_MODELS=/home/ai_user/team_a2/.ollama/models   # 셋 다 있는 경로 (.local/ollama/data 아님)
export PATH="/home/ai_user/team_a2/.local/ollama/bin:$PATH"
LOG_DIR="$EVAL_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== preflight (5초 안에 죽을 수 있는 검증 먼저) ==="
echo "[node] $(hostname)"
nvidia-smi -L || { echo "FATAL: GPU 안 보임 — 로그인 노드에서 실행했거나 gres 미할당"; exit 1; }
python3 -c "import requests, sys; print('[python]', sys.executable)" || { echo "FATAL: n5에 requests 없음"; exit 1; }
ls "$OLLAMA_MODELS/manifests/registry.ollama.ai/library" || { echo "FATAL: OLLAMA_MODELS 경로 이상"; exit 1; }
command -v ollama || { echo "FATAL: ollama 바이너리 없음"; exit 1; }
# 실DB 컨텍스트 JSON 존재 + 7건 + 빈 컨텍스트 없음(1단계 산출물 검증)
python3 - <<'EOF' || { echo "FATAL: realdb_contexts.json 이상 — retrieve_contexts_realdb.py 먼저 실행"; exit 1; }
import json
p = "/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용/realdb_contexts.json"
entries = json.load(open(p, encoding="utf-8"))
assert len(entries) == 7, f"컨텍스트 수 이상: {len(entries)} (기대 7)"
assert all((e.get("context_text") or "").strip() for e in entries), "빈 컨텍스트 존재"
print(f"OK: 컨텍스트 {len(entries)}건")
EOF

echo "=== ollama serve 기동 ==="
ollama serve > "$LOG_DIR/ollama_serve_realdb.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

for i in $(seq 1 30); do
    if curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null; then break; fi
    sleep 2
done
TAGS=$(curl -s --max-time 5 http://localhost:11434/api/tags)
echo "[tags] $TAGS"
for m in "qwen3-vl:8b-instruct" "llama3.1:8b" "gemma3:4b"; do
    echo "$TAGS" | grep -q "\"$m\"" || { echo "FATAL: 모델 태그 누락: $m"; exit 1; }
done

echo "=== 벤치마크 본 실행 ==="
python3 "$EVAL_DIR/evaluate_llm_realdb.py"
RC=$?
echo "[evaluate_llm_realdb.py exit] $RC"

echo "=== 산출물 검증 (exit 0 + 빈 결과 방지 — 체크리스트 5단계) ==="
python3 - <<'EOF'
import json
p = "/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용/llm_benchmark_results_realdb.json"
rows = json.load(open(p, encoding="utf-8"))
models = {r["model"] for r in rows}
assert len(models) == 3, f"모델 수 이상: {models}"
# 질의 7건 × 3모델 = 21건
per = {m: sum(1 for r in rows if r["model"] == m) for m in models}
assert all(v == 7 for v in per.values()), f"모델별 결과 수 이상: {per}"
empty = [(r["model"], r["query_id"]) for r in rows if not (r["generated_answer"] or "").strip()]
assert not empty, f"빈 답변 존재: {empty}"
no_tps = [(r["model"], r["query_id"]) for r in rows if r["success"] and r["tokens_per_sec"] is None]
assert not no_tps, f"성공 호출인데 throughput 누락: {no_tps}"
print(f"OK: {len(rows)}건, 모델별 {per}, 빈 답변 없음, throughput 전건 산출")
EOF
exit $RC
