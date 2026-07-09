#!/bin/bash
# GPU 노드(n5)에서 Ollama 서버 기동 → evaluate_llm.py 실행 → 결과 검증까지 한 번에.
# 로그인 노드에서: srun -p RTX6000 -w n5 --gres=gpu:1 bash run_benchmark_n5.sh
# (--pty 금지: 에이전트/자동화에서는 인터랙티브 셸을 못 쓴다 — CLAUDE.md §2)
set -uo pipefail

EVAL_DIR="/home/ai_user/team_a2/members/안나경/project-ai/LLM_모델링_평가"
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

echo "=== ollama serve 기동 ==="
ollama serve > "$LOG_DIR/ollama_serve.log" 2>&1 &
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
python3 "$EVAL_DIR/evaluate_llm.py"
RC=$?
echo "[evaluate_llm.py exit] $RC"

echo "=== 산출물 검증 (exit 0 + 빈 결과 방지 — 체크리스트 5단계) ==="
python3 - <<'EOF'
import json
p = "/home/ai_user/team_a2/members/안나경/project-ai/LLM_모델링_평가/llm_benchmark_results.json"
rows = json.load(open(p, encoding="utf-8"))
models = {r["model"] for r in rows}
assert len(models) == 3, f"모델 수 이상: {models}"
# 골든레코드 9건 × (A+B) + C 1건 = 모델당 19건
per = {m: sum(1 for r in rows if r["model"] == m) for m in models}
assert all(v == 19 for v in per.values()), f"모델별 결과 수 이상: {per}"
empty = [(r["model"], r["scenario"]) for r in rows if not (r["generated_answer"] or "").strip()]
assert not empty, f"빈 답변 존재: {empty}"
print(f"OK: {len(rows)}건, 모델별 {per}, 빈 답변 없음")
EOF
exit $RC
