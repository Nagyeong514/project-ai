#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -J realdb_gemma3_12b
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용/logs/realdb_gemma3_12b_%j.log
# Gemma-3-12B(Ollama Q4_K_M) 단독 실DB 답변 벤치 — run_benchmark_realdb_n5.sh의 Qwen 전용판.
# 사전 조건: 로그인 노드에서 ollama pull gemma3:12b 완료(OLLAMA_MODELS 공유 경로).
set -uo pipefail

EVAL_DIR="/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용"
export OLLAMA_MODELS=/home/ai_user/team_a2/.ollama/models
export PATH="/home/ai_user/team_a2/.local/ollama/bin:$PATH"
LOG_DIR="$EVAL_DIR/logs"; mkdir -p "$LOG_DIR"

echo "=== preflight ==="
echo "[node] $(hostname)"
nvidia-smi -L || { echo "FATAL: GPU 안 보임"; exit 1; }
python3 -c "import requests, sys; print('[python]', sys.executable)" || { echo "FATAL: requests 없음"; exit 1; }
command -v ollama || { echo "FATAL: ollama 없음"; exit 1; }
ls "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/gemma3" || { echo "FATAL: qwen3 태그 없음 — pull 미완료"; exit 1; }
python3 - <<'EOF' || { echo "FATAL: realdb_contexts.json 이상"; exit 1; }
import json
p = "/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용/realdb_contexts.json"
e = json.load(open(p, encoding="utf-8"))
assert len(e) == 7 and all((x.get("context_text") or "").strip() for x in e)
print("OK: 컨텍스트 7건")
EOF

echo "=== ollama serve ==="
ollama serve > "$LOG_DIR/ollama_serve_gemma3_12b.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
for i in $(seq 1 30); do
    curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null && break; sleep 2
done
curl -s --max-time 5 http://localhost:11434/api/tags | grep -q '"gemma3:12b"' || { echo "FATAL: gemma3:12b 태그 없음"; exit 1; }

echo "=== 벤치 실행 ==="
python3 "$EVAL_DIR/evaluate_llm_realdb_gemma3_12b.py"
RC=$?
echo "[exit] $RC"

echo "=== 산출물 검증 ==="
python3 - <<'EOF'
import json
p = "/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용/llm_benchmark_results_realdb_gemma3_12b.json"
rows = json.load(open(p, encoding="utf-8"))
assert len(rows) == 7, f"결과 수 이상: {len(rows)}"
empty = [r["query_id"] for r in rows if not (r["generated_answer"] or "").strip()]
assert not empty, f"빈 답변: {empty}"
print(f"OK: {len(rows)}건, 빈 답변 없음")
EOF
echo "REALDB_QWEN3_14B_DONE"
exit $RC
