#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -J step8_smoke_qwen3
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/STEP8_RAG서비스/logs/smoke_qwen3_14b_%j.log
# STEP8 /ask 스모크 (2026-07-12): llm_model 기본값 qwen3:14b 교체 + app.py 조건부 think:false
# 코드 경로의 실호출 검증 1건. 실서비스 기동 절차 그대로(ollama serve → uvicorn → curl).
set -uo pipefail

DIR="/home/ai_user/team_a2/members/안나경/project-ai/STEP8_RAG서비스"
export OLLAMA_MODELS=/home/ai_user/team_a2/.ollama/models   # qwen3:14b 있는 공유 경로
export PATH="/home/ai_user/team_a2/.local/ollama/bin:$PATH"
mkdir -p "$DIR/logs"
cd "$DIR"

echo "=== preflight ==="
nvidia-smi -L || { echo "FATAL: GPU 없음"; exit 1; }
ls "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/qwen3" || { echo "FATAL: qwen3 태그 없음"; exit 1; }
[ -x .venv/bin/uvicorn ] || { echo "FATAL: .venv/uvicorn 없음"; exit 1; }

echo "=== ollama serve ==="
ollama serve > logs/ollama_smoke.log 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID $UVICORN_PID 2>/dev/null' EXIT
for i in $(seq 1 30); do curl -s --max-time 2 localhost:11434/api/tags >/dev/null && break; sleep 2; done
curl -s --max-time 5 localhost:11434/api/tags | grep -q '"qwen3:14b"' || { echo "FATAL: qwen3:14b 태그 없음"; exit 1; }

echo "=== uvicorn 기동 ==="
.venv/bin/uvicorn app:app --port 8000 > logs/uvicorn_smoke.log 2>&1 &
UVICORN_PID=$!
for i in $(seq 1 60); do curl -s --max-time 2 localhost:8000/docs >/dev/null && break; sleep 2; done
curl -s --max-time 2 localhost:8000/docs >/dev/null || { echo "FATAL: 앱 기동 실패"; tail -30 logs/uvicorn_smoke.log; exit 1; }

echo "=== /ask 스모크 (모델 워밍업 겸 2회) ==="
for q in "램을 뺄 때 주의할 점이 있나요?" "메모리 채널이 정상인지 어떻게 확인하나요?"; do
  echo "--- Q: $q"
  T0=$(date +%s.%N)
  RESP=$(curl -s --max-time 180 -X POST localhost:8000/ask -H 'Content-Type: application/json' \
    -d "{\"question\": \"$q\"}")
  T1=$(date +%s.%N)
  echo "$RESP" | python3 -c "
import json,sys
r=json.load(sys.stdin)
ans=r.get('answer','')
assert ans.strip(), '빈 답변'
assert '<think>' not in ans, 'think 블록 유출!'
print('[answer]', ans)
print('[sources]', [s.get('id') for s in r.get('sources',[])])
"
  echo "[e2e] $(echo "$T1 - $T0" | bc)초"
done
echo "SMOKE_OK"
