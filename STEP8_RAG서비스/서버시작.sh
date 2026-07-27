#!/bin/bash
# STEP8_RAG서비스 서버 시작 스크립트 (서시은 voice-rag v2 서버시작.sh 이식, 2026-07-08)
# 사용법: bash 서버시작.sh [GPU노드]   (기본: n5, 비어있는지 sinfo로 먼저 확인)
# 종료할 때: fuser -k 8001/tcp && scancel <JOBID>
set -e
NODE=${1:-n5}
cd "$(dirname "$0")"

echo "[1/3] GPU 노드($NODE) 할당..."
salloc -p RTX6000 -w "$NODE" --gres=gpu:1 --no-shell
JOBID=$(squeue -u "$(whoami)" -h -o "%i %N" | awk -v n="$NODE" '$2==n{print $1; exit}')
echo "  JOBID=$JOBID"

echo "[2/3] ollama 서버 기동 (qwen2.5:14b, VRAM 상주)..."
nohup srun --jobid="$JOBID" --overlap bash -c \
  'export LD_LIBRARY_PATH=$HOME/.local/ollama/lib/ollama OLLAMA_MODELS=$HOME/.local/ollama/data OLLAMA_HOST=0.0.0.0:11434; $HOME/.local/ollama/bin/ollama serve' \
  > ollama_node.log 2>&1 &
for i in $(seq 1 20); do
  curl -s -m 2 "http://$NODE:11434/api/tags" >/dev/null 2>&1 && break
  sleep 3
done

echo "[3/3] 앱 서버 기동 (포트 8001 — 8000은 다른 사용자가 점유 중이라 쓰면 안 됨)..."
OLLAMA_URL="http://$NODE:11434/api/chat" nohup .venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001 > server.log 2>&1 &
for i in $(seq 1 60); do
  grep -q "Uvicorn running" server.log 2>/dev/null && break
  sleep 3
done

echo ""
echo "완료! VS Code 포트 탭에서 8001 포워딩 후 http://localhost:8001 접속"
echo "GPU 반납: scancel $JOBID"
