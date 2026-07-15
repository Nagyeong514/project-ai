#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J llmselect
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_LLM선정/run_logs/llmselect_%x_%j.log
# 융합 LLM 선정 벤치 — 모델 1개 실행(웜업1+본3회). 모델별 '순차' 제출(동시 실행 금지).
# 제출: sbatch --export=ALL,MODEL=qwen3vl -J llmsel_qwen3vl llmselect.sbatch
#       (다음 모델은 --dependency=afterany:<이전잡> 로 체인)
# 정밀도: bench_config.yaml precision(fp16 1차) — 웜업 OOM이면 스크립트가 스스로 중단,
# 그 시점에 셋 다 nf4로 전환 재실행(일부만 nf4 금지).
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
cd "$ROOT/00_사전연구/03_LLM선정"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"$PY" run_bench.py --model "${MODEL:?MODEL 미지정}" || exit 1
echo "LLMSELECT_${MODEL}_DONE"
