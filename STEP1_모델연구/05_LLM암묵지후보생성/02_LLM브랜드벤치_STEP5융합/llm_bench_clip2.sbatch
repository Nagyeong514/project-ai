#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J llm_bench
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/06_LLM브랜드벤치_STEP5융합/run_logs/llm_bench_clip2_%j.log
# STEP5 융합 LLM 2종 비교(클립2) — Qwen3-14B / Mistral-Nemo-12B, 전부 nf4 (Gemma 제외 확정).
# 통제: 입력 고정(frozen windows) + 생성 파라미터 통일 + 모델당 스모크 후 2회.
# 사전 조건: 모델 사전 다운로드(로그인 노드) 완료.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/00_사전연구/06_LLM브랜드벤치_STEP5융합"  # 2026-07-09 사전연구 이관
"$PY" llm_brand_bench_clip2.py --config "$CFG" || exit 1
echo "LLM_BENCH DONE"
