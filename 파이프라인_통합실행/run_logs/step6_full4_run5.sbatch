#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=step6_full4_r5
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_full4_run5_%j.log
# STEP6 재판정 — motion_run5 (2026-07-09, STEP5 Pass 2 응집 산출물 첫 판정).
# 입력: step6_adapter_input/motion_run5 (Pass 2 응집 후 4클립, score_step5 채점 후 생성).
# 비교 기준: motion_run2(응집 전 STEP5 2268, 침묵트랙 전 = 1/19/15),
#            motion_run4(run3 산출물 + 침묵트랙 = 9/17/9). 둘 다 보존.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM으로_암묵지_후보생성/output/step6_adapter_input/motion_run5"
OUT="output/motion_run5"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 재판정 (Pass 2 응집 산출물, motion_run5) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
