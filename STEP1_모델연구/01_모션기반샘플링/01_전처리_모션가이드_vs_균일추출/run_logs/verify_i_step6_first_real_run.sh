#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=verify_i_step6
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/verify_i_step6_first_real_run.log
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM으로_암묵지_후보생성/output/step6_adapter_input/motion_run1"
OUT="output/motion_run1"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 첫 실물 실행 (모션가이드 4클립, 어댑터 경유) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
