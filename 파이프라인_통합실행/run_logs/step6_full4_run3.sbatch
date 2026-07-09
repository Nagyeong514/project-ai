#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=step6_full4_r3
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_full4_run3_%j.log
# STEP6 재판정 (2026-07-08, 접지 프롬프트+재시도 정책 개정 후 — motion_run3).
# 입력: step6_adapter_input/motion_run3 (잡 2289 산출물). motion_run2는 전/후 비교용 보존.
# 제출 전제: score_step5.py 8/8 확인 + step6_adapter.py --output motion_run3 완료.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM으로_암묵지_후보생성/output/step6_adapter_input/motion_run3"
OUT="output/motion_run3"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 재판정 (접지 개정 STEP5 산출물, motion_run3) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
