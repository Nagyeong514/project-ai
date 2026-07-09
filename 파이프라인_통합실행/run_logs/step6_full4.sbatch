#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=step6_full4
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_full4_%j.log
# STEP6 전체 재실행 (2026-07-08, STEP5 재설계 후 첫 실전 판정).
# 입력: step6_adapter_input/motion_run2 (재설계 후 tacit_json — 후보=윈도우 결박).
# 출력: STEP6_품질검증/output/motion_run2 (motion_run1은 구버전 before 비교용 보존).
# 제출 전제: score_step5.py 8/8 확인 + step6_adapter.py --output motion_run2 완료.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM으로_암묵지_후보생성/output/step6_adapter_input/motion_run2"
OUT="output/motion_run2"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 전체 재실행 (재설계 STEP5 산출물, motion_run2) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
