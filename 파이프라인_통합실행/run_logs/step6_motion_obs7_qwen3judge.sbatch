#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH --job-name=step6_motion_obs7_qwen3judge
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_motion_obs7_qwen3judge_%j.log
# STEP6 재판정 run7 (2026-07-12): Judge 모델 교체(Qwen2.5-14B→Qwen3-14B) 후 동일 입력(모션 24건) 재판정. Gate 규칙 무수정 — 판정 차이는 순수 모델 효과. run6와 비교용.
# 입력: run6와 동일(step6_adapter_input/motion_obs_run1)
# 출력: STEP6_품질검증/output/motion_obs_run7_qwen3judge — run6(Qwen2.5 judge) 결과는 무수정 보존.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM_암묵지_후보생성/output/step6_adapter_input/motion_obs_run1"
OUT="output/motion_obs_run7_qwen3judge"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 판정 (motion_obs_run7_qwen3judge, 후보 24건) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
