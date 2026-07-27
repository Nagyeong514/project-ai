#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH --job-name=step6_motion_obs6
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_motion_obs6_%j.log
# STEP6 재판정 run6 (2026-07-10, 최종): Gate A 규칙5 핀포인트 2건(언급만 된 조작의 수행방법=delta, 막연 우려 배제 예시). run5와 비교용.
# 입력: step6_adapter_input/motion_obs_run6 (tacit_json_motion_obs 변환본, CLIP2 transcript 수정판)
# 출력: STEP6_품질검증/output/motion_obs_run6 — 균일 기반 run5(31건, accept7/hold19/reject5)와 비교용.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM_암묵지_후보생성/output/step6_adapter_input/motion_obs_run1"
OUT="output/motion_obs_run6"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 판정 (motion_obs_run6, 후보 24건) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
