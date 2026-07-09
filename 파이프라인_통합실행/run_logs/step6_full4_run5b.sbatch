#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=step6_full4_r5b
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_full4_run5b_%j.log
# STEP6 재판정 run5b (2026-07-09) — Gate A judge 기준 개정 검증.
# 개정: ① same은 구체 패턴·수치·절차가 매뉴얼에 명시된 경우만(주제 존재만으로 금지)
#       ② judge 입력에 reasoning 참고 블록 추가(황4백5 같은 구체 패턴 대조 가능하게)
# 입력: motion_run5 어댑터 산출물 그대로(후보 31건 재사용 — STEP5 무변경).
# run5(output/motion_run5)는 보존, 결과는 motion_run5b에.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM으로_암묵지_후보생성/output/step6_adapter_input/motion_run5"
OUT="output/motion_run5b"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 재판정 run5b (Gate A 기준 개정) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
