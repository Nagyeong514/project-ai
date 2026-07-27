#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH --job-name=step6_motion_obs9_qwen25judge
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_motion_obs9_qwen25judge_%j.log
# STEP6 재판정 run9 (2026-07-14): 판정자를 run6과 동일한 Qwen2.5-14B-Instruct 로 되돌려 재판정.
# 코퍼스는 run8과 동일(Chroma + §2-8 매뉴얼). 즉 run9 vs run8 은 '판정자'만 다르다(Qwen2.5 vs Qwen3)
# → ESC 오탐이 순수 판정자 효과임을 격리 검증. run9 vs run6 은 (매뉴얼 §2-8 + 벡터DB Chroma)만 다름.
# 목적: (1) ESC(CLIP4_004)가 same 으로 걸러지는지 (2) 채널(CLIP4_007) same 유지 확인.
# ⚠️ ESC는 코퍼스에 이미 있었고(§2-5, run6/run8 동일 검색 top=0.523) 오탐 원인은 Qwen3 판정자였음
#    → 코퍼스 추가 없음(중복 padding 지양). run9가 그 진단을 증명하는 실행.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_암묵지_품질검증"

export STEP6_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"   # run6 판정자
PY=".venv/bin/python3"
INPUT="../STEP5_LLM_암묵지_후보생성/output/step6_adapter_input/motion_obs_run1"
OUT="output/motion_obs_run9_qwen25judge"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 판정 (run9, Qwen2.5 판정자, Chroma+§2-8, 24건) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
