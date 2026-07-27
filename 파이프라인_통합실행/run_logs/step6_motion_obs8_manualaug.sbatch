#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH --job-name=step6_motion_obs8_manualaug
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step6_motion_obs8_manualaug_%j.log
# STEP6 재판정 run8 (2026-07-14): 매뉴얼 코퍼스 §2-8(메모리 용량·채널 인식 확인 표준 절차) 추가
# + 벡터DB Qdrant→ChromaDB 전환(config.vector_backend=chroma, STEP7/8 표준 통일) 후 동일 입력
# (모션 24건) 재판정. 판정모델 Qwen3-14B.
# 목적: CLIP4_006(용량)·CLIP4_007(채널)이 Gate A same 으로 라우팅되는지 확인.
# ⚠️ 격리 주의: run8은 run7(Qdrant+옛 매뉴얼) 대비 (매뉴얼 + 벡터DB) 두 변수가 동시에 바뀜.
#    STEP7 실측상 Chroma≈Qdrant cosine 점수라 매뉴얼 효과가 지배적일 것으로 보나, 순수 매뉴얼
#    격리가 필요하면 별도 대조(Chroma+옛 매뉴얼) 필요. run6(Qwen2.5)·run7(Qwen3)은 보존.
set -euo pipefail
cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP6_암묵지_품질검증"

PY=".venv/bin/python3"
INPUT="../STEP5_LLM_암묵지_후보생성/output/step6_adapter_input/motion_obs_run1"
OUT="output/motion_obs_run8_manualaug"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP6 판정 (motion_obs_run8_manualaug, 후보 24건) ==================="
"$PY" main.py --input "$INPUT" --output_dir "$OUT"

echo "=================== ALL DONE ==================="
