#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=verify_f_c4score
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/verify_f_clip4_retry_and_score.log
set -euo pipefail
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:/opt/ohpc/pub/apps/cuda/13.0/lib64:${LD_LIBRARY_PATH:-}"
cd "/home/ai_user/team_a2/members/안나경/project-ai"

PY="파이프라인_통합실행/.venv/bin/python3"
CFG="00_사전연구/03_전처리_모션가이드_vs_균일추출/config.motion_clip1_test.yaml"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== STEP5 재시도: CLIP4_재부팅및BIOS.mp4 (신규 프로세스, GPU 메모리 완전 초기화 상태) ==================="
"$PY" STEP5_LLM으로_암묵지_후보생성/run_step5.py --config "$CFG" --video-id "CLIP4_재부팅및BIOS.mp4"

echo "=================== SCORE (motion, --trace) ==================="
"$PY" 파이프라인_통합실행/run_logs/score_step5.py --trace

echo "=================== ALL DONE ==================="
