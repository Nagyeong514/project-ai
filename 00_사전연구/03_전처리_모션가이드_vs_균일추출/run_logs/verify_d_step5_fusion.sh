#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=verify_d_step5
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/verify_d_step5_fusion.log
set -euo pipefail
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:/opt/ohpc/pub/apps/cuda/13.0/lib64:${LD_LIBRARY_PATH:-}"
cd "/home/ai_user/team_a2/members/안나경/project-ai"

PY="파이프라인_통합실행/.venv/bin/python3"
CFG="00_사전연구/03_전처리_모션가이드_vs_균일추출/config.motion_clip1_test.yaml"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

for VID in "CLIP1_정상조립과정.mp4" "CLIP2.mp4" "CLIP3_재부팅시도 전까지.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP5: $VID ==================="
  "$PY" STEP5_LLM으로_암묵지_후보생성/run_step5.py --config "$CFG" --video-id "$VID"
done

echo "=================== SCORE (motion, --trace) ==================="
"$PY" 파이프라인_통합실행/run_logs/score_step5.py --trace

echo "=================== ALL DONE ==================="
