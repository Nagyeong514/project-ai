#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J port4F2
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/port4F2_%j.log

# 4-F 재작업 진단 런(2026-07-06): timestamp 범위강제 제거 → 동결 원인 격리.
# CLIP1만. 통과: 청크 40-78, 120-158(RAM·케이블 실동작)이 다시 관찰을 냄(백업 대비 복원).
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/STEP4_YOLO_VLM관찰"
"$PY" run_step4.py --config "$CFG" --video-id "CLIP1_정상조립과정.mp4" || exit 1
echo "PORT4F2 DONE"
