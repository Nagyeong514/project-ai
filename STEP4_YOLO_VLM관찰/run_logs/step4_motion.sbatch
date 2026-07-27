#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step4_motion
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/STEP4_YOLO_VLM관찰/run_logs/step4_motion_%j.log

# STEP4 모션가이드 클립 실행. 추가 인자는 run_step4_motion.py로 그대로 전달:
#   sbatch step4_motion.sbatch --limit 1   # 테스트(첫 클립 1개)
#   sbatch step4_motion.sbatch             # 전체 15클립(완료분 skip)
# (2026-07-13 STEP4_YOLO_VLM관찰_모션O → 원본 STEP4 병합에 따라 경로/스크립트명 갱신.)
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"

cd "/home/ai_user/team_a2/members/안나경/project-ai/STEP4_YOLO_VLM관찰"
PY=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/.venv/bin/python3

nvidia-smi --query-gpu=index,name,memory.used --format=csv
"$PY" run_step4_motion.py "$@"
rc=$?
echo "STEP4_MOTION exit=$rc"
exit $rc
