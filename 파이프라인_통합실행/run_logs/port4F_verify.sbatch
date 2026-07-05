#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J port4F
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/port4F_%j.log

# 이식 4-F 검증 런(2026-07-05): timestamp 지시 HH:MM:SS 통일 + 청크 구간 명시.
# 검증 대상: CLIP4(두 형식 공존 실측). 통과: raw_by_chunk 안 timestamp가 HH:MM:SS 단일 형식.

set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

cd "$ROOT/STEP4_YOLO_VLM관찰"
for VID in "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP4 $VID ==================="
  "$PY" run_step4.py --config "$CFG" --video-id "$VID" || exit 1
done
echo "PORT4F DONE"
