#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step4_verify
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step4_verify_%j.log

# STEP4 구조 수정(YOLO fail-loud + VLM 청크 관찰) 검증 런 — 2026-07-05
# CLIP1(반복루프 회귀 확인) → CLIP4(이전 OOM/관찰 0건, fps 0.5 재추출본 115프레임)
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ai_user/team_a2/members/안나경/project-ai/STEP4_YOLO_VLM관찰
PY=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/.venv/bin/python3

nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=================== CLIP1 ==================="
"$PY" run_step4.py --config ../파이프라인_통합실행/config.yaml --video-id "CLIP1_정상조립과정.mp4"
echo "CLIP1 exit=$?"

echo "=================== CLIP4 ==================="
"$PY" run_step4.py --config ../파이프라인_통합실행/config.yaml --video-id "CLIP4_재부팅및BIOS.mp4"
echo "CLIP4 exit=$?"
