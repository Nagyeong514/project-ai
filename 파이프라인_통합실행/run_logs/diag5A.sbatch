#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J diag5A
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag5A_%j.log
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
PY=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/.venv/bin/python3
"$PY" /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag_5a_oom.py
echo "DIAG5A DONE"
