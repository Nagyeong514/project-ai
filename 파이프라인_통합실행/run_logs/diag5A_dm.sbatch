#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J diag5Adm
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag5Adm_%j.log
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/.venv/bin/python3
"$PY" /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag_5a_devicemap.py
echo "DIAG5ADM DONE"
