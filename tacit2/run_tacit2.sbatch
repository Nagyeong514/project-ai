#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J tacit2_standalone
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/tacit2/tacit2_standalone_%j.log

set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/tacit2/config.yaml"

cd "$ROOT/tacit2"
"$PY" run_pipeline.py --config "$CFG"
