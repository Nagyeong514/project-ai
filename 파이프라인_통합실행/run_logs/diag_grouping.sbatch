#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J diag_grouping
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag_grouping_%j.log
# Pass 2 응집 진단 — CLIP4 스모크 병합 0건의 원인 규명(복불복 vs 구조적).
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" diag_grouping_clip4.py --config "$ROOT/파이프라인_통합실행/config.yaml" --trials 5 || exit 1
echo "DIAG_GROUPING DONE"
