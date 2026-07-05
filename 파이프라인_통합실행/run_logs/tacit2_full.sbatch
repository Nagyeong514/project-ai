#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J tacit2_full
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/tacit2_full_%j.log

# tacit2(1) 풀런 (2026-07-05): 4클립 전 스테이지. 스모크(CLIP1) 산출물은 skip-if-exists로 재사용.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"

cd "/home/ai_user/team_a2/members/안나경/project-ai/tacit2(1)/tacit2"
PY=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/.venv/bin/python3

nvidia-smi --query-gpu=index,name,memory.used --format=csv
"$PY" run_pipeline.py
rc=$?
echo "FULL exit=$rc"
[ $rc -ne 0 ] && exit $rc

echo "=================== EVAL (tacit2) ==================="
"$PY" eval/score.py
echo "ALL DONE"
