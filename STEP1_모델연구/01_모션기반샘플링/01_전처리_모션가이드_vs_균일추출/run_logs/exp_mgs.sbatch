#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J exp_mgs
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/exp_mgs_%j.log

# 03_전처리_모션가이드_vs_균일추출 실행. 추가 인자는 run_experiment.py로 그대로 전달:
#   sbatch exp_mgs.sbatch --limit 1     # 스모크(첫 클립 1개)
#   sbatch exp_mgs.sbatch               # 전체(완료분 skip)
# gpu:2 — vlm.device=auto + max_memory {0:21,1:17} 확정값(config 주석) 그대로 쓰기 위함.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"

cd "/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_전처리_모션가이드_vs_균일추출"
PY=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/.venv/bin/python3

nvidia-smi --query-gpu=index,name,memory.used --format=csv
"$PY" run_experiment.py "$@"
rc=$?
echo "EXP exit=$rc"
exit $rc
