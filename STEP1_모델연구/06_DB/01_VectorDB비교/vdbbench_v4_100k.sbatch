#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -J vdbbench_v4_100k
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/07_VectorDB비교/vdbbench_v4_100k_%j.log
set -uo pipefail
cd /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/07_VectorDB비교
/home/ai_user/team_a2/members/안나경/project-ai/STEP7_DB/.venv/bin/python3 benchmark_v4.py --scales 100000
echo "V4_100K_DONE"
