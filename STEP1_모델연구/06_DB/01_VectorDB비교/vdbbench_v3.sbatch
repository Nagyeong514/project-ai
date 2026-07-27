#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -J vdbbench_v3
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/07_VectorDB비교/vdbbench_v3_%j.log
set -uo pipefail
cd /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/07_VectorDB비교
/home/ai_user/team_a2/members/안나경/project-ai/STEP7_DB/.venv/bin/python3 benchmark_v3.py
[ -s results_v3.csv ] && [ -s slide_table_v3.md ] || { echo "FATAL: 산출물 누락"; exit 1; }
echo "VDBBENCH_V3_DONE"
