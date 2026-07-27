#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH -t 00:40:00
#SBATCH -o /home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/Pilot_0707/logs/%x_%j.log

# 사용법: sbatch -J pilot_<model> pilot.sbatch <model>
#   <model> ∈ qwen3vl | internvl35 | minicpm45
set -x
MODEL="$1"
BASE=/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/Pilot_0707
PY=/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/env_${MODEL}/bin/python

cd "$BASE"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
"$PY" "run_pilot_${MODEL}.py"
rc=$?
echo "PILOT ${MODEL} exit=$rc"
exit $rc
