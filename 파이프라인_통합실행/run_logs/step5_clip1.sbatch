#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step5_c1
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step5_clip1_%j.log
# CLIP4 STEP5(새 프롬프트) — GPU 재배분 {0:20,1:9} 검증. 통과: OOM 없이 tacit_json 산출.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" run_step5.py --config "$CFG" --video-id "CLIP1_정상조립과정.mp4" || exit 1
echo "STEP5_CLIP1 DONE"
