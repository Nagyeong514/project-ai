#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J port5A
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/port5A_%j.log

# 이식 5-A 검증 런(2026-07-06): aligner.py _merge()에 merge_cap_sec(기본 20.0) 적용.
# CLIP1 STEP5 단독 실행(STEP3 transcript + 1단계에서 새로 만든 STEP4 산출물 사용).
# 통과: 윈도우가 1개로 안 뭉치고 여러 개로 쪼개짐(각 폭 <= 20초) + LLM 융합 OOM 없이 완료.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" run_step5.py --config "$CFG" --video-id "CLIP1_정상조립과정.mp4" || exit 1
echo "PORT5A DONE"
