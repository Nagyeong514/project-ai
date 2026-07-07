#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J port5Ag2
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/port5A_gpu2_%j.log

# 5-A GPU배치 수정 검증(2026-07-06): LLM device=auto+max_memory_gib(2-GPU 분산) 적용 후
# CLIP1 STEP5 단독 재실행. 통과: OOM 없이 fuse() 완주 + tacit_json 산출.
# 부수 검증: 5-A(merge_cap) 적용된 9개 윈도우가 실제로 OOM 없이 융합되는지도 같이 확인됨.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" run_step5.py --config "$CFG" --video-id "CLIP1_정상조립과정.mp4" || exit 1
echo "PORT5A_GPU2 DONE"
