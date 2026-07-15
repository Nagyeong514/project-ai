#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step5_motion_obs_r2
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step5_motion_obs_r2_%j.log

# 2026-07-10: 모션가이드 VLM 관찰(모션O 15클립 → 원본시각 보정 4파일)로 STEP5(정렬+LLM융합)만.
# 관찰 입력: STEP4_YOLO_VLM관찰/output_motion/vlm_observations_original_ts (config_motion_obs.yaml)
# 산출물: STEP5 output/aligned_windows_motion_obs, output/tacit_json_motion_obs (본선과 분리)
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config_motion_obs.yaml"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

cd "$ROOT/STEP5_LLM_암묵지_후보생성"
for VID in "CLIP2.mp4" "CLIP3_재부팅시도 전까지.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP5 $VID ==================="
  "$PY" run_step5.py --config "$CFG" --video-id "$VID" || exit 1
done
echo "ALL DONE"
