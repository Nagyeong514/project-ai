#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J stage1
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/stage1_%j.log

# 1단계 완료 검증(2026-07-06): 4-A/F/B/E/C/D 전부 적용된 코드로 CLIP1~4 STEP4 단독 재실행.
# 통과 기준: OOM 0회 / 동일문장 3연속 0건 / CLIP4 포함 관찰 ≥1건 / raw 청크당 1건 /
#           timestamp HH:MM:SS 단일 / 정적구간 OS·프로그램명 환각 없음 / T값 assert 미발동.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
cd "$ROOT/STEP4_YOLO_VLM관찰"
for VID in "CLIP1_정상조립과정.mp4" "CLIP2.mp4" "CLIP3_재부팅시도 전까지.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP4 $VID ==================="
  "$PY" run_step4.py --config "$CFG" --video-id "$VID" || exit 1
done
echo "STAGE1 DONE"
