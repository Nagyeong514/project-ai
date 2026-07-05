#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J port4BE
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/port4BE_%j.log

# 이식 4-B보완+4-E 검증 런(2026-07-05): dedup 창 2→4(CLIP1 4문장 순환 퇴화) + 규칙9 + LED 색 교정 +
# _collapse_screen_runs(수치·버전 담긴 관찰은 접지 않고 개별 보존 예외 포함).
# 검증 대상: CLIP4. 통과: 120~154초 환각 다발 감소 + OS·프로그램명 추측 항목 소멸 + 관찰 ≥1건.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

cd "$ROOT/STEP4_YOLO_VLM관찰"
for VID in "CLIP1_정상조립과정.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP4 $VID ==================="
  "$PY" run_step4.py --config "$CFG" --video-id "$VID" || exit 1
done
echo "PORT4BE DONE"
