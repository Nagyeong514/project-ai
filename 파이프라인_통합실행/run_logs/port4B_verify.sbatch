#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J port4B
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/port4B_%j.log

# 이식 4-B 검증 런(2026-07-05): dedup 전체청크 통합 + 정규화 + 최근2건(교대) 비교 +
# repeat_count/end_timestamp 필드 압축 + SYSTEM 규칙1 '묶기 금지' 완화.
# 검증 대상: CLIP1(60~190초 반복), CLIP3. 통과: 동일 문장 3연속 0건, repeat_count로 접힘.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

cd "$ROOT/STEP4_YOLO_VLM관찰"
for VID in "CLIP1_정상조립과정.mp4" "CLIP3_재부팅시도 전까지.mp4"; do
  echo "=================== STEP4 $VID ==================="
  "$PY" run_step4.py --config "$CFG" --video-id "$VID" || exit 1
done
echo "PORT4B DONE"
