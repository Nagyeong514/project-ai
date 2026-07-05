#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J port4A
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/port4A_%j.log

# 이식 4-A 검증 런(2026-07-05): raw 필드를 관찰별 → 파일 최상위 raw_by_chunk(청크당 1건)로.
# 검증 대상: CLIP3(이전 29건 부풀림 사고), CLIP4(관찰 40건 × raw 중복 실측).
# 통과 기준: observations 각 원소에 raw 없음 + 최상위 raw_by_chunk 키가 청크 수만큼 + 파일 크기 급감.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

cd "$ROOT/STEP4_YOLO_VLM관찰"
for VID in "CLIP3_재부팅시도 전까지.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP4 $VID ==================="
  "$PY" run_step4.py --config "$CFG" --video-id "$VID" || exit 1
done
echo "PORT4A DONE"
