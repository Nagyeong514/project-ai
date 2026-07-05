#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step45_final
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step45_final_%j.log

# 2026-07-05 최종 검증 런: 프롬프트 하드닝(규칙8, 플레이스홀더) + 청크 내 중복압축까지
# 포함한 코드로 4클립 전부 STEP4 → STEP5. 이전 부분 런(2099/2102/2104)과 달리
# 전 클립이 동일 코드 버전으로 재생성되도록 처음부터 다시 돈다.
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

cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
for VID in "CLIP1_정상조립과정.mp4" "CLIP2.mp4" "CLIP3_재부팅시도 전까지.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP5 $VID ==================="
  "$PY" run_step5.py --config "$CFG" --video-id "$VID" || exit 1
done
echo "ALL DONE"
