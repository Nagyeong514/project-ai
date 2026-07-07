#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step5_re3
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step5_rerun3_%j.log

# STEP5 재실행(2026-07-06): e2e(2163)에서 실패한 CLIP1/3/4 만. CLIP2는 2163에서 성공(제외).
# 클립당 별도 python 프로세스(연쇄 메모리 축적 회피 — 통합러너 단일프로세스에서 3클립 붕괴 실측).
# 변경점: 침묵암묵지+5-B 프롬프트 + max_new_tokens 4096(1차 응답 잘림→재시도 OOM 근본 해소).
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
for VID in "CLIP1_정상조립과정.mp4" "CLIP3_재부팅시도 전까지.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "=================== STEP5 $VID ==================="
  "$PY" run_step5.py --config "$CFG" --video-id "$VID"
  echo "exit=$? for $VID"
done
echo "STEP5_RERUN3 DONE"
