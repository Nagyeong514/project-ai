#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J e2e_final
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/e2e_final_%j.log

# 최종 검증 런(2026-07-06): 1·2단계 이식(4-A/B/C/D/E/F + 5-A/B + LLM 2-GPU/sdpa) 전부 반영된
# 코드로 CLIP1~4 를 STEP3→4→5 처음부터 끝까지. 이후 정답지 9건 회수율 채점(별도 스크립트).
# run_all_clips.py: 클립별 try/except — 한 클립이 죽어도 나머지 진행, 마지막에 SUMMARY.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
cd "$ROOT/파이프라인_통합실행"
"$PY" run_all_clips.py
echo "E2E_FINAL exit=$?"
