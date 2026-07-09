#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J mini_harness
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/mini_harness_%j.log
# STEP5 미니 검증 하네스 + 결정화(온도) 실험 (2026-07-08).
# 축소 페이로드(W04 발화3 + W07 발화2·GT A2 + W10 무발화)로 모델 1회 로드 후
# 온도 4단계(0.3 → 0.2현행 → 0.1 → 0.0 greedy) × 3회씩 1차 통과율·접지율 측정.
# 목표: "1차 통과가 기본값"이 되는 최저 온도 확정 → 그 온도로 4클립 1회만 재실행.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" mini_fusion_harness.py --config "$CFG" \
    --video-id "CLIP1_정상조립과정.mp4" \
    --indices 4,7,10 --temps 0.3,0.2,0.1,0.0 --trials 3 || exit 1
echo "MINI_HARNESS DONE"
