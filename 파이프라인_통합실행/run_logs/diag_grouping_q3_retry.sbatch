#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J diag_q3_retry
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag_grouping_q3_retry_%j.log
# Pass 2 A/B 진단 2단계 — Qwen3-14B + 실전 _group() 재시도 루프.
# 1회 추론에선 5/5 병합 {c08,c09,c11}(의미 정확)이나 서술 누락+비인접 형식 위반.
# 재시도 피드백이 {c08,c09}+단독으로 수렴시키는지 검증. 3회 반복.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" diag_grouping_clip4.py --config "$ROOT/파이프라인_통합실행/config.yaml" \
    --trials 3 --model-override "Qwen/Qwen3-14B" --use-retry-loop || exit 1
echo "DIAG_Q3_RETRY DONE"
