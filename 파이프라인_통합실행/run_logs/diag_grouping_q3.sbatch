#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J diag_group_q3
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag_grouping_q3_%j.log
# Pass 2 응집 A/B 진단 — 프롬프트(v1.1)·입력 동일, 모델만 Qwen3-14B로 교체.
# Qwen2.5-14B가 개정 프롬프트로도 병합 0/5(결정적)라 '모델 능력 문제' 가설 검증.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" diag_grouping_clip4.py --config "$ROOT/파이프라인_통합실행/config.yaml" \
    --trials 5 --model-override "Qwen/Qwen3-14B" || exit 1
echo "DIAG_GROUPING_Q3 DONE"
