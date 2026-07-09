#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J mini_grouping
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/mini_grouping_%j.log
# STEP5 Pass 2(지식 단위 응집) 미니 하네스 (2026-07-09).
# W04+W07+W10 축소 페이로드 + 가짜 파편 후보(c02/c03 = 같은 지식 2파편) —
# Part A(게이트 단위검증, 로그인 노드에서 9/9 선통과)를 노드에서 재확인 후
# Part B: 실제 LLM 3회 호출로 병합 성사·utterance 상속·1차 통과율 측정.
# 승급 규칙: 이 하네스 통과 → CLIP4 스모크 → 4클립 전체.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" mini_grouping_harness.py --config "$CFG" \
    --video-id "CLIP1_정상조립과정.mp4" \
    --indices 4,7,10 --trials 3 || exit 1
echo "MINI_GROUPING DONE"
