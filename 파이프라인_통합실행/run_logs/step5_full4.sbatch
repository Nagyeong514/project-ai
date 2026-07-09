#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step5_full4
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step5_full4_%j.log
# 4클립 전체 STEP5 재실행 (2026-07-08, 어댑터 경계수정 후 확장 검증 ①).
# 클립별 attempt 수·게이트 반려 사유는 로그의 [LLM-FUSE] 라인으로 수집 →
# do_sample 결정화 실험의 근거 데이터로 쓴다.
# 2026-07-08 2차(잡 2267 실패 후): max_retries 2→5 + 한 클립 실패해도 나머지
# 클립은 계속 돈다(한 GPU 할당에서 4클립 전부의 시도 통계 확보 목적).
# 실패 클립이 있으면 마지막에 목록 출력하고 exit 1.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
FAILED=""
for VID in "CLIP1_정상조립과정.mp4" "CLIP2.mp4" "CLIP3_재부팅시도 전까지.mp4" "CLIP4_재부팅및BIOS.mp4"; do
  echo "===== STEP5 START: $VID ====="
  if "$PY" run_step5.py --config "$CFG" --video-id "$VID"; then
    echo "===== STEP5 DONE: $VID ====="
  else
    echo "STEP5_FAIL: $VID"
    FAILED="$FAILED $VID"
  fi
done
if [ -n "$FAILED" ]; then
  echo "STEP5_FULL4 FAILED_CLIPS:$FAILED"
  exit 1
fi
echo "STEP5_FULL4 DONE"
