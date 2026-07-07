#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=verify_g_c4repro
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/verify_g_clip4_repro_check.log
set -euo pipefail
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:/opt/ohpc/pub/apps/cuda/13.0/lib64:${LD_LIBRARY_PATH:-}"
cd "/home/ai_user/team_a2/members/안나경/project-ai"

PY="파이프라인_통합실행/.venv/bin/python3"
CFG="00_사전연구/03_전처리_모션가이드_vs_균일추출/config.motion_clip1_test.yaml"

# 재현성 확인용 — 같은 모션 입력(windows.json은 이미 있으니 STEP5 전체를 다시 돌려
# 정렬은 동일하게 재생성되고 LLM 융합만 새로 샘플링됨)으로 STEP5만 재실행.
# tacit_json을 덮어쓰기 전에 이번 결과(용량 발화 2건 탈락)를 먼저 별도 이름으로 백업.
mkdir -p 00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/clip4_repro_check
cp "STEP5_LLM으로_암묵지_후보생성/output/tacit_json/CLIP4_재부팅및BIOS.mp4.tacit.json" \
   "00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/clip4_repro_check/CLIP4_run1_motion.tacit.json"

echo "=================== STEP5 재현성 확인 2회차: CLIP4_재부팅및BIOS.mp4 ==================="
"$PY" STEP5_LLM으로_암묵지_후보생성/run_step5.py --config "$CFG" --video-id "CLIP4_재부팅및BIOS.mp4"

cp "STEP5_LLM으로_암묵지_후보생성/output/tacit_json/CLIP4_재부팅및BIOS.mp4.tacit.json" \
   "00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/clip4_repro_check/CLIP4_run2_motion.tacit.json"

echo "=================== ALL DONE ==================="
