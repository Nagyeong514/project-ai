#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --job-name=verify_h_port
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/verify_h_port_repro_clip2.log
set -euo pipefail
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:/opt/ohpc/pub/apps/cuda/13.0/lib64:${LD_LIBRARY_PATH:-}"
cd "/home/ai_user/team_a2/members/안나경/project-ai"

PY="파이프라인_통합실행/.venv/bin/python3"
CFG="00_사전연구/03_전처리_모션가이드_vs_균일추출/config.motion_clip1_test.yaml"
MASTER="/home/ai_user/team_a2/members/안나경/master/master_videos"

echo "=================== 이식 후 재현확인: run_step3.py CLIP2.mp4 (auto_generate=false 기본값, 기존 팀원 산출물만 읽음) ==================="
"$PY" STEP3_전처리/run_step3.py --config "$CFG" --video "$MASTER/CLIP2.mp4"

echo "=================== times 비교(이식 전 vs 이식 후) ==================="
"$PY" -c "
import json
before = json.load(open('00_사전연구/03_전처리_모션가이드_vs_균일추출/run_logs/frames_meta_before_port_CLIP2.json'))['times']
after = json.load(open('STEP3_전처리/output/_frames/CLIP2.mp4/frames_meta.json'))['times']
print('이식 전 times 개수:', len(before))
print('이식 후 times 개수:', len(after))
print('완전 일치:', before == after)
if before != after:
    print('불일치 지점:')
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            print(f'  index {i}: before={b} after={a}')
    if len(before) != len(after):
        print(f'  길이 자체가 다름: before={len(before)} after={len(after)}')
"

echo "=================== ALL DONE ==================="
