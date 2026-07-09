#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -J step5_c4_pass2
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/step5_clip4_pass2_smoke_%j.log
# STEP5 Pass 2(지식 단위 응집) 스모크 — CLIP4 단독 (2026-07-09).
# 미니 그루핑 하네스 통과 후 승급 2단계. 통과 기준(하나라도 실패면 4클립 진행 금지):
#  (a) BIOS 계열 파편이 1~2그룹으로 응집 + 그 그룹 window_ids가 GT A8 구간(01:10~02:55) 커버
#  (b) 병합 그룹 reasoning_origin=utterance (W09 접지 멤버의 reasoning 상속)
#  (c) 1건 그룹들의 서술이 Pass 1과 바이트 동일 (pass1.json 스냅샷 대조)
#  (d) 귀속 게이트 통과 (전수/무중복/비인접·과잉병합 반려)
# 사전 확인: 라이브 tacit_json(2268 정본)은 tacit_json_run2_2268_backup_20260709 로 백업됨.
# silent_track_enabled 류 플래그는 STEP5 코드/config에 존재하지 않음(전수 grep 확인) —
# STEP6의 침묵 '가중치' 분기는 판정 쪽이라 이 실행과 무관.
set -x
export LD_LIBRARY_PATH="/home/ai_user/team_a2/.local/lib/python3.12/site-packages/cv2/../../lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/ai_user/team_a2/members/안나경/project-ai
PY="$ROOT/파이프라인_통합실행/.venv/bin/python3"
CFG="$ROOT/파이프라인_통합실행/config.yaml"
cd "$ROOT/STEP5_LLM으로_암묵지_후보생성"
"$PY" run_step5.py --config "$CFG" --video-id "CLIP4_재부팅및BIOS.mp4" || exit 1
echo "STEP5_CLIP4_PASS2_SMOKE DONE"
