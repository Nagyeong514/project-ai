#!/bin/bash
#SBATCH -p RTX6000
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH --job-name=e2e_step3to6
#SBATCH --output=/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/e2e_step3to6_%j.log
# ─────────────────────────────────────────────────────────────────────────────
# 원샷 오케스트레이터 (2026-07-14): 클립 넣으면 STEP3→4→5→adapter→STEP6→STEP7까지 한 번에.
#   STEP3~5(통합러너)  : 전처리·STT·YOLO·VLM·LLM융합 → 암묵지 후보(tacit_json)
#   step6_adapter      : STEP5 출력 → STEP6 입력 형식 변환
#   STEP6(품질검증)    : accept/hold/reject 판정
#   step7_adapter      : STEP6 판정(decision/confidence) → STEP7 스키마(routing/confidence.score)
#   STEP7(DB 적재)     : ChromaDB(tacit_knowledge) 적재
#
# ⚠️ 프로세스를 합치지 않는다 — 각 단계를 '자기 venv'로 순차 호출한다(venv/모델 충돌 회피,
#    CLAUDE.md 규칙). 단계 사이 데이터는 전부 파일로 주고받는다.
# ⚠️ STEP8(RAG 서비스)은 상주 웹서버라 배치에 미포함 — main.py 종료 후 별도 기동.
#
# 실행: srun -p RTX6000 -w n5 --gres=gpu:2 bash run_e2e_step3to6.sbatch
#   또는: sbatch run_e2e_step3to6.sbatch
#   출력 태그 바꾸려면: RUN_TAG=myrun sbatch run_e2e_step3to6.sbatch
#   (STEP6까지만: main.py --skip-db / accept만 적재: main.py --accept-only-db)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="/home/ai_user/team_a2/members/안나경/project-ai"
# 실제 오케스트레이션 로직은 main.py 하나에 있다(단일 진입점). 이 sbatch 는 SLURM 래퍼일 뿐.
cd "$ROOT/파이프라인_통합실행"

nvidia-smi --query-gpu=index,name,memory.used --format=csv

.venv/bin/python3 main.py --run-tag "${RUN_TAG:-e2e_run}"
