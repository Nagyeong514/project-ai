#!/bin/bash
#SBATCH -p RTX6000
#SBATCH -w n5
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -J vdbbench
#SBATCH -o /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/07_VectorDB비교/vdbbench_%j.log
# Vector DB 비교 본 실험 — CPU 전용(GPU 불필요, --gres 미지정). 로그인 노드 부하 회피용
# 계산 노드 실행(10만 벡터 삽입·검색이 로그인 노드 정책에 걸릴 수 있음 — 2026-07-12 지적 반영).
set -uo pipefail
cd /home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/07_VectorDB비교
rm -f embeddings.npy chunks.json results.csv slide_table.md scaling_curve.png
V=/home/ai_user/team_a2/members/안나경/project-ai/STEP7_DB/.venv/bin/python3
"$V" benchmark.py 2>&1 | grep -v "Batches"
[ -s results.csv ] && [ -s slide_table.md ] && [ -s scaling_curve.png ] || { echo "FATAL: 산출물 누락"; exit 1; }
echo "VDBBENCH_DONE"
