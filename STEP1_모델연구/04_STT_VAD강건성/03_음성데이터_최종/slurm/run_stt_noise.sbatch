#!/bin/bash
# ======================================================================
# [인수인계 헤더] 용도: run_stt_noise.py를 GPU 노드 n5(RTX6000)에 제출하는 SLURM 스크립트 — CUDA12 라이브러리 LD_LIBRARY_PATH 설정 포함
# 입력: dataset/audio_noise/{D1..D4}/clip{1..4}.wav (run_stt_noise.py가 읽음)
# 출력: results_noise_robustness/stt_outputs/, timing_log.csv, slurm_stt_<jobid>.log
# 실행 예시: (프로젝트 루트에서) sbatch scripts/run_stt_noise.sbatch  →  squeue -u $USER 로 상태 확인
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
#SBATCH --job-name=noise_stt
#SBATCH --partition=RTX6000
#SBATCH --nodelist=n5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/ai_user/team_a2/members/길예진/stt-project/results_noise_robustness/slurm_stt_%j.log

cd /home/ai_user/team_a2/members/길예진/stt-project
# ctranslate2(faster-whisper)는 CUDA 12용 cublas/cudnn을 요구하므로
# venv에 pip으로 설치한 nvidia-cublas-cu12/nvidia-cudnn-cu12 경로를 잡아준다
SITE=.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$PWD/$SITE/nvidia/cublas/lib:$PWD/$SITE/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
.venv/bin/python scripts/run_stt_noise.py
