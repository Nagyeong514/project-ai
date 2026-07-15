#!/bin/bash
# ======================================================================
# [인수인계 헤더] 용도: AI 글래스 원본 영상(mp4/mov/mkv)을 STT 입력용 16kHz mono PCM16 wav로 일괄 변환
# 입력: dataset/raw_videos/clip{1..4}.mp4  (스크립트 위치 기준 ./raw_videos)
# 출력: dataset/audio/clip{1..4}.wav, conversion_log.txt  (스크립트 위치 기준 ./audio)
# 실행 예시: cd dataset && chmod +x video_to_wav.sh && ./video_to_wav.sh
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
# =========================================================
# video_to_wav.sh
# AI 글래스 원본 영상(mp4/mov/mkv)을 STT 입력용 wav로 일괄 변환
#
# 변환 규격 (3개 비교 모델 - faster-whisper / speechbrain / owsm 공통 요구사항)
#   - Sample rate : 16,000 Hz
#   - Channel     : mono (1채널)
#   - Codec       : PCM 16bit (pcm_s16le)
#
# 사용법:
#   1) raw_videos/ 폴더에 clip1.mp4 ~ clip4.mp4 형태로 원본 영상을 넣는다
#   2) chmod +x video_to_wav.sh && ./video_to_wav.sh
#   3) audio/ 폴더에 clip1.wav ~ clip4.wav 가 생성된다
# =========================================================

set -euo pipefail

RAW_DIR="./raw_videos"
OUT_DIR="./audio"
LOG_FILE="$OUT_DIR/conversion_log.txt"

mkdir -p "$OUT_DIR"

if ! command -v ffmpeg &> /dev/null; then
  echo "[오류] ffmpeg가 설치되어 있지 않습니다. 먼저 설치해주세요. (예: brew install ffmpeg / apt install ffmpeg)"
  exit 1
fi

echo "=== WAV 변환 로그 $(date) ===" > "$LOG_FILE"

FOUND=0
while IFS= read -r -d '' video; do
  FOUND=1
  filename=$(basename "$video")
  name="${filename%.*}"
  output="$OUT_DIR/${name}.wav"

  echo "[변환 시작] $filename → ${name}.wav"

  ffmpeg -y -i "$video" \
    -ar 16000 \
    -ac 1 \
    -acodec pcm_s16le \
    -vn \
    "$output" >> "$LOG_FILE" 2>&1

  duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$output")
  echo "[완료] ${name}.wav (길이: 약 ${duration}초)" | tee -a "$LOG_FILE"
done < <(find "$RAW_DIR" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" \) -print0 | sort -z)

if [ "$FOUND" -eq 0 ]; then
  echo "[경고] $RAW_DIR 에서 mp4/mov/mkv 파일을 찾지 못했습니다. 파일명을 clip1.mp4 ~ clip4.mp4 형태로 넣어주세요."
  exit 1
fi

echo ""
echo "모든 변환이 완료되었습니다. 결과 폴더: $OUT_DIR"
echo "이 wav 파일들은 faster-whisper / speechbrain / owsm 3개 모델에 동일하게 입력됩니다."

# -----------------------------------------------------------------
# 참고) 배경 잡음(작업대 소음, 공구 소리 등)이 심한 경우, 아래처럼
# 하이패스/로우패스 필터를 추가할 수 있습니다. 단, 필터를 걸면 모델 간
# 비교 조건이 "원본"이 아니라 "전처리 적용본"으로 바뀌므로, CLI 연구
# 계획서의 "실험 조건"에 필터 적용 여부를 반드시 함께 명시해야 합니다.
#
# ffmpeg -y -i "$video" -ar 16000 -ac 1 -acodec pcm_s16le \
#   -af "highpass=f=80, lowpass=f=8000" -vn "$output"
# -----------------------------------------------------------------
