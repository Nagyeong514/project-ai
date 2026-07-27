#!/bin/bash
# 3개 모델(qwen3vl, internvl35, minicpm45) x CLIP1~4 전체 프레임 배치 실행.
#
# 모델당 한 번씩만 로드되도록(로딩 비용 상각) 영상(클립) 단위로 --frames_dir를 넘긴다
# (모델 1개당 클립 4번, 즉 클립마다 재로딩 — 완전히 1번만은 아니지만 로딩 자체가
# 20~30초 수준이라 몇 시간짜리 배치 전체에 비하면 무시 가능한 수준).
#
# GPU가 필요하므로 살아있는 SLURM 잡에 붙여서 실행해야 한다. 예:
#   srun --jobid=<현재 잡ID> --overlap bash inference/run_all.sh
#
# 실패한 (모델,클립) 조합이 있어도 나머지는 계속 진행한다(set -e 안 씀) — 끝나고
# 어떤 조합이 실패했는지 stderr/로그로 확인.

cd "$(dirname "$0")/.." || exit 1
VLM_COMPARE_DIR="$PWD"
BASE="/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/260704_VLM"
PROMPT="$VLM_COMPARE_DIR/prompts/common_prompt.md"

CLIPS=(
  "CLIP1_normal_assembly"
  "CLIP2_0704"
  "CLIP3_before_reboot_0704"
  "CLIP4_reboot_bios"
)

run_model() {
  local model_key="$1"
  local python_bin="$2"
  local script="$3"

  echo "===================================================================="
  echo "===== 모델: $model_key ====="
  echo "===================================================================="
  for clip in "${CLIPS[@]}"; do
    echo "--------------------------------------------------------------------"
    echo "--- $model_key / $clip ---"
    echo "--------------------------------------------------------------------"
    "$python_bin" "$script" \
      --frames_dir "$BASE/Sampling/frames/$clip" \
      --yolo_json "$BASE/Detection/results/$clip.detections.json" \
      --prompt_file "$PROMPT"
    if [ $? -ne 0 ]; then
      echo "[FAIL] $model_key / $clip 실패 — 다음으로 계속 진행"
    fi
  done
}

run_model "qwen3vl"    "$VLM_COMPARE_DIR/env_qwen3vl/bin/python3"    "$VLM_COMPARE_DIR/inference/run_qwen3vl.py"
run_model "internvl35" "$VLM_COMPARE_DIR/env_internvl35/bin/python3" "$VLM_COMPARE_DIR/inference/run_internvl35.py"
run_model "minicpm45"  "$VLM_COMPARE_DIR/env_minicpm45/bin/python3"  "$VLM_COMPARE_DIR/inference/run_minicpm45.py"

echo "===================================================================="
echo "[DONE] 전체 배치 종료"
echo "===================================================================="
