#!/usr/bin/env bash
# Turing(RTX2080, sm75)에서 멀티모달 VLM을 vLLM으로 서빙 — InternVL2.5-8B-AWQ / MiniCPM-V-2.6(bnb) 비교용.
# LLM용 serve_vllm.sh의 Turing 우회 3종(libstdc++/FlashInfer/half)에 VLM 전용 플래그 추가:
#   --trust-remote-code      : InternVL/MiniCPM 커스텀코드
#   --limit-mm-per-prompt    : 한 프롬프트에 프레임 8장 허용(기본 1장)
#   --quantization           : awq(InternVL) | bitsandbytes(MiniCPM)
#
# 사용: ./serve_vlm.sh <model_path> <served_name> <port> <quant> [tp=2]
#   ./serve_vlm.sh OpenGVLab/InternVL2_5-8B-AWQ  InternVL2_5-8B  8011 awq          2
#   ./serve_vlm.sh openbmb/MiniCPM-V-2_6-int4    MiniCPM-V-2_6   8011 bitsandbytes 2
set -euo pipefail
MODEL="${1:?model_path}"; SERVED="${2:?served_name}"; PORT="${3:?port}"; QUANT="${4:?quant}"; TP="${5:-2}"
# 6번째 인자: --mm-processor-kwargs JSON (예: InternVL 타일링 제한 '{"max_dynamic_patch":1}').
# Qwen은 프레임당 ~400토큰인데 InternVL 기본(max_dynamic_patch=12)은 ~3328토큰 → 공정성+컨텍스트 위해 낮춤.
MMKW="${6:-}"

export LD_LIBRARY_PATH="/home/piai/anaconda3/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$TP" -ge 2 ]; then GPUS="0,1"; else GPUS="0"; fi

EXTRA=()
# bnb 사전양자화 체크포인트는 load-format도 bitsandbytes로 줘야 로드됨
[ "$QUANT" = "bitsandbytes" ] && EXTRA+=(--load-format bitsandbytes)
[ -n "$MMKW" ] && EXTRA+=(--mm-processor-kwargs "$MMKW")

echo "[serve_vlm] model=$MODEL served=$SERVED port=$PORT quant=$QUANT tp=$TP gpus=$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --quantization "$QUANT" \
  --trust-remote-code \
  --dtype half \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":8}' \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  "${EXTRA[@]}"
