#!/usr/bin/env bash
# Turing(RTX 2080, sm75)에서 vLLM 띄우는 헬퍼 — 이 본체의 하드 이슈 3개를 한 번에 처리한다.
#  1) conda libstdc++ 우선: 시스템 /lib 의 libstdc++ 는 CXXABI 가 낮아(≤1.3.13) pyarrow import 가
#     "CXXABI_1.3.15 not found" 로 죽는다. anaconda3/lib(1.3.15 포함)을 LD_LIBRARY_PATH 앞에 둔다.
#  2) FlashInfer 우회: 어텐션/샘플러 FlashInfer 백엔드는 실행 시 nvcc 로 JIT 컴파일을 시도하는데
#     이 머신엔 CUDA 툴킷(nvcc)이 없어 "Could not find nvcc" 로 죽는다. Triton 어텐션 + 네이티브 샘플러로 우회.
#     ※ vLLM 0.23 에선 어텐션 백엔드는 환경변수(VLLM_ATTENTION_BACKEND, 이젠 "unknown")가 아니라
#       CLI 플래그 --attention-backend 로 줘야 먹는다. 샘플러만 VLLM_USE_FLASHINFER_SAMPLER=0 env.
#  3) bf16 미지원(Turing) → --dtype half 고정.
#
# 사용:
#   ./serve_vllm.sh <model_path> <served_name> <port> [quant=awq] [tp=1]
# 예 (config.yaml 의 model_path/name/port 와 일치):
#   ./serve_vllm.sh hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 Llama-3.1-8B-Instruct 8001 awq 1
#   ./serve_vllm.sh Qwen/Qwen2.5-14B-Instruct-AWQ                      Qwen2.5-14B-Instruct  8001 awq 2   # 14B=2장
#   ./serve_vllm.sh shuyuej/gemma-2-9b-it-GPTQ                         Gemma-2-9B-it         8010 gptq 1   # 심판
set -euo pipefail
MODEL="${1:?model_path 필요}"; SERVED="${2:?served_name 필요}"; PORT="${3:?port 필요}"
QUANT="${4:-awq}"; TP="${5:-1}"; UTIL="${6:-0.90}"; DTYPE="${7:-half}"
# DTYPE: Turing은 bf16 하드웨어 미지원 → 대부분 half. 단 gemma2는 vLLM이 float16을 거부(수치불안정)하므로
#        심판 Gemma는 7번째 인자로 float32 를 넘긴다(예: ... 8010 gptq 2 0.90 float32).

export LD_LIBRARY_PATH="/home/piai/anaconda3/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0   # 샘플러는 env로 우회(어텐션은 아래 --attention-backend CLI로)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 단편화 완화. Turing AWQ는 추론 시 dequantize 스크래치가 커서 필수

# TP>=2 면 2장(0,1), 아니면 0번 1장
if [ "$TP" -ge 2 ]; then GPUS="0,1"; else GPUS="0"; fi

echo "[serve_vllm] model=$MODEL served=$SERVED port=$PORT quant=$QUANT tp=$TP gpus=$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --quantization "$QUANT" \
  --trust-remote-code \
  --dtype "$DTYPE" \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization "$UTIL" \
  --tensor-parallel-size "$TP" \
  --port "$PORT"
# --max-num-seqs 16: 기본 256은 샘플러 워밍업에서 (max_num_seqs × vocab) 로짓 버퍼가 8GB를 넘겨
#   "CUDA out of memory ... warming up sampler with 256 dummy requests"로 죽음(특히 vocab 큰 14B).
#   평가는 순차 호출이라 16이면 충분. 더 줄여도 됨.
# --enforce-eager: 8GB 카드에선 CUDA 그래프 캡처(~0.74GB)+torch.compile 메모리 때문에 KV 캐시가
#   마이너스가 됨("No available memory for the cache blocks"). eager로 끄면 그만큼 KV로 회수+기동 빠름.
#   (추론 처속은 약간 손해지만 평가용엔 무방.) 16GB로 여유로우면 떼고 util 낮춰도 됨.
