# -*- coding: utf-8 -*-
"""황색 LED 질문 1건 — Qwen3-14B(nf4, transformers) RAG 답변 생성.
체크리스트 준수: 무거운 로드 전에 import/파일 체크만 먼저(5초 내 사망 가능 구간)."""
import json, os, sys, time

SCRATCH = "/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/05_LLM모델링평가_RAG답변용"
CTX_FILE = os.path.join(SCRATCH, "led_context.json")
OUT_FILE = os.path.join(SCRATCH, "led_answer_qwen3_14b.json")

# ── 프리플라이트(가벼운 검증 먼저) ──
print("[preflight] python:", sys.executable)
import importlib
for pkg in ("torch", "transformers", "bitsandbytes"):
    importlib.import_module(pkg)
    print(f"[preflight] import {pkg} OK")
assert os.path.exists(CTX_FILE), f"컨텍스트 파일 없음: {CTX_FILE}"
ctx = json.load(open(CTX_FILE))
assert ctx["context_text"].strip(), "빈 컨텍스트"
import torch
assert torch.cuda.is_available(), "CUDA 불가 — GPU 노드가 아님"
print("[preflight] GPU:", torch.cuda.device_count(), "장")

# ── SYSTEM_PROMPT: evaluate_llm_realdb.py와 동일(비교 가능성 유지) ──
SYSTEM_PROMPT = """
너는 조립 현장의 신입 작업자를 돕는 음성 작업 보조 AI다.
아래 [검색된 암묵지]만 근거로 답하고, 근거가 부족하면 솔직히 부족하다고 말하라.
규칙:
- 첫 1~2문장에 핵심 답을 먼저 말한다 (음성으로 읽힐 것을 가정, 간결한 존댓말)
- 절차가 있으면 번호 단계로 나열
- 마지막 줄에 "출처: 영상ID (시작~끝 타임스탬프)" 표기
- 마크다운 강조 기호 사용 금지, 일반 텍스트로만
"""

MODEL = "Qwen/Qwen3-14B"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tok = AutoTokenizer.from_pretrained(MODEL)
# Qwen3 hybrid thinking 차단 — STEP5 llm_fusion._load와 동일 패턴
_orig = tok.apply_chat_template
def _no_think(*a, **k):
    k.setdefault("enable_thinking", False)
    return _orig(*a, **k)
tok.apply_chat_template = _no_think
print("[load] enable_thinking=False 주입")

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    max_memory={0: "20GiB", 1: "20GiB"},
)
print(f"[load] 완료 {time.time()-t0:.1f}s, device_map GPUs:",
      sorted(set(str(d) for d in model.hf_device_map.values())))

user_msg = f"[검색된 암묵지]: {ctx['context_text']}\n\n질문: {ctx['query']}"
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
print("[gen] 입력", inputs["input_ids"].shape[1], "토큰")

from torch.nn.attention import SDPBackend, sdpa_kernel
t1 = time.time()
with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
    gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
lat = time.time() - t1
n_new = gen.shape[1] - inputs["input_ids"].shape[1]
answer = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"[gen] 완료 {lat:.1f}s, 생성 {n_new}토큰 ({n_new/lat:.1f} tok/s)")
print("=" * 60)
print(answer)
print("=" * 60)
result = {"model": MODEL, "quant": "nf4", "decoding": "greedy(do_sample=False)",
          "query": ctx["query"], "retrieved": ctx["retrieved"],
          "input_tokens": int(inputs["input_ids"].shape[1]),
          "output_tokens": int(n_new), "latency_seconds": round(lat, 2),
          "tokens_per_sec": round(n_new / lat, 2), "generated_answer": answer}
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("saved:", OUT_FILE)
