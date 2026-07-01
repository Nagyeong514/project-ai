"""
LLM 융합 어댑터 (Qwen2.5-14B). 스펙 5.7.

입력: 정렬된 (VLM 행동 + 정제 transcript) 구간 → 암묵지 후보 JSON.
출력은 tacit_schema 로 검증(실패 시 재시도 훅).

⚠️ Turing(sm75): fp16/4bit만. ⚠️ 오늘은 모델 로딩 금지 — 지연 import/로딩.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..prompts.llm_fusion_prompt import build_fusion_messages
from ..schema.intermediate import AlignedWindow, FrameMeta, seconds_to_hhmmss
from ..schema.tacit_schema import TacitKnowledgeDocument


class QwenLLMFusion:
    """Qwen2.5-14B 융합 어댑터. registry 키: 'qwen2_5_14b'."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        backend: str = "vllm",  # AWQ는 vLLM(autoawq 미설치). Turing 우회 플래그 적용.
        device: str = "cuda",
        dtype: str = "half",  # Turing: bf16 금지
        quantization: str | None = "awq",
        tensor_parallel_size: int = 1,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        max_retries: int = 2,  # 스키마 검증 실패 시 재시도
        # vLLM Turing 우회(필수 — [[step3-runtime-recipe]])
        attention_backend: str = "TRITON_ATTN",  # FA2/FlashInfer는 sm80+/nvcc 필요라 死
        enforce_eager: bool = True,
        max_num_seqs: int = 16,  # 256은 샘플러 워밍업 OOM
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        **extra: Any,
    ):
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.attention_backend = attention_backend
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra = extra
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        if self.backend == "vllm":
            import os
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            from vllm import LLM  # noqa

            self._model = LLM(
                model=self.model_name,
                dtype=self.dtype,
                quantization="bitsandbytes" if self.quantization == "nf4" else self.quantization,
                tensor_parallel_size=self.tensor_parallel_size,
                attention_backend=self.attention_backend,
                enforce_eager=self.enforce_eager,
                max_num_seqs=self.max_num_seqs,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                **self.extra,
            )
        elif self.backend == "hf_transformers":
            import torch  # noqa
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            quant_kwargs: Dict[str, Any] = {}
            if self.quantization == "nf4":
                from transformers import BitsAndBytesConfig  # noqa

                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,  # Turing: fp16
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quant_kwargs["torch_dtype"] = getattr(torch, self.dtype)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.device, **quant_kwargs
            )
        else:
            raise ValueError(f"알 수 없는 LLM backend: {self.backend}")

    def unload(self) -> None:
        """GPU 메모리 해제(대칭용). 보통 LLM이 마지막 단계라 호출 안 해도 됨."""
        import gc

        self._model = None
        self._tok = None
        gc.collect()
        try:
            import torch  # noqa

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 윈도우 → LLM 입력 직렬화 ──────────────────────────────────────
    def _serialize(self, windows: List[AlignedWindow]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for w in windows:
            payload.append(
                {
                    "case": w.case,  # fusion | action_only | utterance_only
                    "window_start": seconds_to_hhmmss(w.window_start),
                    "window_end": seconds_to_hhmmss(w.window_end),
                    "actions": [
                        {"timestamp": seconds_to_hhmmss(a.timestamp),
                         "actor": a.actor, "action": a.action,
                         "objects_visible": a.objects}
                        for a in w.actions
                    ],
                    "utterances": [
                        {
                            "timestamp": seconds_to_hhmmss(u.start),
                            "raw_text": u.raw_text,  # 발화 원문(source_utterance 보존)
                            "normalized_text": u.normalized_text,
                            "repeat_hallucination": u.repeat_hallucination,
                        }
                        for u in w.utterances
                    ],
                }
            )
        return payload

    def fuse(self, windows: List[AlignedWindow], meta: FrameMeta) -> TacitKnowledgeDocument:
        self._load()
        payload = self._serialize(windows)
        messages = build_fusion_messages(meta.video_id, payload)

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            raw = self._infer(messages)
            try:
                doc = self._parse_and_validate(raw, meta.video_id)
                self._assign_ids(doc, meta.video_id)
                return doc
            except Exception as e:  # 스키마 검증 실패 → 재시도(스펙: 검증 실패 시 에러/재시도 훅)
                last_err = e
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"JSON 스키마 검증 실패: {e}. 스키마를 정확히 지켜 JSON만 다시 출력하라."},
                ]
        raise RuntimeError(f"LLM 융합 스키마 검증 {self.max_retries+1}회 실패: {last_err}")

    def _infer(self, messages: List[Dict[str, str]]) -> str:
        """텍스트 추론. vLLM chat (Turing 플래그는 _load 에서 적용)."""
        if self.backend == "vllm":
            from vllm import SamplingParams  # noqa
            sp = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
            out = self._model.chat(messages, sp)
            return out[0].outputs[0].text
        elif self.backend == "hf_transformers":
            import torch  # noqa
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tok(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                           do_sample=self.temperature > 0, temperature=self.temperature)
            return self._tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        raise ValueError(f"알 수 없는 LLM backend: {self.backend}")

    @staticmethod
    def _parse_and_validate(raw: str, video_id: str) -> TacitKnowledgeDocument:
        """모델 출력에서 JSON 추출 → Pydantic 검증."""
        text = raw.strip()
        # 코드펜스 제거
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):] if "{" in text else text
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1])
        obj.setdefault("video_id", video_id)
        return TacitKnowledgeDocument.model_validate(obj)

    @staticmethod
    def _assign_ids(doc: TacitKnowledgeDocument, video_id: str) -> None:
        """id 비어있으면 자동 부여: tk_<video_stem>_<task약어>_NNN."""
        stem = video_id.rsplit(".", 1)[0].replace("-", "_")
        for i, c in enumerate(doc.candidates, start=1):
            if not c.id:
                c.id = f"tk_{stem}_{i:03d}"
