"""
LLM 융합 어댑터 (Qwen2.5-14B). 스펙 5.7.

입력: 정렬된 (VLM 행동 + 정제 transcript) 구간 → 암묵지 후보 JSON.
출력은 tacit_schema 로 검증(실패 시 재시도 훅).

⚠️ Turing(sm75): fp16/4bit만. ⚠️ 오늘은 모델 로딩 금지 — 지연 import/로딩.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from step5_prompts.llm_fusion_prompt import build_fusion_messages
from tacit_common.schema.intermediate import AlignedWindow, FrameMeta, seconds_to_hhmmss
from step5_schema.tacit_schema import TacitKnowledgeDocument


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
        # 2026-07-06(탈선 방어): 입력 발화 중 최소 이 비율이 출력 source_utterance에
        # 반영돼야 합격. 미달이면 '발화 누락 탈선'으로 재시도(형식만 보던 합격 기준 보강).
        min_utterance_coverage: float = 0.5,
        # vLLM Turing 우회(필수 — [[step3-runtime-recipe]])
        attention_backend: str = "TRITON_ATTN",  # FA2/FlashInfer는 sm80+/nvcc 필요라 死
        enforce_eager: bool = True,
        max_num_seqs: int = 16,  # 256은 샘플러 워밍업 OOM
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        # 2026-07-06: STEP4 VLM(vlm_qwen.py)과 동일 패턴 이식 — device="auto"일 때만 의미.
        # 실측(gpu:1 단독): 로딩만 9.25GiB, 첫 forward에서 peak 21.60GiB로 22GB 한도 초과 OOM.
        max_memory_gib: float | Dict[int, float] | None = None,
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
        self.min_utterance_coverage = min_utterance_coverage
        self.attention_backend = attention_backend
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_memory_gib = max_memory_gib
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
            # 2026-07-06: device="auto"(2-GPU 분산) + max_memory_gib 이식(vlm_qwen.py:_load
            # 동일 패턴). 로딩(9.25GiB)만으로는 1장 안에 들어가지만 첫 forward에서 활성화
            # 메모리까지 더하면 22GB 한도를 넘어(실측 peak 21.60GiB) OOM — 2장에 나눠 여유 확보.
            if self.device == "auto" and self.max_memory_gib is not None and torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                if isinstance(self.max_memory_gib, dict):
                    mm = {int(k): v for k, v in self.max_memory_gib.items() if int(k) < n_gpus}
                    dropped = {k: v for k, v in self.max_memory_gib.items() if int(k) >= n_gpus}
                    if dropped:
                        print(f"[LLM-DEVICE] max_memory 중 없는 장치 제외: {dropped} (가시 GPU {n_gpus}장)")
                    quant_kwargs["max_memory"] = {k: f"{v}GiB" for k, v in mm.items()}
                else:
                    quant_kwargs["max_memory"] = {i: f"{self.max_memory_gib}GiB" for i in range(n_gpus)}
                print(f"[LLM-DEVICE] max_memory 강제: {quant_kwargs['max_memory']}")
            # 2026-07-06(사용자 예외승인): attn_implementation 미지정 시 기본 attention 구현이
            # 시퀀스 길이에 비례해 activation 메모리를 크게 잡아 GPU0에서 8.99GiB 요구 OOM이
            # max_memory 배분(14/12, 6/16, 20/20 전부 동일 재현)과 무관하게 발생함을 실측 확정.
            # vlm_qwen.py:_load 가 이미 쓰는 검증된 sdpa로 통일(생성 결과/방식 불변, attention의
            # 메모리 구현 방식만 바뀜 — Turing은 flash-attn2 불가, sdpa는 지원).
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.device,
                attn_implementation="sdpa", **quant_kwargs
            )
            hf_device_map = getattr(self._model, "hf_device_map", None)
            if hf_device_map is not None:
                devices_used = sorted(set(str(d) for d in hf_device_map.values()))
                print(f"[LLM-DEVICE] hf_device_map 사용 GPU: {devices_used}")
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
                         "objects_visible": a.objects,
                         # STEP4 dedup(4-B)이 접은 반복 관찰은 필드로만 남으므로 LLM에도
                         # 전달한다(예전 action 접미사 텍스트가 하던 역할 — 무손실 원칙 유지).
                         **({"repeat_count": a.repeat_count,
                             "repeat_until": seconds_to_hhmmss(a.end_timestamp)}
                            if getattr(a, "repeat_count", 1) > 1 and a.end_timestamp is not None
                            else {})}
                        for a in w.actions
                    ],
                    "utterances": [
                        {
                            "timestamp": seconds_to_hhmmss(u.start),
                            "raw_text": u.raw_text,  # 발화 원문(source_utterance 보존)
                            # normalized_text 는 2026-07-06부터 미전송 — raw_text와 거의 동일한
                            # 문장 2벌을 보내 프롬프트만 키움(CLIP4 6,330토큰 → prefill OOM의
                            # 일부). source_utterance 는 어차피 raw_text 원문만 쓴다.
                            "repeat_hallucination": u.repeat_hallucination,
                        }
                        for u in w.utterances
                    ],
                }
            )
        return payload

    # ── 탈선 방어(2026-07-06): CLIP1 실측 사고 — 1차 생성이 깨진 토큰(比特rigesimal)으로
    # 탈선한 뒤 재시도 출력이 발화 6건을 통째로 버렸는데 스키마 검증만 통과해 살아남았다.
    # 형식(스키마)만 보던 합격 기준에 내용 검증 2종 + 재시도 다변화를 추가한다. ──

    @staticmethod
    def _norm_txt(s: str) -> str:
        import re
        return re.sub(r"[\s\W]+", "", s or "")

    @staticmethod
    def _looks_derailed(raw: str) -> List[str]:
        """생성 탈선 신호: 한자(CJK Unified Ideographs)·대체문자(�) 등 이 파이프라인
        출력(한국어/영문/JSON)에 나올 수 없는 문자. 발견 문자 목록 반환(비면 정상)."""
        import re
        bad = re.findall(r"[一-鿿�]", raw)
        return bad

    def _missing_utterances(self, doc: TacitKnowledgeDocument, input_utts: List[str]) -> List[str]:
        """입력 payload의 발화(rep_hallucination=False) 중 출력 어디의 source_utterance에도
        안 나타난 것들. 부분 복사를 감안해 정규화 후 포함관계(양방향 앞 20자)로 판정."""
        sources = []
        for c in doc.candidates:
            for s in c.knowledge.diagnostic_steps:
                if s.source_utterance:
                    sources.append(self._norm_txt(s.source_utterance))
        joined = " ".join(sources)
        missing = []
        for u in input_utts:
            nu = self._norm_txt(u)
            head = nu[:20] if len(nu) >= 20 else nu
            if head and head not in joined:
                missing.append(u)
        return missing

    def fuse(self, windows: List[AlignedWindow], meta: FrameMeta) -> TacitKnowledgeDocument:
        self._load()
        payload = self._serialize(windows)
        messages = build_fusion_messages(meta.video_id, payload)
        # 탈선 검증 기준선: 입력에 실제로 들어간(환각 아님) 발화 원문들
        input_utts = [u.raw_text for w in windows for u in w.utterances
                      if not u.repeat_hallucination]

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            # 재시도 다변화: 같은 조건으로 다시 돌리면 같은 탈선을 반복하므로 시도마다
            # 온도를 올린다(0.2→0.45→0.7 상한). 1차는 기존 설정 그대로(정상 클립 무영향).
            temp = self.temperature if attempt == 0 \
                else min(0.7, (self.temperature or 0.2) + 0.25 * attempt)
            raw = None
            try:
                # 2026-07-08(차단버그 수정): raw=self._infer(...)가 이 try 밖에 있어서 생성
                # 자체가 던지는 예외(CUDA OOM 등)를 재시도 루프가 못 잡고 프로세스 전체가
                # 죽는 문제가 CLIP4에서 실측됨(1차 탈선 재시도 중 OOM). 모션가이드와 무관한
                # STEP5 자체 버그 — 균일추출 경로에서도 프롬프트 길이/GPU 여유가 같은 조건이면
                # 동일하게 재현될 수 있다. _infer 호출을 try 안으로 옮겨 여기서도 잡히게 한다.
                raw = self._infer(messages, temperature=temp)
                bad = self._looks_derailed(raw)
                if bad:
                    raise ValueError(f"생성 탈선(비정상 문자 {len(bad)}개: {''.join(bad[:5])!r}) — 이 출력 폐기")
                doc = self._parse_and_validate(raw, meta.video_id)
                self._assign_ids(doc, meta.video_id)
                # 내용 검증: 입력 발화의 절반 이상이 source_utterance에 없으면 발화 누락 탈선
                missing = self._missing_utterances(doc, input_utts)
                if input_utts and len(missing) > len(input_utts) * (1 - self.min_utterance_coverage):
                    preview = " / ".join(m[:24] for m in missing[:3])
                    raise ValueError(
                        f"발화 누락 탈선: 입력 발화 {len(input_utts)}건 중 {len(missing)}건이 "
                        f"출력 source_utterance에 없음(예: {preview})")
                if missing:
                    print(f"[LLM-FUSE] 경고: 발화 {len(missing)}/{len(input_utts)}건 미반영(허용 범위)")
                return doc
            except Exception as e:  # 생성 실패(OOM 등) + 형식·내용 검증 실패 → 재시도
                last_err = e
                print(f"[LLM-FUSE] attempt {attempt + 1} 실패: {str(e)[:140]}")
                # 재시도 사이 GPU 캐시 비우기 — 이전 시도의 activation/단편화가 다음 시도의
                # OOM을 유발하지 않게 한다(CLIP4 실측: 1차 시도 직후 캐시 미정리 상태로 2차
                # 시도가 시작돼 6.88GiB 요청에 6.84GiB만 가용해 OOM).
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if raw is not None:
                    # 생성 자체는 됐지만 검증 실패(탈선/스키마/발화누락)한 경우만 직전 출력을
                    # 대화 맥락에 남겨 재작성을 유도한다. raw=None(생성 자체가 예외로 실패,
                    # 예: OOM)이면 남길 출력이 없으니 messages는 그대로 두고 온도만 올려 재생성.
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content":
                            f"출력 검증 실패: {e}. 스키마를 정확히 지키고, 입력의 발화"
                            f"(repeat_hallucination=false)를 하나도 빠짐없이 diagnostic_steps의 "
                            f"source_utterance에 원문 그대로 반영해 JSON만 다시 출력하라."},
                    ]
        raise RuntimeError(f"LLM 융합 검증 {self.max_retries+1}회 실패: {last_err}")

    def _infer(self, messages: List[Dict[str, str]], temperature: float | None = None) -> str:
        """텍스트 추론. vLLM chat (Turing 플래그는 _load 에서 적용).

        temperature: 재시도 다변화용 오버라이드(2026-07-06). None이면 self.temperature —
        1차 시도와 기존 호출 경로의 동작은 그대로다."""
        temp = self.temperature if temperature is None else temperature
        if self.backend == "vllm":
            from vllm import SamplingParams  # noqa
            sp = SamplingParams(temperature=temp, max_tokens=self.max_new_tokens)
            out = self._model.chat(messages, sp)
            return out[0].outputs[0].text
        elif self.backend == "hf_transformers":
            import torch  # noqa
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # device="auto"(멀티 GPU 분산)면 ".to('auto')"가 안 통함 → 모델의 첫 레이어 디바이스로 보냄
            target_device = self._model.device if self.device == "auto" else self.device
            inputs = self._tok(text, return_tensors="pt").to(target_device)
            # 2026-07-06: attn_implementation="sdpa"를 줘도 이 조합(nf4+auto 분산)에서 O(n²)
            # math 커널로 폴백되는 것을 실측(6,330토큰 prefill에서 GPU1 활성화 ~14GiB —
            # math-attention 워크스페이스 크기와 일치, CLIP4 OOM의 실제 원인). 메모리 효율
            # 커널을 명시 강제한다(Turing sm75에서 EFFICIENT_ATTENTION 지원, 수치 결과 동일).
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                               SDPBackend.MATH]):  # MATH는 최후 폴백용
                gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                           do_sample=temp > 0, temperature=temp)
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
