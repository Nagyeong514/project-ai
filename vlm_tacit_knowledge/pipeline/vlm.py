"""VLM 추론 — Qwen2.5-VL (로컬, 폐쇄망).

backend='stub' : 가중치 없을 때. 입력만 받아 가짜 출력 → 파이프라인 흐름 검증용.
backend='qwen' : 가중치 도착 후. 실제 추론.

가중치 오면 config.yaml 의 vlm.backend 를 'qwen'으로만 바꾸면 됨.
"""
from __future__ import annotations
import json
from functools import lru_cache

from .prompt import build_messages


# ── STUB ──────────────────────────────────────────────────
def _infer_stub(stt_text, frame_paths, cfg) -> dict:
    """모델 없이 형식만 맞춘 가짜 출력. 채점 하니스까지 흐름 확인용."""
    return {
        "knowledge_points": [
            {
                "action": f"[STUB] 프레임 {len(frame_paths)}장 기반 행동 서술",
                "tacit": "[STUB] 여기에 판단/감각/요령이 들어갈 자리",
                "evidence": f"[STUB] stt='{(stt_text or '')[:30]}...', frames={len(frame_paths)}",
            }
        ],
        "_stub": True,
    }


# ── QWEN ──────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_qwen(model_path: str, load_in_4bit: bool):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    kw = {"torch_dtype": torch.float16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kw)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def _infer_qwen(stt_text, frame_paths, cfg) -> dict:
    from qwen_vl_utils import process_vision_info
    v = cfg["vlm"]
    model, processor = _load_qwen(v["model_path"], v["load_in_4bit"])
    messages = build_messages(stt_text, frame_paths)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=v["max_new_tokens"],
                         do_sample=v["temperature"] > 0, temperature=max(v["temperature"], 1e-6))
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return _parse_json(out)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:  # 코드펜스 제거
        raw = raw.split("```")[1].lstrip("json").strip() if raw.count("```") >= 2 else raw
    try:
        return json.loads(raw)
    except Exception:
        # JSON 실패 시 원문 보존 (채점에서 환각/형식오류로 처리)
        return {"knowledge_points": [], "_parse_error": True, "_raw": raw}


# ── 디스패치 ───────────────────────────────────────────────
def infer(stt_text: str | None, frame_paths: list[str], cfg: dict) -> dict:
    backend = cfg["vlm"]["backend"]
    if backend == "stub":
        return _infer_stub(stt_text, frame_paths, cfg)
    if backend == "qwen":
        return _infer_qwen(stt_text, frame_paths, cfg)
    raise ValueError(f"unknown vlm.backend: {backend}")
