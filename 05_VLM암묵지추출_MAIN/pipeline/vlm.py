"""VLM 추론 — Qwen3-VL / Qwen2.5-VL (로컬, 폐쇄망). 클래스는 model_path로 자동 선택.

backend='stub' : 가중치 없을 때. 입력만 받아 가짜 출력 → 파이프라인 흐름 검증용.
backend='qwen' : 가중치 도착 후. 실제 추론.

가중치 오면 config.yaml 의 vlm.backend 를 'qwen'으로만 바꾸면 됨.
"""
from __future__ import annotations
import json
from functools import lru_cache

from .prompt import build_messages, build_text, SYSTEM


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
def _load_qwen(model_path: str, load_in_4bit: bool, min_pixels: int, max_pixels: int):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    # AutoModelForImageTextToText: config의 architectures로 클래스 자동 선택
    # → Qwen2.5-VL / Qwen3-VL 모두 동일 코드로 로드 (2026-06-29 8B 업그레이드 대응)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = {"dtype": torch.float16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kw)
    # max_pixels: 이미지당 토큰 수 상한 → 비전 인코더 OOM 방지의 핵심 레버
    processor = AutoProcessor.from_pretrained(model_path,
                                              min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


def _infer_qwen(stt_text, frame_paths, cfg) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info
    v = cfg["vlm"]
    torch.manual_seed(v.get("seed", 0))  # 재현성(4bit라 완전 보장은 아님)
    model, processor = _load_qwen(v["model_path"], v["load_in_4bit"],
                                  v["min_pixels"], v["max_pixels"])
    messages = build_messages(stt_text, frame_paths)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs or None, videos=video_inputs or None,
                       padding=True, return_tensors="pt").to(model.device)
    gen_kw = {
        "max_new_tokens": v["max_new_tokens"],
        "repetition_penalty": v.get("repetition_penalty", 1.05),  # 약한 반복 억제
    }
    nrn = v.get("no_repeat_ngram_size", 0)
    if nrn:  # 0이면 비활성 (JSON 키 반복 보존 위해 끄는 게 기본)
        gen_kw["no_repeat_ngram_size"] = nrn
    if v["temperature"] > 0:
        gen_kw.update(do_sample=True, temperature=v["temperature"])
    else:
        gen_kw.update(do_sample=False)
    gen = model.generate(**inputs, **gen_kw)
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return _parse_json(out)


def _parse_json(raw: str) -> dict:
    import re
    s = (raw or "").strip()
    if not s:
        return {"knowledge_points": [], "_parse_error": True, "_raw": ""}
    if "```" in s:  # 코드펜스 제거
        s = s.split("```")[1]
        s = s[4:].strip() if s.lower().startswith("json") else s.strip()
    s = re.sub(r"//.*?$", "", s, flags=re.M)            # JS 주석 제거
    # 키 오타/변형 정규화 (no_repeat_ngram 부작용 잔재 대비)
    s = re.sub(r'"(?:tac\w*|tactic\w*)"\s*:', '"tacit":', s)
    s = re.sub(r'"(?:action\w+)"\s*:', '"action":', s)
    s = re.sub(r'"(?:evidence\w*|evides|证据)"\s*:', '"evidence":', s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # 잘린/깨진 JSON 복구: 항목을 정규식으로 건져냄 (tacit 키 변형 허용)
    pts = []
    for m in re.finditer(
        r'"action"\s*:\s*"(.*?)"\s*,\s*"tacit"\s*:\s*"(.*?)"'
        r'(?:\s*,\s*"evidence"\s*:\s*"(.*?)")?', s, re.S):
        pts.append({"action": m.group(1), "tacit": m.group(2), "evidence": m.group(3) or ""})
    if pts:
        return {"knowledge_points": pts, "_recovered": True}
    return {"knowledge_points": [], "_parse_error": True, "_raw": s[:500]}


# ── vLLM (OpenAI 호환 서버) — InternVL2.5 / MiniCPM-V 비교용 ──
# Turing(RTX2080)에선 InternVL-8B fp16=OOM, 4bit는 커스텀코드 비호환 → vLLM 서빙(AWQ/bnb)로 우회.
# 프레임을 base64 data-URI 이미지로 /v1/chat/completions 에 전송. 프롬프트·파서는 qwen 경로와 동일.
def _img_data_uri(path: str) -> str:
    import base64, mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _infer_vllm(stt_text, frame_paths, cfg) -> dict:
    from openai import OpenAI
    v = cfg["vlm"]
    client = OpenAI(base_url=v["vllm_base_url"], api_key="EMPTY")
    content = [{"type": "text", "text": build_text(stt_text, bool(frame_paths))}]
    for p in frame_paths:
        content.append({"type": "image_url", "image_url": {"url": _img_data_uri(p)}})
    resp = client.chat.completions.create(
        model=v["vllm_served_name"],
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": content}],
        max_tokens=v["max_new_tokens"],
        temperature=v["temperature"],
    )
    return _parse_json(resp.choices[0].message.content or "")


# ── INTERNVL (다른 모델 probe) ─────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@lru_cache(maxsize=1)
def _load_internvl(model_path: str, load_in_4bit: bool):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel
    # transformers 5.x는 InternVL 옛 커스텀코드에 없는 all_tied_weights_keys를 로딩 곳곳에서
    # 참조 → 클래스 기본값 주입으로 우회. device_map/4bit도 회피하고 fp16 단일 GPU 로드.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float16, low_cpu_mem_usage=True
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tok


def _preprocess_internvl(path, size):
    """단일 타일(동적 타일링 X)로 전처리 — 8GB 메모리 절약 + Qwen 세팅과 비교 가능."""
    import torch
    import torchvision.transforms as T
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    tf = T.Compose([T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    return tf(img).unsqueeze(0).to(torch.float16)


def _infer_internvl(stt_text, frame_paths, cfg) -> dict:
    import torch
    v = cfg["vlm"]
    torch.manual_seed(v.get("seed", 0))
    model, tok = _load_internvl(v["internvl_model_path"], v["load_in_4bit"])
    size = v.get("internvl_input_size", 448)
    gen_cfg = {"max_new_tokens": v["max_new_tokens"],
               "do_sample": v["temperature"] > 0,
               "repetition_penalty": v.get("repetition_penalty", 1.05)}
    if v["temperature"] > 0:
        gen_cfg["temperature"] = v["temperature"]
    text = SYSTEM + "\n\n" + build_text(stt_text, bool(frame_paths))

    if frame_paths:
        pv = torch.cat([_preprocess_internvl(p, size) for p in frame_paths]).to(model.device)
        num_patches = [1] * len(frame_paths)
        prefix = "".join(f"Image-{i+1}: <image>\n" for i in range(len(frame_paths)))
        out = model.chat(tok, pv, prefix + text, gen_cfg, num_patches_list=num_patches)
    else:  # text-only
        out = model.chat(tok, None, text, gen_cfg)
    return _parse_json(out)


# ── 디스패치 ───────────────────────────────────────────────
def infer(stt_text: str | None, frame_paths: list[str], cfg: dict) -> dict:
    backend = cfg["vlm"]["backend"]
    if backend == "stub":
        return _infer_stub(stt_text, frame_paths, cfg)
    if backend == "qwen":
        return _infer_qwen(stt_text, frame_paths, cfg)
    if backend == "vllm":
        return _infer_vllm(stt_text, frame_paths, cfg)
    if backend == "internvl":
        return _infer_internvl(stt_text, frame_paths, cfg)
    raise ValueError(f"unknown vlm.backend: {backend}")
