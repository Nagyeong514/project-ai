"""Qwen3-VL-8B-Instruct 파일럿 — STEP4 본선과 동일한 native video 입력.

로딩/생성 방식은 안나경 STEP4 vlm_qwen.py를 참조해 동등하게 구성:
NF4 4bit(fp16 compute), max_pixels=192², do_sample_frames=False + video_metadata 실측 fps,
greedy + repetition_penalty 1.2, max_new_tokens 4000, 빈 관찰 시 temp 0.3 1회 재시도.
실행 env: vlm_compare/env_qwen3vl
"""
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pilot_common import (Timer, build_texts, load_images_resized, load_inputs,
                          parse_observations, save_result)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def main():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    frames, times, parts, _ = load_inputs()
    system_text, user_text = build_texts(parts, times[-1])
    # 본선 파라미터 max_pixels=192²를 코드에서 직접 적용(OOM 재발 방지, 2026-07-07 실측)
    images = load_images_resized(frames, max_pixels=192 * 192)
    print(f"[INPUT] 프레임 {len(images)}장, 축소 후 크기 {images[0].size}")

    with Timer() as t_load:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, max_pixels=192 * 192)
        # transformers 5.13: 비디오 경로는 video_processor 설정을 따로 봄 — 방어적으로 통일
        for attr in ("image_processor", "video_processor"):
            sub = getattr(processor, attr, None)
            if sub is not None and hasattr(sub, "max_pixels"):
                sub.max_pixels = 192 * 192
            if sub is not None and hasattr(sub, "size") and isinstance(sub.size, dict):
                if "longest_edge" in sub.size:
                    sub.size["longest_edge"] = 192 * 192
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, device_map="cuda:0", attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        ).eval()

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [{"type": "video", "video": images},
                                     {"type": "text", "text": user_text}]},
    ]
    real_fps = 1.0 / (times[1] - times[0])
    video_metadata = {
        "total_num_frames": len(images), "fps": real_fps,
        "frames_indices": list(range(len(images))), "duration": times[-1],
    }

    def generate(sample_override=False):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            do_sample_frames=False, video_metadata=video_metadata,
        ).to("cuda:0")
        vgt = inputs.get("video_grid_thw")
        if vgt is not None:
            T = int(vgt[0][0])
            assert T >= max(1, len(images) // 2), f"프레임 축소 버그: T={T}, frames={len(images)}"
        kw = dict(max_new_tokens=4000, repetition_penalty=1.2)
        if sample_override:
            kw.update(do_sample=True, temperature=0.3, top_p=0.9)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            gen = model.generate(**inputs, **kw)
        return processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    torch.cuda.reset_peak_memory_stats()
    retry = False
    with Timer() as t_inf:
        raw = generate()
        obs = parse_observations(raw)
        if not obs:
            print("[RETRY] 관찰 0건 → temp 0.3 재시도")
            retry = True
            raw = generate(sample_override=True)
            obs = parse_observations(raw)
    peak = torch.cuda.max_memory_allocated() / 1024**3

    save_result("qwen3vl", raw, obs or [], {
        "model_name": MODEL_NAME, "n_frames": len(images), "injected_parts": parts,
        "load_time": t_load.elapsed, "infer_time": t_inf.elapsed,
        "peak_gib": peak, "retry_used": retry,
    })


if __name__ == "__main__":
    main()
