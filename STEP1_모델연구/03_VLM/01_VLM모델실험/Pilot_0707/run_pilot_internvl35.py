"""InternVL3.5-8B 파일럿 — STEP4 관찰 프롬프트를 멀티프레임(Frame-i: <image>)으로 등가 적용.

native video 미지원이라 InternVL 공식 비디오 패턴을 사용: 프레임당 448px 타일 1개
(max_num=1), num_patches_list로 프레임 경계 전달. 로딩은 vlm_compare/inference/
run_internvl35.py의 InternVLRunner(NF4 4bit, transformers 4.57.6 검증본)를 재사용.
실행 env: vlm_compare/env_internvl35
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/inference")

from pilot_common import Timer, build_texts, load_inputs, parse_observations, save_result
from run_internvl35 import InternVLRunner, load_image_as_pixel_values

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"


def main():
    import torch

    frames, times, parts, _ = load_inputs()
    system_text, user_text = build_texts(parts, times[-1])

    runner = InternVLRunner(device="cuda:0", max_new_tokens=4000)
    with Timer() as t_load:
        runner.load()

    # 프레임당 1타일(448px) — 공식 비디오 예제 방식(멀티프레임에서 타일 폭증 방지)
    pv_list = [load_image_as_pixel_values(str(p), max_num=1) for p in frames]
    pixel_values = torch.cat(pv_list).to(torch.bfloat16).to(runner._model.device)
    num_patches_list = [pv.size(0) for pv in pv_list]
    frame_prefix = "".join(
        f"Frame-{i+1} (t={t:.0f}s): <image>\n" for i, t in enumerate(times))
    question = f"{system_text}\n\n{frame_prefix}\n{user_text}"

    def generate(sample_override=False):
        cfg = dict(max_new_tokens=4000, do_sample=False)
        if sample_override:
            cfg = dict(max_new_tokens=4000, do_sample=True, temperature=0.3, top_p=0.9)
        with torch.no_grad():
            return runner._model.chat(
                runner._tokenizer, pixel_values, question, cfg,
                num_patches_list=num_patches_list)

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

    save_result("internvl35", raw, obs or [], {
        "model_name": MODEL_NAME, "n_frames": len(frames), "injected_parts": parts,
        "load_time": t_load.elapsed, "infer_time": t_inf.elapsed,
        "peak_gib": peak, "retry_used": retry,
    })


if __name__ == "__main__":
    main()
