"""MiniCPM-V 4.5 파일럿 — STEP4 관찰 프롬프트를 프레임 리스트 입력으로 등가 적용.

msgs의 user content에 PIL 프레임 17장 + 텍스트를 나란히 넣는 MiniCPM-V 멀티이미지
방식. 로딩은 vlm_compare/inference/run_minicpm45.py의 MiniCPMRunner
(NF4 4bit, resampler 제외 — 실측 우회 포함)를 재사용.
실행 env: vlm_compare/env_minicpm45
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/inference")

from pilot_common import (Timer, build_texts, load_images_resized, load_inputs,
                          parse_observations, save_result)
from run_minicpm45 import MiniCPMRunner

MODEL_NAME = "openbmb/MiniCPM-V-4_5"


def main():
    import torch

    frames, times, parts, _ = load_inputs()
    system_text, user_text = build_texts(parts, times[-1])
    # 448픽셀급으로 축소 + 아래 chat에서 max_slice_nums=1 — 원본(1080×1920)을 그대로 주면
    # 프레임당 다중 슬라이스로 토큰 폭증 → attention 51GiB OOM (2026-07-07 실측)
    images = load_images_resized(frames, max_pixels=448 * 448)
    print(f"[INPUT] 프레임 {len(images)}장, 축소 후 크기 {images[0].size}")

    runner = MiniCPMRunner(device="cuda:0", max_new_tokens=4000)
    with Timer() as t_load:
        runner.load()

    header = "다음은 영상에서 2초 간격으로 추출한 프레임 17장이다(시간순)."
    msgs = [{"role": "user", "content": [header, *images, user_text]}]

    def generate(sampling=False):
        with torch.no_grad():
            return runner._model.chat(
                image=None, msgs=msgs, tokenizer=runner._tokenizer,
                system_prompt=system_text, sampling=sampling,
                temperature=0.3 if sampling else None,
                max_slice_nums=1,  # 프레임당 슬라이스 1개 고정(멀티프레임 토큰 폭증 방지)
                max_new_tokens=4000)

    torch.cuda.reset_peak_memory_stats()
    retry = False
    with Timer() as t_inf:
        raw = generate()
        obs = parse_observations(raw)
        if not obs:
            print("[RETRY] 관찰 0건 → 저온 샘플링 재시도")
            retry = True
            raw = generate(sampling=True)
            obs = parse_observations(raw)
    peak = torch.cuda.max_memory_allocated() / 1024**3

    save_result("minicpm45", raw, obs or [], {
        "model_name": MODEL_NAME, "n_frames": len(images), "injected_parts": parts,
        "load_time": t_load.elapsed, "infer_time": t_inf.elapsed,
        "peak_gib": peak, "retry_used": retry,
    })


if __name__ == "__main__":
    main()
