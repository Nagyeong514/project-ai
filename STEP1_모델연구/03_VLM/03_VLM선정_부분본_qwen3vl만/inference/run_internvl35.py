"""
InternVL3.5-8B 단일 이미지 추론 — VLM 비교용 (run_qwen3vl.py와 CLI/출력 형식 동일).

InternVL은 Qwen3-VL과 로딩·추론 방식이 다르다:
  - trust_remote_code=True로 자체 모델링 코드를 사용(AutoModel, 표준 ConditionalGeneration 클래스 없음)
  - 이미지를 고정 크기로 리사이즈하지 않고, 종횡비에 맞춰 여러 448x448 타일로 쪼개는
    "dynamic preprocessing"을 씀(공식 레포 예제 코드 그대로 이식)
  - generate()가 아니라 model.chat(tokenizer, pixel_values, question, generation_config) 사용
  - system prompt는 chat() 인자가 아니라 question 앞에 직접 붙인다(공식 예제 방식).

video 처리 코드는 없음 — 이미지 단건 입력 전용(decord 미설치 전제).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"
OUTPUT_SUBDIR = "internvl35"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def detect_attn_implementation() -> str:
    """InternVLChatModel의 원격코드는 sdpa 디스패치를 아직 지원하지 않는다(실측 확인,
    2026-07-05: "InternVLChatModel does not support an attention implementation
    through torch.nn.functional.scaled_dot_product_attention yet... load your model
    with the argument attn_implementation='eager' meanwhile"). Qwen3-VL과 달리
    flash-attn 없을 때 fallback은 sdpa가 아니라 eager여야 한다."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


def load_prompt_template(prompt_file: str) -> tuple[str, str]:
    text = Path(prompt_file).read_text(encoding="utf-8")
    sys_match = re.search(r"## System Prompt\s*\n(.*?)\n## User Prompt", text, re.DOTALL)
    user_match = re.search(r"## User Prompt\s*\n(.*)", text, re.DOTALL)
    if not sys_match or not user_match:
        raise ValueError(f"{prompt_file}에서 '## System Prompt'/'## User Prompt' 섹션을 못 찾음")
    return sys_match.group(1).strip(), user_match.group(1).strip()


def format_detections(detections: List[Dict[str, Any]], frame_width: int, frame_height: int) -> str:
    if not detections:
        return f"(프레임 크기 {frame_width}x{frame_height}px, YOLO 검출 없음)"
    lines = [f"프레임 크기: {frame_width}x{frame_height}px (bbox는 이 기준 픽셀 좌표)"]
    for d in detections:
        b = d["bbox"]
        lines.append(
            f"- {d['cls']} (conf={d['conf']:.2f}, "
            f"bbox=[x={b['x']:.0f}, y={b['y']:.0f}, w={b['w']:.0f}, h={b['h']:.0f}])"
        )
    return "\n".join(lines)


def fill_user_prompt(template: str, video_id: str, frame_idx: int, timestamp_sec: float,
                      detections_text: str) -> str:
    out = template
    out = out.replace("{video_id}", video_id)
    out = out.replace("{frame_idx_list}", f"[{frame_idx}]")
    out = out.replace("{timestamp_list}", f"[{timestamp_sec}]")
    out = out.replace("{detections_formatted_per_frame}", detections_text)
    out = out.replace("{n}", "1")
    out = out.replace("{multi_frame_note}", "")
    return out


def parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


# ── InternVL 공식 예제의 dynamic preprocessing(타일링) 그대로 이식 ──────────
def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_as_pixel_values(image_path: str, input_size: int = 448, max_num: int = 12):
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


class InternVLRunner:
    def __init__(self, device: str = "cuda:0", max_new_tokens: int = 512):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = detect_attn_implementation()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        print(f"[internvl35] 모델 로딩 중... (attn_implementation={self.attn_implementation})")
        t0 = time.monotonic()
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, use_fast=False,
        )
        self._model = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, device_map=self.device,
            quantization_config=quant_config, attn_implementation=self.attn_implementation,
        ).eval()
        print(f"[internvl35] 로딩 완료 ({time.monotonic() - t0:.1f}s)")

    def unload(self) -> None:
        import gc
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run_one(self, image_path: str, system_text: str, user_text: str) -> Dict[str, Any]:
        import torch

        pixel_values = load_image_as_pixel_values(image_path).to(
            torch.bfloat16
        ).to(self._model.device)
        # InternVL 공식 예제 방식: system prompt는 별도 인자가 없어 question 앞에 붙인다.
        question = f"{system_text}\n\n<image>\n{user_text}"
        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.no_grad():
            response = self._model.chat(self._tokenizer, pixel_values, question, generation_config)
        elapsed = time.monotonic() - t0
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        return {"raw": response, "elapsed": elapsed, "peak_gib": peak_gib}


def process_frame(runner: InternVLRunner, image_path: Path, video_id: str, frame_idx: int,
                   timestamp_sec: float, detections: List[Dict[str, Any]], frame_width: int,
                   frame_height: int, system_text: str, user_template: str,
                   output_dir: Path) -> None:
    det_text = format_detections(detections, frame_width, frame_height)
    user_text = fill_user_prompt(user_template, video_id, frame_idx, timestamp_sec, det_text)
    result = runner.run_one(str(image_path), system_text, user_text)
    parsed = parse_json_response(result["raw"])

    out = {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "frame_file": image_path.name,
        "model_response_raw": result["raw"],
        "parsed": parsed,
        "inference_time_sec": round(result["elapsed"], 2),
        "gpu_memory_peak_gib": round(result["peak_gib"], 2),
        "attn_implementation": runner.attn_implementation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_{frame_idx}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[internvl35] {video_id} frame_idx={frame_idx}: {result['elapsed']:.1f}s, "
          f"peak={result['peak_gib']:.2f}GiB -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="InternVL3.5-8B 단일/배치 이미지 추론 (VLM 비교용)")
    ap.add_argument("--image", help="키프레임 경로(단일 모드)")
    ap.add_argument("--frames_dir", help="프레임 폴더(배치 모드, 이 영상의 모든 프레임)")
    ap.add_argument("--yolo_json", required=True, help="이 영상의 YOLO 탐지결과 json(전체 프레임 포함)")
    ap.add_argument("--prompt_file", required=True, help="공통 프롬프트 템플릿(common_prompt.md)")
    ap.add_argument("--output_dir", default=None, help="결과 저장 폴더(기본: results/internvl35)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="배치 모드에서 앞에서부터 N프레임만(테스트용)")
    args = ap.parse_args()

    if not args.image and not args.frames_dir:
        ap.error("--image 또는 --frames_dir 중 하나는 필요합니다.")

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir.parent / "results" / OUTPUT_SUBDIR

    system_text, user_template = load_prompt_template(args.prompt_file)
    yolo_data = json.load(open(args.yolo_json, encoding="utf-8"))
    video_id = yolo_data["video_id"]
    frame_width, frame_height = yolo_data["frame_width"], yolo_data["frame_height"]
    frames_by_idx = {f["frame_idx"]: f for f in yolo_data["frames"]}

    runner = InternVLRunner(device=args.device, max_new_tokens=args.max_new_tokens)
    runner.load()

    try:
        if args.image:
            image_path = Path(args.image)
            m = re.search(r"frame_(\d+)\.jpg$", image_path.name)
            if not m:
                raise ValueError(f"파일명에서 프레임 번호를 못 찾음: {image_path.name}")
            frame_idx = int(m.group(1)) - 1
            fdata = frames_by_idx.get(frame_idx, {"timestamp_sec": None, "detections": []})
            process_frame(runner, image_path, video_id, frame_idx, fdata["timestamp_sec"],
                          fdata["detections"], frame_width, frame_height,
                          system_text, user_template, output_dir)
        else:
            frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
            if args.limit:
                frame_paths = frame_paths[: args.limit]
            print(f"[internvl35] 배치 모드: {video_id} {len(frame_paths)}프레임")
            for image_path in frame_paths:
                m = re.search(r"frame_(\d+)\.jpg$", image_path.name)
                frame_idx = int(m.group(1)) - 1
                fdata = frames_by_idx.get(frame_idx, {"timestamp_sec": None, "detections": []})
                process_frame(runner, image_path, video_id, frame_idx, fdata["timestamp_sec"],
                              fdata["detections"], frame_width, frame_height,
                              system_text, user_template, output_dir)
    finally:
        runner.unload()


if __name__ == "__main__":
    main()
