"""
Qwen3-VL-8B-Instruct 단일 이미지 추론 — VLM 비교용.

인터페이스:
  단일 프레임: --image <jpg> --yolo_json <영상 전체 탐지 json> --prompt_file <공통 프롬프트>
  배치(영상 전체): --frames_dir <프레임 폴더> --yolo_json <같은 영상 탐지 json> --prompt_file ...
    → 모델을 1번만 로드하고 프레임 전체를 순회(로딩 비용 상각).

출력: results/qwen3vl/<video_id>_<frame_idx>.json
  {video_id, frame_idx, frame_file, model_response_raw, parsed,
   inference_time_sec, gpu_memory_peak_gib, attn_implementation}

flash-attn 설치 여부를 자동 감지해 attn_implementation을 고른다(설치 안 됐으면 sdpa).
video 처리 코드는 없음 — 이미지 단건 입력 전용(decord 미설치 전제).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_SUBDIR = "qwen3vl"


def detect_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_prompt_template(prompt_file: str) -> tuple[str, str]:
    """common_prompt.md를 '## System Prompt' / '## User Prompt' 섹션으로 분리해서 반환."""
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


def fill_user_prompt(
    template: str, video_id: str, frame_idx: int, timestamp_sec: float,
    detections_text: str,
) -> str:
    """literal '{...}' JSON 스키마 블록이 템플릿 안에 있어서 str.format()은 위험 —
    지정된 placeholder만 정확히 replace()로 채운다."""
    out = template
    out = out.replace("{video_id}", video_id)
    out = out.replace("{frame_idx_list}", f"[{frame_idx}]")
    out = out.replace("{timestamp_list}", f"[{timestamp_sec}]")
    out = out.replace("{detections_formatted_per_frame}", detections_text)
    out = out.replace("{n}", "1")
    out = out.replace("{multi_frame_note}", "")  # 단일 프레임이라 멀티프레임 안내 불필요
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


class Qwen3VLRunner:
    def __init__(self, device: str = "cuda:0", max_new_tokens: int = 512):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = detect_attn_implementation()
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        print(f"[qwen3vl] 모델 로딩 중... (attn_implementation={self.attn_implementation})")
        t0 = time.monotonic()
        self._processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, device_map=self.device,
            attn_implementation=self.attn_implementation,
            quantization_config=quant_config,
        )
        print(f"[qwen3vl] 로딩 완료 ({time.monotonic() - t0:.1f}s)")

    def unload(self) -> None:
        import gc
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run_one(self, image_path: str, system_text: str, user_text: str) -> Dict[str, Any]:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ]},
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        elapsed = time.monotonic() - t0
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3

        raw = self._processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return {"raw": raw, "elapsed": elapsed, "peak_gib": peak_gib}


def process_frame(
    runner: Qwen3VLRunner, image_path: Path, video_id: str, frame_idx: int,
    timestamp_sec: float, detections: List[Dict[str, Any]], frame_width: int,
    frame_height: int, system_text: str, user_template: str, output_dir: Path,
) -> None:
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
    print(f"[qwen3vl] {video_id} frame_idx={frame_idx}: {result['elapsed']:.1f}s, "
          f"peak={result['peak_gib']:.2f}GiB -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-VL-8B 단일/배치 이미지 추론 (VLM 비교용)")
    ap.add_argument("--image", help="키프레임 경로(단일 모드)")
    ap.add_argument("--frames_dir", help="프레임 폴더(배치 모드, 이 영상의 모든 프레임)")
    ap.add_argument("--yolo_json", required=True, help="이 영상의 YOLO 탐지결과 json(전체 프레임 포함)")
    ap.add_argument("--prompt_file", required=True, help="공통 프롬프트 템플릿(common_prompt.md)")
    ap.add_argument("--output_dir", default=None, help="결과 저장 폴더(기본: results/qwen3vl)")
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

    runner = Qwen3VLRunner(device=args.device, max_new_tokens=args.max_new_tokens)
    runner.load()

    try:
        if args.image:
            image_path = Path(args.image)
            m = re.search(r"frame_(\d+)\.jpg$", image_path.name)
            if not m:
                raise ValueError(f"파일명에서 프레임 번호를 못 찾음: {image_path.name}")
            frame_idx = int(m.group(1)) - 1  # frame_0001.jpg -> frame_idx=0
            fdata = frames_by_idx.get(frame_idx, {"timestamp_sec": None, "detections": []})
            process_frame(runner, image_path, video_id, frame_idx, fdata["timestamp_sec"],
                          fdata["detections"], frame_width, frame_height,
                          system_text, user_template, output_dir)
        else:
            frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
            if args.limit:
                frame_paths = frame_paths[: args.limit]
            print(f"[qwen3vl] 배치 모드: {video_id} {len(frame_paths)}프레임")
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
