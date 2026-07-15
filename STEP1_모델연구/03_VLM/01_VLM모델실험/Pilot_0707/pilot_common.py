"""파일럿 공통: 프레임/검출 로드 + STEP4 관찰 프롬프트 조립 + 결과 저장.

프롬프트는 prompt_step4_observation.py(본선 STEP4 사본)의 build_video_observation_messages()
를 그대로 쓰고, 본선 런타임(vlm_qwen.py)이 덧붙이는 '영상 길이/timestamp 기준' 문장을
동일하게 추가한다. 구간 33초 = 단일 청크 → 호출 1회.
"""
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
FPS = 0.5
VIDEO_ID = "CLIP2_0704.mp4"
SEGMENT = "80-113"


def seconds_to_hhmmss(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_inputs():
    frames = sorted((BASE / "frames").glob("frame_*.jpg"))
    times = [i / FPS for i in range(len(frames))]  # 구간 로컬 0,2,...s
    det = json.loads((BASE / "detections" / "pilot.json").read_text(encoding="utf-8"))
    parts = sorted({d["cls"] for d in det["detections"]})
    return frames, times, parts, det


def load_images_resized(frames, max_pixels):
    """원본 1080×1920 프레임을 총 픽셀수 max_pixels 이하로 축소해 PIL로 반환.

    2026-07-07 실측: 원본 해상도 그대로 넣으면 vision 토큰 폭증으로 attention OOM
    (Qwen 24.6GiB / MiniCPM 51GiB 할당 시도). 본선 STEP4의 max_pixels=192² 축소가
    transformers 5.13 비디오 경로에는 적용되지 않아 코드에서 직접 줄인다.
    """
    from PIL import Image
    out = []
    for p in frames:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            # 28의 배수로 내림(Qwen 패치 크기 정합; 다른 모델에도 무해)
            nw = max(28, int(w * scale) // 28 * 28)
            nh = max(28, int(h * scale) // 28 * 28)
            im = im.resize((nw, nh), Image.BICUBIC)
        out.append(im)
    return out


def build_texts(parts, duration_sec):
    import sys
    sys.path.insert(0, str(BASE))
    from prompt_step4_observation import build_video_observation_messages
    base = build_video_observation_messages(parts)
    system_text = base[0]["content"]
    user_text = base[1]["content"] + (
        f"\n\n이 영상의 길이는 약 {seconds_to_hhmmss(duration_sec)}이다. "
        "timestamp는 영상 시작(00:00:00) 기준 시각을 HH:MM:SS 형식으로 적어라."
    )
    return system_text, user_text


def parse_observations(raw: str):
    """본선과 같은 관용도로 raw에서 observations 배열을 꺼낸다."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
    try:
        obj = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j <= i:
            return None
        try:
            obj = json.loads(s[i:j + 1])
        except Exception:
            return None
    if isinstance(obj, dict) and isinstance(obj.get("observations"), list):
        return obj["observations"]
    return None


def save_result(model_key, raw, observations, meta):
    out = {
        "video_id": VIDEO_ID,
        "segment_abs_sec": [80.0, 113.4],
        "model": meta.get("model_name"),
        "prompt": "STEP4 관찰 프롬프트(본선, prompt_step4_observation.py)",
        "n_frames": meta.get("n_frames"),
        "injected_parts": meta.get("injected_parts"),
        "n_observations": len(observations or []),
        "observations": observations,
        "raw": raw,
        "load_time_sec": round(meta.get("load_time", 0), 2),
        "inference_time_sec": round(meta.get("infer_time", 0), 2),
        "gpu_memory_peak_gib": round(meta.get("peak_gib", 0), 2),
        "retry_used": meta.get("retry_used", False),
    }
    d = BASE / "results" / model_key
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{VIDEO_ID}_{SEGMENT}.observations.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {model_key}: 관찰 {out['n_observations']}건, "
          f"load {out['load_time_sec']}s + infer {out['inference_time_sec']}s, "
          f"peak {out['gpu_memory_peak_gib']}GiB → {p}")


class Timer:
    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *a):
        self.elapsed = time.monotonic() - self.t0
