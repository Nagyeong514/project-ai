"""파일럿 프레임 17장에 YOLO(best.pt, dell7920_v1) CPU 검출 → detections/pilot.json.

로그인 노드 CPU에서 실행(GPU 의존성 제거 목적). 프레임 timestamp는 구간 로컬 기준
(0,2,...,32s — 절대시각은 +80s, metadata에 기록).
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
WEIGHTS = "/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/260704_VLM/dell7920_v1/weights/best.pt"
SEGMENT_START = 80.0
FPS = 0.5

from ultralytics import YOLO

model = YOLO(WEIGHTS)
frames = sorted((BASE / "frames").glob("frame_*.jpg"))
dets = []
for i, fp in enumerate(frames):
    t_local = i / FPS  # 0,2,4,...
    r = model.predict(str(fp), conf=0.25, iou=0.7, imgsz=640, device="cpu", verbose=False)[0]
    for b in r.boxes:
        x, y, w, h = b.xywh[0].tolist()
        dets.append({
            "timestamp": t_local, "frame_idx": i, "frame_file": fp.name,
            "cls": r.names[int(b.cls[0])], "conf": float(b.conf[0]),
            "bbox": {"x": x - w / 2, "y": y - h / 2, "w": w, "h": h},
        })

out = {
    "video_id": "CLIP2_0704.mp4",
    "segment_abs_sec": [SEGMENT_START, SEGMENT_START + (len(frames) - 1) / FPS],
    "n_frames": len(frames),
    "fps": FPS,
    "weights": WEIGHTS,
    "detections": dets,
}
(BASE / "detections").mkdir(exist_ok=True)
(BASE / "detections" / "pilot.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
classes = sorted({d["cls"] for d in dets})
print(f"[OK] 검출 {len(dets)}건, 클래스: {classes} → detections/pilot.json")
