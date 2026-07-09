"""YOLO 서브프로세스 엔트리. 이 프로세스 안에서만 ultralytics 를 import 한다.

CUDA_VISIBLE_DEVICES 오염이 여기서 일어나도 프로세스가 끝나면 같이 사라진다.
단독 실행도 가능(디버깅용):
  python stages/s3_yolo_entry.py --frames-dir ... --times-json ... --weights ... --out ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--times-json", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO  # 자식 프로세스 안에서만

    times = json.loads(Path(args.times_json).read_text(encoding="utf-8"))["times"]
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg")))

    model = YOLO(args.weights)
    names = model.names
    dets = []
    for i in range(0, len(frames), args.batch):  # 배치 predict (프레임별 순차 호출 폐지)
        batch = frames[i:i + args.batch]
        results = model.predict(batch, imgsz=args.imgsz, conf=args.conf, verbose=False)
        for j, res in enumerate(results):
            fidx = i + j
            for b in res.boxes:
                dets.append({
                    "timestamp": times[fidx] if fidx < len(times) else 0.0,
                    "frame_idx": fidx,
                    "cls": names[int(b.cls[0])],
                    "conf": float(b.conf[0]),
                })

    out = {"video_id": Path(args.frames_dir).name, "n": len(dets), "detections": dets}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S3-child] {len(dets)} detections → {args.out}")


if __name__ == "__main__":
    main()
