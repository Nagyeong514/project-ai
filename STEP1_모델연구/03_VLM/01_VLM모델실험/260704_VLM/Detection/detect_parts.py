"""
Sampling/manifest.json에 정리된 4개 클립 프레임(총 326장)에 대해
tacit_pipeline/components/detector_yolo.py(UltralyticsYOLODetector)로
튜닝된 best.pt 부품 검출을 실행한다.

frame_extract.py로 다시 프레임을 뽑지 않고, 이미 만들어둔 jpg 경로 + timestamp를
그대로 FrameRef로 감싸서 재사용한다(Sampling 단계 산출물 재활용).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DETECTION_DIR = Path(__file__).resolve().parent
BASE_DIR = DETECTION_DIR.parent  # .../260704_VLM
MEMBER_DIR = BASE_DIR.parent  # .../오유빈
sys.path.insert(0, str(MEMBER_DIR))

from tacit_pipeline.components.detector_yolo import UltralyticsYOLODetector  # noqa: E402
from tacit_pipeline.interfaces.detector import FrameRef  # noqa: E402
from tacit_pipeline.schema.intermediate import FrameMeta  # noqa: E402

SAMPLING_DIR = BASE_DIR / "Sampling"
MANIFEST_PATH = SAMPLING_DIR / "manifest.json"
WEIGHTS_PATH = BASE_DIR / "dell7920_v1" / "weights" / "best.pt"

OUT_DIR = DETECTION_DIR / "results"
SUMMARY_PATH = DETECTION_DIR / "summary.json"


def main() -> None:
    manifest = json.load(MANIFEST_PATH.open(encoding="utf-8"))

    detector = UltralyticsYOLODetector(
        weights_path=str(WEIGHTS_PATH),
        device="cuda:0",
        conf=0.25,
        iou=0.7,
        imgsz=640,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_summary: dict[str, dict] = {}

    for name, entry in manifest.items():
        video_id = entry["video_id"]
        frame_refs = [
            FrameRef(
                frame_idx=i,
                timestamp=f["timestamp_sec"],
                image=str(SAMPLING_DIR / f["path"]),
            )
            for i, f in enumerate(entry["frames"])
        ]
        meta = FrameMeta(
            video_id=video_id, path="", fps=entry["fps"], n_frames=len(frame_refs),
            width=entry["frame_width"], height=entry["frame_height"],
        )

        detections_by_frame = detector.detect(frame_refs, meta)

        class_counts: dict[str, int] = {}
        frames_out = []
        for fd in detections_by_frame:
            frames_out.append({
                "frame_idx": fd.frame_idx,
                "timestamp_sec": fd.timestamp,
                "detections": [
                    {"cls": d.cls, "conf": round(d.conf, 4),
                     "bbox": {"x": d.bbox.x, "y": d.bbox.y, "w": d.bbox.w, "h": d.bbox.h}}
                    for d in fd.detections
                ],
            })
            for d in fd.detections:
                class_counts[d.cls] = class_counts.get(d.cls, 0) + 1

        out_path = OUT_DIR / f"{video_id}.detections.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "video_id": video_id,
                "n_frames": len(frame_refs),
                # bbox의 x/y/w/h는 전부 이 프레임 크기 기준 픽셀 좌표(정규화 아님).
                # 경계 근처 박스는 x나 y가 1 미만으로 작게 나올 수 있는데, 그건 정규화가
                # 아니라 실제로 가장자리에 거의 붙어있다는 뜻이다 — 헷갈리지 않도록 명시.
                "frame_width": entry["frame_width"],
                "frame_height": entry["frame_height"],
                "bbox_format": "pixel_xywh",
                "frames": frames_out,
            }, f, ensure_ascii=False, indent=2)

        total_dets = sum(class_counts.values())
        overall_summary[name] = {
            "video_id": video_id,
            "n_frames": len(frame_refs),
            "frame_width": entry["frame_width"],
            "frame_height": entry["frame_height"],
            "total_detections": total_dets,
            "class_counts": class_counts,
            "frames_with_zero_detections": sum(1 for fd in detections_by_frame if not fd.detections),
        }
        print(f"[OK] {name}: {len(frame_refs)}프레임 -> 검출 {total_dets}건 -> {out_path}")
        print(f"     클래스별: {class_counts}")

    detector.unload()

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] summary 저장 -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
