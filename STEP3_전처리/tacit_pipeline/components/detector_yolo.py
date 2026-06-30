"""
YOLO 검출 어댑터 (스펙 5.2).

- 튜닝된 .pt 를 ultralytics 로 로딩, 프레임별 검출.
- 출력: 프레임별 {timestamp(초), class, conf, bbox}.
- bbox 포맷/정규화는 config로(기본: 픽셀 xywh).
- 모델 .names 를 constants.EXPECTED_NAMES 와 대조 검증.

⚠️ 오늘은 모델 로딩 금지 → 지연 import/지연 로딩.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..constants import validate_model_names
from ..interfaces.detector import FrameRef
from ..schema.intermediate import BBox, Detection, FrameDetections, FrameMeta


class UltralyticsYOLODetector:
    """ultralytics YOLO 어댑터. registry 키: 'yolo_ultralytics'."""

    def __init__(
        self,
        weights_path: str,  # TODO(fill): best.pt 경로 (내일)
        device: str = "cuda:0",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        bbox_format: str = "pixel_xywh",  # TODO(decision): "pixel_xywh" | "norm_xywh"
        strict_names: bool = False,  # True면 names 불일치 시 예외
        **extra: Any,
    ):
        self.weights_path = weights_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.bbox_format = bbox_format
        self.strict_names = strict_names
        self.extra = extra
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO  # noqa: 지연 import

        self._model = YOLO(self.weights_path)
        problems = validate_model_names(self.names)
        if problems:
            msg = "YOLO names 가 EXPECTED_NAMES 와 불일치:\n  " + "\n  ".join(problems)
            if self.strict_names:
                raise ValueError(msg)
            print("[WARN] " + msg)  # 경고만(후보 누락 방지 우선)

    @property
    def names(self) -> Dict[int, str]:
        self._load()
        return dict(self._model.names)

    def detect(self, frames: List[FrameRef], meta: FrameMeta) -> List[FrameDetections]:
        self._load()
        out: List[FrameDetections] = []
        for fr in frames:
            # 이미지가 없으면 ultralytics에 프레임을 직접 못 주므로 image 필요.
            # 샘플러/오케스트레이터가 image를 채워 넘긴다(없으면 경로+frame로 재디코드 TODO).
            source = fr.image if fr.image is not None else self._read_frame(meta, fr.frame_idx)
            results = self._model.predict(
                source, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
            dets: List[Detection] = []
            for r in results:
                names = r.names
                for box in r.boxes:
                    cls_idx = int(box.cls.item())
                    bbox = self._to_bbox(box, meta)
                    dets.append(
                        Detection(
                            timestamp=fr.timestamp,
                            frame_idx=fr.frame_idx,
                            cls=names.get(cls_idx, str(cls_idx)),
                            conf=float(box.conf.item()),
                            bbox=bbox,
                        )
                    )
            out.append(
                FrameDetections(timestamp=fr.timestamp, frame_idx=fr.frame_idx, detections=dets)
            )
        return out

    def _to_bbox(self, box, meta: FrameMeta) -> BBox:
        # ultralytics: box.xywh = [cx, cy, w, h] (픽셀). 좌상단 xywh로 변환.
        cx, cy, w, h = (float(v) for v in box.xywh[0].tolist())
        x, y = cx - w / 2, cy - h / 2
        if self.bbox_format == "norm_xywh" and meta.width and meta.height:
            return BBox(x=x / meta.width, y=y / meta.height,
                        w=w / meta.width, h=h / meta.height)
        return BBox(x=x, y=y, w=w, h=h)

    def _read_frame(self, meta: FrameMeta, frame_idx: int):
        """image가 없을 때 영상에서 단일 프레임 재디코드(opencv). 비효율적이므로 보통은 샘플러가 image를 채운다."""
        import cv2  # noqa

        cap = cv2.VideoCapture(meta.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"프레임 {frame_idx} 디코드 실패: {meta.path}")
        return img
