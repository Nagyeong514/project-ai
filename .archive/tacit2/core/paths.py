"""산출물 경로 계약(단일 진실 공급원).

각 스테이지는 앞 스테이지의 폴더만 읽고 자기 폴더에만 쓴다.
skip-if-exists 판정도 전부 이 경로 기준.

output/
├── transcripts/<clip>.json          [S1 STT]
├── frames/<clip>/frame_0001.jpg...  [S2 프레임]  ※ 클립별 폴더 — 덮어쓰기 버그 원천 차단
├── frames/<clip>/times.json         [S2]
├── detections/<clip>.json           [S3 YOLO]
├── vlm_observations/<clip>.json     [S4 VLM]
└── tacit_json/<clip>.tacit.json     [S5 융합]
"""
from __future__ import annotations

from pathlib import Path


def clip_id(video_path: str) -> str:
    return Path(video_path).name  # 확장자 포함 원본 파일명


class Contract:
    def __init__(self, output_dir: str):
        self.root = Path(output_dir)

    # --- S1 ---
    def transcript(self, cid: str) -> Path:
        return self.root / "transcripts" / f"{cid}.json"

    # --- S2 ---
    def frames_dir(self, cid: str) -> Path:
        return self.root / "frames" / cid

    def times(self, cid: str) -> Path:
        return self.frames_dir(cid) / "times.json"

    # --- S3 ---
    def detections(self, cid: str) -> Path:
        return self.root / "detections" / f"{cid}.json"

    # --- S4 ---
    def observations(self, cid: str) -> Path:
        return self.root / "vlm_observations" / f"{cid}.json"

    # --- S5 ---
    def tacit(self, cid: str) -> Path:
        return self.root / "tacit_json" / f"{cid}.tacit.json"

    def ensure_dirs(self):
        for d in ["transcripts", "frames", "detections", "vlm_observations", "tacit_json"]:
            (self.root / d).mkdir(parents=True, exist_ok=True)
