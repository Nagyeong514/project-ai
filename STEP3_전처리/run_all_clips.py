"""
CLIP1~4 전체를 한 프로세스에서 순차 실행 (모델 재로딩 방지).
run.py를 4번 따로 부르면 STT/YOLO/VLM/LLM을 매번 새로 로드해야 해서 비효율적 —
Pipeline 객체 하나를 만들고 run(video)만 4번 호출한다.
"""

from __future__ import annotations

import sys
import traceback

from tacit_pipeline import Pipeline
from preflight import run_preflight

VIDEO_DIR = "/home/ai_user/team_a2/members/안나경/master/master_videos"
VIDEOS = [
    f"{VIDEO_DIR}/CLIP1_정상조립과정.mp4",
    f"{VIDEO_DIR}/CLIP2.mp4",
    f"{VIDEO_DIR}/CLIP3_재부팅시도 전까지.mp4",
    f"{VIDEO_DIR}/CLIP4_재부팅및BIOS.mp4",
]


def main() -> None:
    # VIDEOS 4개 전부 있는지 먼저 확인 — 3번째 클립 로딩 중간에 파일 없음을 발견해
    # 이미 로드한 STT/YOLO/VLM/LLM을 헛수고로 만드는 사고를 방지.
    cfg = run_preflight("config.yaml", VIDEOS)
    pipe = Pipeline(cfg)

    results = {}
    for video in VIDEOS:
        print(f"\n{'=' * 80}\n[RUN] {video}\n{'=' * 80}")
        try:
            doc = pipe.run(video)
            results[video] = f"OK ({len(doc.candidates)}건)"
        except Exception as e:
            print(f"[FAIL] {video}: {e}")
            traceback.print_exc()
            results[video] = f"FAIL: {e}"

    print(f"\n{'=' * 80}\n[SUMMARY]\n{'=' * 80}")
    for video, status in results.items():
        print(f"  {video}: {status}")

    if any(s.startswith("FAIL") for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
