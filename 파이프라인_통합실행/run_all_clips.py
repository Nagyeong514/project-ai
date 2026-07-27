"""
CLIP1~4 전체를 한 프로세스에서 순차 실행 (모델 재로딩 방지).

STEP3/4/5를 폴더로는 분리했지만, STT/YOLO/VLM/LLM 컴포넌트는 각 Step{N}Runner 생성자
안에서 한 번만 빌드하고(3개 Runner 자체도 여기서 한 번만 만듦) 4클립 루프 안에서는
.run()만 반복 호출한다 — 별도 프로세스로 안 쪼개서 재로딩을 피한다는 원래
run_all_clips.py의 이점을 그대로 유지. 대신 단계 사이 데이터는 항상 파일을 거친다
(STEP 단독 실행과 동일 코드 경로 — 두 실행 방식이 갈라지지 않게).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEP_DIRS = ["STEP3_전처리", "STEP4_YOLO_VLM관찰", "STEP5_LLM_암묵지_후보생성"]
for _p in STEP_DIRS:
    sys.path.insert(0, str(ROOT / _p))
sys.path.insert(0, str(ROOT))  # tacit_common

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

    # 각 STEP 폴더가 sys.path에 있어야 import 가능(위 부트스트랩 참고).
    from step3_runner import Step3Runner
    from step4_runner import Step4Runner
    from step5_runner import Step5Runner

    step3 = Step3Runner(cfg)
    step4 = Step4Runner(cfg, config_path="config.yaml")
    step5 = Step5Runner(cfg)

    results = {}
    for video in VIDEOS:
        print(f"\n{'=' * 80}\n[RUN] {video}\n{'=' * 80}")
        try:
            video_id = step3.run(video)
            step4.run(video_id)
            doc = step5.run(video_id)
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
