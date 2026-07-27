"""모션가이드 샘플링 — 이식 원본 및 출처.

원본 위치: 00_사전연구/04_모션가이드샘플링_프로토타입/01_코드/ (팀원 제공, 원 작성자 실명은
계획서 승인 후 실행 시점에 확정해 채운다)
이식일: 2026-07-08
계획서: docs/계획서_모션가이드_STEP3_이식.md

로직 무변경 이식 — abc_pipeline_runner.py의 import 3줄(상대경로 전환)만 바뀌었고,
motion_score.py / select_clip.py / extract_video.py는 원본과 바이트 단위로 동일하다.
사전연구 원본(00_사전연구/04_.../01_코드/)은 삭제·이동하지 않고 그대로 보존한다.
"""

from .abc_pipeline_runner import run_abc_pipeline

__all__ = ["run_abc_pipeline"]
