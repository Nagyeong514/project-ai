"""
run_all_clips.py
03_clip 폴더의 클립 1~4를 자동 샘플러(A/B/C 파이프라인)로 일괄 처리한다.

- CSV / 요약 / 타임스탬프 파일 → 02_파일/<영상명>/
- 분리(샘플링)된 영상          → 04_샘플링 영상/<영상명>/

실행: python 01_코드/run_all_clips.py   (폴더 루트 기준)
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from abc_pipeline_runner import build_result_table, run_abc_pipeline

CLIP_DIR  = os.path.join(BASE_DIR, "03_clip")
FILES_DIR = os.path.join(BASE_DIR, "02_파일")
OUT_DIR   = os.path.join(BASE_DIR, "04_샘플링 영상")


def main():
    videos = sorted(
        f for f in os.listdir(CLIP_DIR) if f.lower().endswith(".mp4")
    )
    if not videos:
        raise FileNotFoundError(f"03_clip 폴더에 mp4 영상이 없습니다: {CLIP_DIR}")

    print(f"[run_all_clips] 처리할 영상 {len(videos)}개: {videos}\n")

    stats_list = []
    for video in videos:
        name = os.path.splitext(video)[0]
        print("\n" + "#" * 70)
        print(f"# {name}")
        print("#" * 70)
        stats = run_abc_pipeline(
            os.path.join(CLIP_DIR, video),
            files_dir=os.path.join(FILES_DIR, name),
            clips_dir=os.path.join(OUT_DIR, name),
        )
        stats_list.append(stats)

    table = build_result_table(stats_list)
    table_path = os.path.join(FILES_DIR, "샘플링_결과표.txt")
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(table)

    print("\n" + table)
    print("\n[run_all_clips] 전체 완료")
    print(f"  파일(CSV/요약/타임스탬프): {FILES_DIR}")
    print(f"  샘플링 영상: {OUT_DIR}")
    print(f"  결과 요약표: {table_path}")


if __name__ == "__main__":
    main()
