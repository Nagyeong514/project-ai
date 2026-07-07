"""
motion_score.py
AR Glass 영상 → 1초 간격 샘플링 → absdiff 기반 Motion Score → CSV 저장
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_motion_score(
    video_path,
    output_csv="motion_scores.csv",
    sample_interval=1.0,
    blur_ksize=5,
    noise_threshold=1.5,
):
    """
    1초 간격 샘플링으로 인접 프레임 간 absdiff 평균을 Motion Score로 계산.
    GaussianBlur로 카메라 노이즈 사전 억제.

    Returns: pd.DataFrame with columns [time_sec, frame_idx, motion_score]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    step         = max(1, int(fps * sample_interval))
    n_samples    = total_frames // step

    print(f"[motion_score] FPS={fps:.1f} | 총 프레임={total_frames} | "
          f"길이={duration:.0f}s ({duration/60:.1f}분)")
    print(f"[motion_score] 샘플링 간격={sample_interval}s → 처리 프레임={n_samples}개")

    records   = []
    prev_gray = None

    for i in tqdm(range(n_samples), desc="Motion Score 계산", unit="frame"):
        frame_pos = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        if prev_gray is not None:
            diff  = cv2.absdiff(gray, prev_gray)
            score = float(np.mean(diff))
        else:
            score = 0.0

        records.append({
            "time_sec"    : round(frame_pos / fps, 2),
            "frame_idx"   : int(frame_pos),
            "motion_score": round(score, 4),
        })
        prev_gray = gray

    cap.release()

    df = pd.DataFrame(records)

    # 노이즈 데드존 처리
    df.loc[df["motion_score"] < noise_threshold, "motion_score"] = 0.0
    nonzero_pct = (df["motion_score"] > 0).mean() * 100

    df.to_csv(output_csv, index=False)

    print(f"\n[motion_score] CSV 저장 완료 → {output_csv}")
    print(f"[motion_score] 유효 모션 비율: {nonzero_pct:.1f}%")
    print(df["motion_score"].describe().round(4).to_string())
    return df


if __name__ == "__main__":
    VIDEO = "다시 돌아온 1인칭 컴퓨터 조립으로 다 알려드립니다..mp4"
    calculate_motion_score(VIDEO, "motion_scores.csv", sample_interval=1.0)
