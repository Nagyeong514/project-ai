"""
손익분기점 분석: VAD 오버헤드가 정당화되는 무음 비율 임계점 도출.
A' ↔ B 총 처리시간 차이를 무음 비율의 함수로 모델링.
"""
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_breakeven(df: pd.DataFrame) -> dict:
    """
    df 컬럼: file_id, silence_ratio, rtf_A_prime, rtf_B, vad_time_s, audio_duration_s
    손익분기점: rtf_B < rtf_A_prime 인 최소 silence_ratio 추정.
    """
    df = df.copy()
    df["rtf_gain"] = df["rtf_A_prime"] - df["rtf_B"]   # 양수 = B가 빠름

    # 선형 회귀로 손익분기점 추정
    x = df["silence_ratio"].values
    y = df["rtf_gain"].values
    coeffs = np.polyfit(x, y, deg=1)   # y = a*x + b
    a, b = coeffs
    breakeven = -b / a if a != 0 else None

    return {
        "slope": float(a),
        "intercept": float(b),
        "breakeven_silence_ratio": float(breakeven) if breakeven is not None else None,
        "n_files": len(df),
        "df": df,
    }


def plot_breakeven(result: dict, save_path: str | None = None) -> None:
    df = result["df"]
    a, b = result["slope"], result["intercept"]
    breakeven = result["breakeven_silence_ratio"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="silence_ratio", y="rtf_gain", ax=ax, s=80)

    x_range = np.linspace(0, 1, 100)
    ax.plot(x_range, a * x_range + b, color="crimson", label="선형 회귀")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    if breakeven is not None and 0 <= breakeven <= 1:
        ax.axvline(breakeven, color="navy", linestyle=":", label=f"손익분기점 ≈ {breakeven:.2f}")

    ax.set_xlabel("무음 비율")
    ax.set_ylabel("RTF 이득 (A' − B, 양수 = B 빠름)")
    ax.set_title("VAD 손익분기점: 무음 비율 vs RTF 이득")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    else:
        plt.show()
    plt.close()
