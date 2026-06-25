"""
손익분기점 분석: VAD 오버헤드가 정당화되는 무음 비율 임계점 도출.
A' ↔ B 총 처리시간 차이를 무음 비율의 함수로 모델링.

실질 RTF(B) = STT 처리 RTF(rtf_mean) + VAD 실행시간(vad_time_s) / 오디오 길이.
오디오 길이는 결과 CSV에 없으므로 metadata.csv(duration_min)에서 가져온다.
"""
from analysis._plot_style import set_korean_font  # noqa: F401 (Agg 백엔드 설정 포함)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_csv: str, metadata_csv: str | None = None) -> pd.DataFrame:
    """
    long 포맷 결과 CSV를 file_id 단위 wide 포맷으로 변환.
    반환 컬럼: file_id, silence_ratio, rtf_A_prime, rtf_B, vad_time_s,
              duration_s, rtf_B_real, rtf_gain
    rtf_gain = rtf_A_prime - rtf_B_real  (양수 = B가 빠름)
    """
    df = pd.read_csv(results_csv)

    rtf = df.pivot_table(index="file_id", columns="condition", values="rtf_mean")
    out = pd.DataFrame(index=rtf.index)
    out["rtf_A_prime"] = rtf.get("A_prime")
    out["rtf_B"] = rtf.get("B")

    # 파일별 부가 정보 (silence_ratio, vad_time_s)는 조건 B 행에서 취득
    silence = df.groupby("file_id")["silence_ratio"].first()
    vad_time = df[df["condition"] == "B"].set_index("file_id")["vad_time_s"]
    out["silence_ratio"] = silence
    out["vad_time_s"] = vad_time

    # 오디오 길이: metadata 우선, 없으면 VAD 오버헤드 0으로 처리(STT RTF만)
    if metadata_csv:
        meta = pd.read_csv(metadata_csv).set_index("file_id")
        out["duration_s"] = meta["duration_min"] * 60.0
    else:
        out["duration_s"] = np.nan

    vad_rtf = (out["vad_time_s"] / out["duration_s"]).fillna(0.0)
    out["rtf_B_real"] = out["rtf_B"] + vad_rtf
    out["rtf_gain"] = out["rtf_A_prime"] - out["rtf_B_real"]

    out = out.reset_index().dropna(subset=["silence_ratio", "rtf_gain"])
    return out


def compute_breakeven(df: pd.DataFrame) -> dict:
    """
    df: load_results() 출력 (silence_ratio, rtf_gain 필요).
    손익분기점: rtf_gain = 0 이 되는 silence_ratio (선형 회귀 외삽).
    """
    df = df.copy()
    x = df["silence_ratio"].values
    y = df["rtf_gain"].values

    breakeven = None
    a = b = None
    if len(df) >= 2 and np.ptp(x) > 0:
        a, b = np.polyfit(x, y, deg=1)  # y = a*x + b
        breakeven = -b / a if a != 0 else None

    return {
        "slope": float(a) if a is not None else None,
        "intercept": float(b) if b is not None else None,
        "breakeven_silence_ratio": float(breakeven) if breakeven is not None else None,
        "n_files": len(df),
        "mean_rtf_gain": float(df["rtf_gain"].mean()),
        "df": df,
    }


def plot_breakeven(result: dict, save_path: str | None = None) -> None:
    set_korean_font()
    df = result["df"]
    a, b = result["slope"], result["intercept"]
    breakeven = result["breakeven_silence_ratio"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="silence_ratio", y="rtf_gain", ax=ax, s=80)

    if a is not None:
        x_range = np.linspace(0, 1, 100)
        ax.plot(x_range, a * x_range + b, color="crimson", label="선형 회귀")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    if breakeven is not None and 0 <= breakeven <= 1:
        ax.axvline(breakeven, color="navy", linestyle=":", label=f"손익분기점 ≈ {breakeven:.2f}")

    ax.set_xlabel("무음 비율")
    ax.set_ylabel("RTF 이득 (A' − 실질 B, 양수 = B 빠름)")
    ax.set_title("VAD 손익분기점: 무음 비율 vs RTF 이득")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="VAD 손익분기점 분석")
    parser.add_argument("--results", default="results/raw/results.csv")
    parser.add_argument("--metadata", default="data/metadata.csv",
                        help="오디오 길이(duration_min) 출처 — 미지정 시 VAD 오버헤드 제외")
    parser.add_argument("--plot", default=None, help="손익분기 그래프 저장 경로")
    parser.add_argument("--output", default=None, help="요약 JSON 저장 경로")
    args = parser.parse_args()

    df = load_results(args.results, args.metadata)
    result = compute_breakeven(df)

    print(f"{'='*60}\nVAD 손익분기점 분석 (n={result['n_files']})")
    print(f"  평균 RTF 이득(A'−실질B): {result['mean_rtf_gain']:+.4f} "
          f"({'B가 평균적으로 빠름' if result['mean_rtf_gain'] > 0 else 'A’가 평균적으로 빠름'})")
    if result["breakeven_silence_ratio"] is not None:
        be = result["breakeven_silence_ratio"]
        print(f"  회귀선: y = {result['slope']:.4f}·x + {result['intercept']:.4f}")
        print(f"  손익분기 무음비율 ≈ {be:.3f}", end="")
        print(" (범위 밖 — 관측 구간에서 B 우위 전환 없음)" if not (0 <= be <= 1) else "")
    else:
        print("  회귀 불가 (표본 부족 또는 무음비율 분산 없음)")

    print("\n  파일별 RTF 이득:")
    for _, r in result["df"].sort_values("silence_ratio").iterrows():
        print(f"    {r['file_id']:>4}  무음 {r['silence_ratio']:.3f}  "
              f"A'={r['rtf_A_prime']:.4f}  실질B={r['rtf_B_real']:.4f}  "
              f"이득={r['rtf_gain']:+.4f}")

    if args.plot:
        plot_breakeven(result, args.plot)
    if args.output:
        summary = {k: v for k, v in result.items() if k != "df"}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n요약 저장: {args.output}")


if __name__ == "__main__":
    main()
