"""
추론 지연시간(latency) 채점 (사후 검증, 재추론 없음, 읽기 전용).

latency 값은 VLM 출력 JSON에 이미 있는 inference_time_sec 필드를 그대로 사용한다
(SLURM 로그 등 별도 소스 불필요 — 확인 완료).

출력:
  scoring/results/latency_summary.json         (모델별, 모델x클립별 통계)
  scoring/results/latency_detail.csv           (프레임 단위 상세)
"""
import json
import glob
import csv
import statistics
from pathlib import Path

RESULTS_DIR = Path("/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/results")
OUT_DIR = Path("/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/scoring/results")

MODELS = [("qwen3vl", "Qwen3-VL-8B"), ("internvl35", "InternVL3.5-8B"), ("minicpm45", "MiniCPM-V 4.5")]


def percentile(values, p):
    """선형보간 없는 최근접 순위 방식(간단한 p95)."""
    s = sorted(values)
    if not s:
        return None
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": round(percentile(values, 95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    per_model = {}
    per_model_clip = {}

    for model_key, model_label in MODELS:
        files = sorted(glob.glob(str(RESULTS_DIR / model_key / "*.json")))
        model_times = []
        clip_times = {}

        for fp in files:
            data = json.load(open(fp, encoding="utf-8"))
            t = data.get("inference_time_sec")
            clip = data.get("video_id")
            detail_rows.append({
                "model": model_key, "video_id": clip,
                "frame_idx": data.get("frame_idx"), "inference_time_sec": t,
            })
            if t is not None:
                model_times.append(t)
                clip_times.setdefault(clip, []).append(t)

        per_model[model_key] = {"model_label": model_label, **stats(model_times)}
        per_model_clip[model_key] = {
            clip: stats(times) for clip, times in clip_times.items()
        }

    summary = {"per_model": per_model, "per_model_per_clip": per_model_clip}
    (OUT_DIR / "latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(OUT_DIR / "latency_detail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "video_id", "frame_idx", "inference_time_sec"])
        writer.writeheader()
        writer.writerows(detail_rows)

    print("[OK] latency_summary.json / latency_detail.csv 저장 완료")
    for model_key, s in per_model.items():
        print(f"  {s['model_label']}: mean={s['mean']}s median={s['median']}s p95={s['p95']}s "
              f"min={s['min']}s max={s['max']}s (n={s['n']})")


if __name__ == "__main__":
    main()
