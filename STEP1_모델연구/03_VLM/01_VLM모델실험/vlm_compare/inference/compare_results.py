"""
results/{qwen3vl,internvl35,minicpm45}/<video_id>_<frame_idx>.json 를 모아
클립·프레임 기준으로 3개 모델 출력을 나란히 놓은 markdown 표를 생성한다.

사용:
  python3 compare_results.py [--results_dir ../results] [--output ../compare_results.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

MODELS = ["qwen3vl", "internvl35", "minicpm45"]
FIELDS = ["action", "tools_or_parts", "notable_technique", "visual_confidence", "uncertain_points"]


def load_model_results(results_dir: Path, model: str) -> Dict[tuple, Dict[str, Any]]:
    """{(video_id, frame_idx): 결과dict} 매핑. 폴더/파일이 없으면 빈 dict."""
    model_dir = results_dir / model
    out: Dict[tuple, Dict[str, Any]] = {}
    if not model_dir.is_dir():
        return out
    for path in model_dir.glob("*.json"):
        data = json.load(open(path, encoding="utf-8"))
        key = (data["video_id"], data["frame_idx"])
        out[key] = data
    return out


def format_cell(entry: Optional[Dict[str, Any]]) -> str:
    if entry is None:
        return "(결과 없음)"
    parsed = entry.get("parsed")
    quant = entry.get("quantization", "nf4 4bit")  # qwen3vl 결과는 이 필드 도입 전이라 기본값(nf4 4bit) 사용
    meta = (
        f"`{entry.get('inference_time_sec', '?')}s / {entry.get('gpu_memory_peak_gib', '?')}GiB / "
        f"{entry.get('attn_implementation', '?')} / {quant}`"
    )
    if parsed is None:
        raw_preview = (entry.get("model_response_raw") or "")[:200].replace("\n", " ")
        return f"⚠️ JSON 파싱 실패<br>원본: {raw_preview}...<br>{meta}"
    lines = []
    for field in FIELDS:
        val = parsed.get(field)
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val) if val else "-"
        lines.append(f"**{field}**: {val}")
    lines.append(meta)
    return "<br>".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="3개 VLM 결과를 클립·프레임 기준으로 비교하는 markdown 생성")
    script_dir = Path(__file__).resolve().parent
    ap.add_argument("--results_dir", default=str(script_dir.parent / "results"))
    ap.add_argument("--output", default=str(script_dir.parent / "compare_results.md"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    by_model = {model: load_model_results(results_dir, model) for model in MODELS}

    all_keys = sorted(set().union(*[set(d.keys()) for d in by_model.values()]))
    if not all_keys:
        print(f"[WARN] {results_dir} 안에 결과가 하나도 없습니다.")
        return

    lines = ["# VLM 비교 결과", "", f"총 {len(all_keys)}개 (클립, 프레임) 조합", ""]
    lines.append("| 클립 | frame_idx | " + " | ".join(MODELS) + " |")
    lines.append("|---|---|" + "---|" * len(MODELS))

    for video_id, frame_idx in all_keys:
        row = [video_id, str(frame_idx)]
        for model in MODELS:
            entry = by_model[model].get((video_id, frame_idx))
            row.append(format_cell(entry))
        lines.append("| " + " | ".join(row) + " |")

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {len(all_keys)}개 조합 비교표 저장 -> {output_path}")

    # 모델별 커버리지도 같이 출력(어느 모델이 몇 개나 빠졌는지 한눈에)
    for model in MODELS:
        n = len(by_model[model])
        missing = len(all_keys) - n
        print(f"  {model}: {n}개 결과" + (f" ({missing}개 누락)" if missing else ""))


if __name__ == "__main__":
    main()
