"""
프롬프트 v2 출력 스키마 준수 여부 채점 (사후 검증, 재추론 없음, 읽기 전용).

필수 필드 5개(action, tools_or_parts, visual_confidence, notable_technique,
uncertain_points) 존재 여부 + 타입/값 검증 + 일관성 규칙(uncertain_points
비어있지 않은데 visual_confidence가 high면 위반) 을 프레임 단위로 채점한다.

출력:
  scoring/results/schema_compliance_summary.json  (모델별 요약)
  scoring/results/schema_compliance_detail.csv    (프레임 단위 상세)
"""
import json
import glob
import csv
from pathlib import Path

RESULTS_DIR = Path("/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/results")
OUT_DIR = Path("/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/scoring/results")

MODELS = [("qwen3vl", "Qwen3-VL-8B"), ("internvl35", "InternVL3.5-8B"), ("minicpm45", "MiniCPM-V 4.5")]
REQUIRED_FIELDS = ["action", "tools_or_parts", "visual_confidence", "notable_technique", "uncertain_points"]


def validate_schema(sample_vlm_output):
    """반환: 위반 사항 코드 리스트 (비어있으면 유효)."""
    issues = []
    if sample_vlm_output is None:
        return ["NO_OBJECT"]

    parsed = sample_vlm_output.get("parsed")
    if parsed is None:
        return ["PARSE_FAILED"]

    for field in REQUIRED_FIELDS:
        if field not in parsed:
            issues.append(f"MISSING_FIELD_{field.upper()}")

    action = parsed.get("action")
    if not isinstance(action, str) or not action.strip():
        issues.append("ACTION_INVALID")

    tools = parsed.get("tools_or_parts")
    if not isinstance(tools, list):
        issues.append("TOOLS_NOT_LIST")

    tech = parsed.get("notable_technique")
    if tech is not None and not isinstance(tech, str):
        issues.append("TECH_NOT_STRING_OR_NULL")
    if isinstance(tech, str) and len(tech) > 50:
        issues.append("TECH_TOO_LONG")

    conf = parsed.get("visual_confidence")
    if conf not in ("high", "medium", "low"):
        issues.append("CONF_INVALID_VALUE")

    uncertain = parsed.get("uncertain_points")
    if not isinstance(uncertain, list):
        issues.append("UNCERTAIN_NOT_LIST")
    elif len(uncertain) > 0 and conf == "high":
        issues.append("CONF_UNCERTAIN_MISMATCH")

    return issues


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    summary = []

    for model_key, model_label in MODELS:
        files = sorted(glob.glob(str(RESULTS_DIR / model_key / "*.json")))
        total = len(files)
        valid = 0
        code_counts = {}

        for fp in files:
            data = json.load(open(fp, encoding="utf-8"))
            issues = validate_schema(data)
            is_valid = len(issues) == 0
            if is_valid:
                valid += 1
            for code in issues:
                code_counts[code] = code_counts.get(code, 0) + 1
            detail_rows.append({
                "model": model_key,
                "video_id": data.get("video_id"),
                "frame_idx": data.get("frame_idx"),
                "pass": is_valid,
                "issues": ";".join(issues),
            })

        rate = valid / total * 100 if total else 0.0
        summary.append({
            "model": model_key,
            "model_label": model_label,
            "total_frames": total,
            "valid_frames": valid,
            "violated_frames": total - valid,
            "compliance_rate_pct": round(rate, 2),
            "violation_code_counts": code_counts,
        })

    (OUT_DIR / "schema_compliance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(OUT_DIR / "schema_compliance_detail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "video_id", "frame_idx", "pass", "issues"])
        writer.writeheader()
        writer.writerows(detail_rows)

    print("[OK] schema_compliance_summary.json / schema_compliance_detail.csv 저장 완료")
    for s in summary:
        print(f"  {s['model_label']}: {s['valid_frames']}/{s['total_frames']} 유효 ({s['compliance_rate_pct']}%)")


if __name__ == "__main__":
    main()
