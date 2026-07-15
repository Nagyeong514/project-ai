"""
tools_or_parts / action / notable_technique 세 필드를 스캔해, YOLO가 그 프레임에서
탐지하지 못한 클래스가 헷지/부정 없이 언급된 경우를 "환각 후보"로 집계한다.
(사후 검증, 재추론 없음, 읽기 전용)

이 지표는 YOLO의 미검출(false negative)과 VLM의 실제 환각을 구분하지 못하며,
YOLO 검출 결과를 정답이 아닌 약한 기준(weak reference)으로 사용함.

동의어 사전은 2026-07-05 사용자 승인 버전:
  - "메모리"/"모듈" 단독 표기는 RAM으로 자동매핑하지 않고 unresolved_ambiguous로 분류
  - monitor의 "화면"은 제외(디스플레이 내용/상태를 가리키는 용법이 대부분이라 실측 확인함)
  - YOLO 7개 클래스에 대응어가 없는 언급(키보드/마우스/케이스류/컵/메인보드/케이블류/
    테이블 등)은 환각 판정 대상에서 제외하고 out_of_vocabulary_mentions.csv로 별도 기록
  - 깨진 표현(스crewdriver, 전원ユニット, 중국어 혼입 등)은 사전에 안 걸려 자동으로
    환각 계산에서 빠지며, possible_encoding_corruption.csv로 별도 기록(VLM 품질 문제로 취급)
"""
import json
import glob
import csv
import re
from pathlib import Path

RESULTS_DIR = Path("/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/results")
DET_DIR = Path("/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/260704_VLM/Detection/results")
OUT_DIR = Path("/home/ai_user/team_a2/members/오유빈/오유빈_VLM모델실험/vlm_compare/scoring/results")

MODELS = [("qwen3vl", "Qwen3-VL-8B"), ("internvl35", "InternVL3.5-8B"), ("minicpm45", "MiniCPM-V 4.5")]

# YOLO 클래스 -> 승인된 동의어 목록 (긴 문자열부터 매칭되도록 각 리스트 내 순서도 길이순 유지)
SYNONYMS = {
    "GPU": ["NVIDIA GeForce RTX 그래픽카드", "GPU(GeForce RTX)", "그래픽카드", "그래픽 카드", "GPU", "gpu"],
    "RAM": ["메모리 모듈", "RAM 모듈", "RAM"],
    "RAM_slot": ["RAM 슬롯", "RAM_slot", "RAM Slot", "램슬롯"],
    "hand": ["손", "hand"],
    "monitor": ["컴퓨터 모니터", "모니터", "monitor"],
    "eraser": ["지우개", "eraser"],
    "power_button_LED": ["전원 버튼 LED", "전원 버튼", "파워 버튼", "전원 LED"],
}
# 전체를 (문자열, 클래스) 쌍으로 펴고 길이 내림차순 정렬 -> greedy longest-match-first
_ALL_SYNONYMS = sorted(
    ((s, cls) for cls, slist in SYNONYMS.items() for s in slist),
    key=lambda x: -len(x[0]),
)

# 표준보류(unresolved_ambiguous) 단독 표기 — 이미 매칭된 구간(예: "메모리 모듈")은 제외하고 검사
UNRESOLVED_STANDALONE = ["메모리", "모듈"]

HEDGE_PATTERNS = ["일 수도", "인 것 같", "로 보임", "로 추정", "불확실", "확인 불가", "판단 불가", "명확하지 않"]
NEGATION_PATTERNS = ["없음", "없다", "아님", "아니다", "보이지 않"]
CONTEXT_WINDOW = 20  # 매칭된 표현 앞뒤로 이 글자 수 안에 헷지/부정이 있으면 boundary_case 처리

# 인코딩 손상 휴리스틱: 한국어 프로젝트인데 중국어 한자/일본어 가나/치환문자가 섞여 있거나
# 한글-라틴 문자가 공백 없이 붙어있는 경우. 완전 검출은 불가능하므로 참고용 신호로만 사용.
_CJK_NON_KOREAN = re.compile(r"[぀-ヿ一-鿿�]")  # 히라가나/가타카나/한자/치환문자
# 한글 뒤에 라틴 문자(하이픈 허용)가 바로 붙는 경우만 검사한다 — "RAM을"/"GPU를"처럼
# 라틴 약어에 한국어 조사가 붙는 표준 표기(라틴+한글 방향)는 정상이라 오탐이 심해서 제외했다
# (실측 확인, 2026-07-05: CLIP1만 60건 오탐).
_KOREAN_LATIN_GLUED = re.compile(r"[가-힣]-?[a-zA-Z]{2,}")


def is_hedged_or_negated(text, start, end):
    window = text[max(0, start - CONTEXT_WINDOW): end + CONTEXT_WINDOW]
    for pat in HEDGE_PATTERNS + NEGATION_PATTERNS:
        if pat in window:
            return True, pat
    return False, None


def find_class_mentions(text):
    """text 안에서 (start,end,class,matched_str) 목록을 greedy longest-match로 찾고,
    소비된 구간을 반환한다."""
    mentions = []
    consumed = []  # [(start,end), ...]

    def overlaps(s, e):
        return any(not (e <= cs or s >= ce) for cs, ce in consumed)

    for syn, cls in _ALL_SYNONYMS:
        start = 0
        while True:
            idx = text.find(syn, start)
            if idx == -1:
                break
            end = idx + len(syn)
            if not overlaps(idx, end):
                mentions.append((idx, end, cls, syn))
                consumed.append((idx, end))
            start = idx + 1
    return mentions, consumed


def find_unresolved(text, consumed):
    """이미 소비된 구간을 제외하고, '메모리'/'모듈' 단독 표기를 찾는다."""
    found = []
    for term in UNRESOLVED_STANDALONE:
        start = 0
        while True:
            idx = text.find(term, start)
            if idx == -1:
                break
            end = idx + len(term)
            if not any(not (end <= cs or idx >= ce) for cs, ce in consumed):
                found.append(term)
            start = idx + 1
    return found


def has_encoding_corruption(text):
    if not text:
        return False
    return bool(_CJK_NON_KOREAN.search(text)) or bool(_KOREAN_LATIN_GLUED.search(text))


def process_frame(model_key, data, detected_classes):
    """한 프레임(한 모델)의 parsed를 스캔해 각 카테고리 리스트를 반환."""
    parsed = data.get("parsed")
    video_id, frame_idx = data["video_id"], data["frame_idx"]
    hallucinations, boundary_cases, unresolved, corruption = [], [], [], []
    oov = []

    if parsed is None:
        return hallucinations, boundary_cases, unresolved, corruption, oov

    action = parsed.get("action") if isinstance(parsed.get("action"), str) else ""
    tech = parsed.get("notable_technique") if isinstance(parsed.get("notable_technique"), str) else ""
    tools = parsed.get("tools_or_parts") if isinstance(parsed.get("tools_or_parts"), list) else []

    text_sources = [("action", action), ("notable_technique", tech)]
    for t in tools:
        if isinstance(t, str):
            text_sources.append(("tools_or_parts", t))

    for field, text in text_sources:
        if not text:
            continue
        if has_encoding_corruption(text):
            corruption.append({
                "model": model_key, "video_id": video_id, "frame_idx": frame_idx,
                "field": field, "raw_text": text,
            })

        mentions, consumed = find_class_mentions(text)
        for start, end, cls, syn in mentions:
            hedged, pat = is_hedged_or_negated(text, start, end)
            row = {
                "model": model_key, "video_id": video_id, "frame_idx": frame_idx,
                "field": field, "matched_class": cls, "matched_term": syn,
                "context": text, "detected_in_frame": cls in detected_classes,
            }
            if cls in detected_classes:
                continue  # 실제로 탐지된 클래스 -> 환각도 boundary_case도 아님 (집계 대상 아님)
            if hedged:
                row["hedge_or_negation"] = pat
                boundary_cases.append(row)
            else:
                hallucinations.append(row)

        for term in find_unresolved(text, consumed):
            unresolved.append({
                "model": model_key, "video_id": video_id, "frame_idx": frame_idx,
                "field": field, "term": term, "full_text": text,
            })

        # out-of-vocabulary: tools_or_parts 항목 중 위 사전에 전혀 안 걸리고,
        # unresolved(메모리/모듈 단독)로도 안 걸린 경우만 로그
        if field == "tools_or_parts" and not mentions and not find_unresolved(text, []):
            oov.append({
                "model": model_key, "video_id": video_id, "frame_idx": frame_idx,
                "term": text, "action_context": action,
            })

    return hallucinations, boundary_cases, unresolved, corruption, oov


def run(model_keys, clip_filter=None):
    all_halluc, all_boundary, all_unresolved, all_corruption, all_oov = [], [], [], [], []
    frame_level = []  # (model, video_id, frame_idx, halluc_count, boundary_count)
    det_cache = {}

    for model_key, model_label in model_keys:
        files = sorted(glob.glob(str(RESULTS_DIR / model_key / "*.json")))
        for fp in files:
            data = json.load(open(fp, encoding="utf-8"))
            video_id = data["video_id"]
            if clip_filter and video_id != clip_filter:
                continue
            if video_id not in det_cache:
                det_cache[video_id] = json.load(open(DET_DIR / f"{video_id}.detections.json", encoding="utf-8"))
            det_data = det_cache[video_id]
            frame_det = next((f for f in det_data["frames"] if f["frame_idx"] == data["frame_idx"]), None)
            detected_classes = {d["cls"] for d in (frame_det["detections"] if frame_det else [])}

            h, b, u, c, o = process_frame(model_key, data, detected_classes)
            all_halluc.extend(h)
            all_boundary.extend(b)
            all_unresolved.extend(u)
            all_corruption.extend(c)
            all_oov.extend(o)
            frame_level.append({
                "model": model_key, "video_id": video_id, "frame_idx": data["frame_idx"],
                "hallucination_count": len(h), "boundary_case_count": len(b),
                "has_hallucination": len(h) > 0,
            })

    return all_halluc, all_boundary, all_unresolved, all_corruption, all_oov, frame_level


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(model_keys, frame_level, all_halluc):
    total_by_model = {}
    for model_key, model_label in model_keys:
        frames = [r for r in frame_level if r["model"] == model_key]
        total = len(frames)
        frames_with_halluc = sum(1 for r in frames if r["has_hallucination"])
        total_mentions = sum(r["hallucination_count"] for r in frames)
        total_by_model[model_key] = {
            "model_label": model_label,
            "total_frames": total,
            "frame_level_hallucination_rate_pct": round(frames_with_halluc / total * 100, 2) if total else None,
            "mention_level_hallucination_rate_per_frame": round(total_mentions / total, 3) if total else None,
            "total_hallucination_mentions": total_mentions,
            "frames_with_hallucination": frames_with_halluc,
        }
    return total_by_model


def main(clip_filter=None, out_suffix=""):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_halluc, all_boundary, all_unresolved, all_corruption, all_oov, frame_level = run(MODELS, clip_filter)
    summary = summarize(MODELS, frame_level, all_halluc)

    result = {
        "caveat": (
            "이 지표는 YOLO의 미검출(false negative)과 VLM의 실제 환각을 구분하지 못하며, "
            "YOLO 검출 결과를 정답이 아닌 약한 기준(weak reference)으로 사용함"
        ),
        "per_model": summary,
        "out_of_vocabulary_mention_counts": {
            model_key: sum(1 for r in all_oov if r["model"] == model_key) for model_key, _ in MODELS
        },
    }

    suf = f"_{out_suffix}" if out_suffix else ""
    (OUT_DIR / f"hallucination_summary{suf}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(OUT_DIR / f"hallucination_detail{suf}.csv", frame_level,
              ["model", "video_id", "frame_idx", "hallucination_count", "boundary_case_count", "has_hallucination"])
    write_csv(OUT_DIR / f"hallucinations{suf}.csv", all_halluc,
              ["model", "video_id", "frame_idx", "field", "matched_class", "matched_term", "context", "detected_in_frame"])
    write_csv(OUT_DIR / f"boundary_cases{suf}.csv", all_boundary,
              ["model", "video_id", "frame_idx", "field", "matched_class", "matched_term", "context", "detected_in_frame", "hedge_or_negation"])
    write_csv(OUT_DIR / f"unresolved_mentions{suf}.csv", all_unresolved,
              ["model", "video_id", "frame_idx", "field", "term", "full_text"])
    write_csv(OUT_DIR / f"out_of_vocabulary_mentions{suf}.csv", all_oov,
              ["model", "video_id", "frame_idx", "term", "action_context"])
    write_csv(OUT_DIR / f"possible_encoding_corruption{suf}.csv", all_corruption,
              ["model", "video_id", "frame_idx", "field", "raw_text"])

    print(f"[OK] hallucination_rate 결과 저장 완료 (suffix={out_suffix or '(none)'})")
    for model_key, s in summary.items():
        print(f"  {s['model_label']}: 프레임단위 {s['frame_level_hallucination_rate_pct']}% "
              f"({s['frames_with_hallucination']}/{s['total_frames']}), "
              f"언급단위 {s['mention_level_hallucination_rate_per_frame']}건/프레임")
    print(f"  전체 boundary_case: {len(all_boundary)}건, unresolved: {len(all_unresolved)}건, "
          f"out_of_vocabulary: {len(all_oov)}건, encoding_corruption 의심: {len(all_corruption)}건")


if __name__ == "__main__":
    import sys
    clip = sys.argv[1] if len(sys.argv) > 1 else None
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    main(clip_filter=clip, out_suffix=suffix)
