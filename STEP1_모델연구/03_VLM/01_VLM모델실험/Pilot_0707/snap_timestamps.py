"""파일럿 결과 3종의 timestamp를 실제 프레임 그리드(0,2,...,32s)로 스냅 교정.

본선 STEP4의 snap_to_grid와 동일한 원리: 모델이 지어낸 timestamp를 신뢰하지 않고
가장 가까운 실제 프레임 시각으로 붙인다. 파싱 불가 timestamp는 관찰 순서대로
그리드에 균등 배분(본선 _even_grid_ts와 동일). 원본은 보존, *_snapped.json으로 저장.
각 관찰에 timestamp_snapped_sec(구간 로컬)와 timestamp_abs_sec(원본 영상 기준 +80s)를 추가.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
GRID = [i * 2.0 for i in range(17)]  # 0,2,...,32
SEGMENT_START = 80.0
_TS = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


def to_sec(ts):
    if isinstance(ts, (int, float)):
        return float(ts)
    m = _TS.match(str(ts).strip())
    if not m:
        return None
    h, mi, s = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s


def snap(sec):
    return min(GRID, key=lambda g: abs(g - sec))


for model in ["qwen3vl", "internvl35", "minicpm45"]:
    p = BASE / "results" / model / "CLIP2_0704.mp4_80-113.observations.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    obs = d.get("observations") or []
    n = len(obs)
    for i, o in enumerate(obs):
        sec = to_sec(o.get("timestamp"))
        if sec is None:  # 파싱 실패 → 순서 기반 균등 배분
            pos = GRID[int(round(i * (len(GRID) - 1) / max(1, n - 1)))]
            o["timestamp_snap_method"] = "even_grid(순서 균등 배분)"
        else:
            pos = snap(sec)
            o["timestamp_snap_method"] = "nearest_grid"
        o["timestamp_snapped_sec"] = pos
        o["timestamp_abs_sec"] = SEGMENT_START + pos
    d["timestamp_correction"] = ("모델 생성 timestamp를 실제 프레임 그리드(0,2,...,32s)로 스냅. "
                                 "본선 STEP4 snap_to_grid와 동일 원리. 원본 timestamp 필드는 유지.")
    out = p.with_name(p.name.replace(".observations.json", ".observations_snapped.json"))
    out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    snapped = [(o.get("timestamp"), o["timestamp_snapped_sec"]) for o in obs]
    print(f"[{model}] {n}건: {snapped}")
print("[OK] *_snapped.json 저장 완료")
