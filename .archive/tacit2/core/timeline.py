"""시간축 유틸. 타임스탬프가 뼈대다 — 전 스테이지 공통 초(float) 단위."""
from __future__ import annotations

from typing import List, Sequence


def sec_to_hhmmss(sec: float) -> str:
    s = max(0, int(round(sec)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def hhmmss_to_sec(ts: str) -> float:
    parts = [float(p) for p in ts.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_ts(value) -> float:
    """모델이 뱉는 timestamp는 형식이 제멋대로다 — 숫자든 'HH:MM:SS'든 '12초'든 초로."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if ":" in s:
        try:
            return hhmmss_to_sec(s)
        except Exception:
            pass
    import re

    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else 0.0


def snap_to_grid(ts: float, grid: Sequence[float]) -> float:
    """모델 추정 시각을 실제 추출 프레임 시각 중 가장 가까운 값으로 강제."""
    if not grid:
        return ts
    return min(grid, key=lambda g: abs(g - ts))


def even_spread(n: int, lo: float, hi: float) -> List[float]:
    """timestamp가 아예 없는 관찰 n건을 [lo, hi] 구간에 균등 배치(0초 몰림 방지)."""
    if n <= 0:
        return []
    if n == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]
