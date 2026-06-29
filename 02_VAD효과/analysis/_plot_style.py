"""matplotlib 공용 설정 — 헤드리스 백엔드 + 한글 폰트 등록."""
import os

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경 (디스플레이 없음)

from matplotlib import font_manager, rcParams

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


def set_korean_font() -> str | None:
    """
    사용 가능한 한글 폰트를 matplotlib에 등록하고 기본 family로 지정.
    .ttc/.otf는 자동 인식이 안 되므로 파일 경로로 직접 등록한다.
    반환: 적용된 폰트 이름 (없으면 None).
    """
    rcParams["axes.unicode_minus"] = False
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            rcParams["font.family"] = name
            return name
    return None
