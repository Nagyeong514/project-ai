"""
YOLO 클래스 상수 — **한 곳에서만 정의**(스펙 6번).

팀이 대소문자/표기를 바꿔도 여기 한 곳만 수정하면 된다. 코드 곳곳에 'hand' 같은
문자열을 흩지 마라. `Classes.HAND` 처럼 참조한다.

────────────────────────────────────────────────────────────────────────────
⚠️ 실제 학습 모델(best.pt)에서 추출한 names — 2026-06-30 확인:
    {0: 'GPU', 1: 'RAM', 2: 'RAM_slot', 3: 'eraser',
     4: 'hand', 5: 'monitor', 6: 'power_button_LED'}  → 총 7개

스펙 6번 표는 8개(여기에 'motherboard' 포함)였으나, motherboard는 **불필요하다고 판단해
의도적으로 학습에서 제외**(팀 확인, 2026-06-30). 즉 누락이 아니라 설계 결정이다.
인덱스 순서도 스펙과 다르다(알파벳순). 아래는 '실제 모델' 기준이 진실이다.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List


class Classes:
    """실제 best.pt 기준 클래스 이름 상수."""

    GPU = "GPU"
    RAM = "RAM"
    RAM_SLOT = "RAM_slot"
    ERASER = "eraser"
    HAND = "hand"
    MONITOR = "monitor"
    POWER_BUTTON_LED = "power_button_LED"

    # NOTE: 스펙엔 있었으나 '불필요'로 판단해 의도적으로 학습 제외(팀 확인).
    #   따라서 motherboard 상수/좌표기준점 로직은 두지 않는다. 기준점이 필요하면
    #   다른 부품(예: RAM_slot)으로 대체. (히스토리 메모로만 남김)


# best.pt 의 실제 idx→name 매핑(검증 기준). 모델의 .names 와 대조하는 데 쓴다.
EXPECTED_NAMES: Dict[int, str] = {
    0: Classes.GPU,
    1: Classes.RAM,
    2: Classes.RAM_SLOT,
    3: Classes.ERASER,
    4: Classes.HAND,
    5: Classes.MONITOR,
    6: Classes.POWER_BUTTON_LED,
}

# 파이프라인이 인지하는 클래스 전체(실제 모델 기준 7개).
ALL_CLASSES: List[str] = list(EXPECTED_NAMES.values())

# ── 의미 분류(샘플링/판단 로직에서 카테고리로 참조) ─────────────────────────
HAND_CLASSES: List[str] = [Classes.HAND]  # 행동 주체
PART_CLASSES: List[str] = [Classes.RAM, Classes.RAM_SLOT, Classes.GPU]  # 부품
TOOL_CLASSES: List[str] = [Classes.ERASER]  # 도구(핵심 암묵지 도구: 접점 세척)
DIAGNOSTIC_CLASSES: List[str] = [Classes.POWER_BUTTON_LED]  # 진단 단서(VLM 집중영역)
OUTPUT_CLASSES: List[str] = [Classes.MONITOR]  # 출력장치(화면 상태)

# motion-guided 샘플링의 '움직임 추적 대상'(ego-motion에 안 휘둘리게):
# 손/도구/부품 bbox의 움직임만 본다.
MOTION_TRACK_CLASSES: List[str] = HAND_CLASSES + TOOL_CLASSES + PART_CLASSES


def validate_model_names(model_names: Dict[int, str]) -> List[str]:
    """모델의 .names 를 우리 EXPECTED_NAMES 와 대조. 불일치 메시지 리스트 반환(빈 리스트=정상).

    스펙 5.2 '우리 상수와 대조하는 검증 함수'. detector 어댑터 로딩 직후 호출한다.
    """
    problems: List[str] = []
    for idx, name in EXPECTED_NAMES.items():
        got = model_names.get(idx)
        if got != name:
            problems.append(f"idx {idx}: expected '{name}', model has '{got}'")
    extra = set(model_names) - set(EXPECTED_NAMES)
    if extra:
        problems.append(f"모델에 예상 밖 인덱스 존재: {sorted(extra)} → {{i: model_names[i] for i in extra}}")
    return problems
