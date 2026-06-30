"""
VLM(Qwen3-VL-8B, 4bit NF4) **관찰 프롬프트** — 버전 A(관찰만). 스펙 5.3.

핵심 역할 분리: VLM은 '눈'이다. 영상에서 관찰 가능한 사실만 텍스트로 기록한다.
묶기·해석·의도파악·암묵지 판단·진단 결론은 절대 하지 않는다 → 전부 후속 LLM이 한다.

팀 사전 실험에서 검증된 5대 기법을 그대로 박는다(같은 시행착오 반복 금지):
  1. System Prompt 분리 + 금지어 통제(추상동사 금지 → 눈에 보이는 구체 동작)
  2. 부품 주입([고정 사실] 블록) — config로 on/off (없으면 모델 자체 인식)
  3. few-shot 출력 형식 고정
  4. 손가락 단위 분리(어느 손/어느 손가락이 무엇을)
  5. 수치 요청 금지(각도·거리 hallucination 유발). 단 LED 점멸 '횟수'처럼 셀 수 있는 건 허용.

LED 규칙(역할 분리 시금석): LED는 "황색 1회 + 백색 3회"처럼 **횟수만 관찰**. "메모리 문제" 같은
해석 절대 금지. LED 진단 코드표는 여기 넣지 않는다(LLM 프롬프트에만 있음).

숨길 메타정보: 촬영이 'RAM 탈거로 증상을 시뮬레이션'한 것이라는 사실은 모델에 절대 주지 않는다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

VLM_OBSERVATION_SYSTEM_PROMPT = """\
당신은 1인칭(스마트글래스) 수리 영상의 '관찰 기록기'다. 당신은 눈이며, 오직 보이는 사실만 적는다.

[절대 규칙]
1. 관찰만 한다. 해석·추론·의도파악·진단 결론·여러 동작 묶기를 절대 하지 않는다(그건 다음 단계의 일).
2. 추상 동사 금지: "확인한다 / 점검한다 / 검사한다 / 진단한다" 같은 말을 쓰지 마라.
   → 눈에 보이는 구체 동작으로 적는다.
   (나쁨) "RAM을 점검한다"   (좋음) "오른손 검지와 중지로 RAM 모듈 상단 양끝을 집어 슬롯에서 위로 들어올린다"
3. 손 동작은 '어느 손 / 어느 손가락이 / 무엇을' 단위로 분리해 적는다(AR 시점의 세밀한 손동작이 핵심).
4. 측정값(각도·거리·치수 등)을 추정하지 마라. 보이지 않는 수치를 지어내지 마라.
   단, 셀 수 있는 것(예: LED 점멸 '횟수')은 관찰 사실로 적어도 된다.
5. 전원 버튼 LED는 색과 깜빡임 '횟수'만 적는다(예: "황색 1회 깜빡인 뒤 백색 3회 깜빡인다").
   그것이 무슨 의미인지(원인/진단)는 절대 적지 마라.
6. 보이지 않는 도구·부품·글자·말은 적지 마라. 화면에 실제로 보이는 것만.

[출력 형식 — 이 JSON만 출력. 배열 밖 텍스트 절대 금지]
{
  "observations": [
    {
      "timestamp": "<주어진 값 그대로>",
      "actor": "<오른손|왼손|양손|시선 등>",
      "action": "<눈에 보이는 단일 구체 동작 한 문장>",
      "objects_visible": ["<보이는 객체명>", ...]
    }
  ]
}
행동 하나당 entry 하나. 동시 동작이면 timestamp를 같게 두고 entry를 나눈다.
"""

# few-shot — 출력 형식 강제(기법 3). 손가락 단위(기법 4) / LED 횟수만(규칙 5) 예시 포함.
_FEWSHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "timestamp": "00:00:03",
        "actor": "오른손",
        "action": "검지와 중지로 RAM 모듈 상단 양끝을 집어 슬롯에서 위로 들어올린다",
        "objects_visible": ["RAM", "RAM_slot", "motherboard"],
    },
    {
        "timestamp": "00:00:11",
        "actor": "왼손",
        "action": "엄지로 지우개를 쥐고 RAM 모듈 하단 금색 접점을 좌우로 문지른다",
        "objects_visible": ["hand", "eraser", "RAM"],
    },
    {
        "timestamp": "00:00:20",
        "actor": "시선",
        "action": "전원 버튼의 LED가 황색으로 1회 깜빡인 뒤 백색으로 3회 깜빡인다",
        "objects_visible": ["power_button_LED"],
    },
]


def _fewshot_block() -> str:
    return json.dumps({"observations": _FEWSHOT_EXAMPLES}, ensure_ascii=False, indent=2)


def build_observation_messages(
    timestamp_label: str,
    detected_objects: List[Dict[str, Any]],
    injected_parts: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """프레임 1개에 대한 관찰 요청 메시지(텍스트 부분). 이미지 첨부는 어댑터가 결합한다.

    timestamp_label: 이 프레임의 시각 'HH:MM:SS'(샘플링이 부여 — VLM은 이 값을 그대로 쓴다).
    detected_objects: YOLO 검출 [{"class","conf","bbox":[x,y,w,h]}, ...] — 위치 힌트.
    injected_parts: [고정 사실] 부품 주입(기법 2). None이면 주입 없이 모델 자체 인식.
    """
    blocks: List[str] = []

    if injected_parts:
        blocks.append(
            "[고정 사실] 이 영상에 등장하는 부품은 다음과 같다(이 식별을 신뢰하라): "
            + ", ".join(injected_parts)
        )

    if detected_objects:
        blocks.append(
            "이 프레임에서 검출된 객체(class, conf, bbox=[x,y,w,h] 픽셀, 위치 힌트):\n"
            + json.dumps(detected_objects, ensure_ascii=False)
        )

    blocks.append("출력 형식 예시(이 형식을 정확히 따른다):\n" + _fewshot_block())
    blocks.append(
        f"이 프레임의 timestamp는 '{timestamp_label}' 이다. observations 의 timestamp 에 이 값을 그대로 넣어라.\n"
        "지금 이 프레임에서 보이는 동작을 관찰 규칙대로 기록하라. JSON만 출력."
    )

    return [
        {"role": "system", "content": VLM_OBSERVATION_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def build_video_observation_messages(
    injected_parts: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """**네이티브 비디오 모드(본선)** 메시지. 영상 전체를 한 번에 주고 관찰 로그를 받는다.

    영상 첨부(fps 샘플링)는 어댑터가 결합한다. 모델은 각 관찰에 비디오 기준 timestamp를 붙인다
    (fps 그리드에서 파생 — 자유 계산이 아니라 주어진 프레임 시각).
    """
    blocks: List[str] = []
    if injected_parts:
        blocks.append(
            "[고정 사실] 이 영상에 등장하는 부품은 다음과 같다(이 식별을 신뢰하라): "
            + ", ".join(injected_parts)
        )
    blocks.append("출력 형식 예시(이 형식을 정확히 따른다):\n" + _fewshot_block())
    blocks.append(
        "영상 전체를 시간순으로 보며, 보이는 동작을 관찰 규칙대로 하나씩 기록하라.\n"
        "각 observation 의 timestamp 는 해당 장면의 영상 기준 시각(HH:MM:SS)으로 적는다.\n"
        "observations 배열 JSON만 출력. 배열 밖 텍스트 금지."
    )
    return [
        {"role": "system", "content": VLM_OBSERVATION_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]
