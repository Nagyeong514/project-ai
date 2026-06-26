"""공통 프롬프트 — 세 조건(A/B/C)에 100% 동일하게 적용.

프롬프트가 변수가 되면 안 됨. 여기 한 곳에서만 정의.
출력은 채점·정답지 대조가 쉽도록 구조화된 JSON을 요구.
"""
from __future__ import annotations

SYSTEM = (
    "당신은 숙련 작업자의 작업 영상을 분석해 '암묵지(tacit knowledge)'를 추출하는 전문가다. "
    "암묵지란 단순 동작 나열이 아니라, 말로 잘 드러나지 않는 판단·감각·요령(예: 각도/압력/타이밍/"
    "상태 판단의 기준)을 뜻한다. 영상 프레임과 발화 전사를 함께 보고, 실제로 보이거나 들린 근거에만 "
    "기반해 서술하라. 영상/음성에 없는 내용을 지어내지 마라."
)

# 출력 스키마 — 정답지(answerkey)와 같은 필드로 맞춰 자동 대조 가능
INSTRUCTION = """다음 입력(프레임 시퀀스 + 발화 전사)을 보고, 이 구간에서 드러나는 암묵지를 추출하라.

[발화 전사]
{stt}

아래 JSON 형식으로만 답하라. 다른 말 금지.
{{
  "knowledge_points": [
    {{
      "action": "관찰된 구체적 행동(무엇을 어떻게)",
      "tacit": "그 행동 속 판단/감각/요령(말로 잘 안 드러나는 핵심)",
      "evidence": "근거 (프레임에서 본 것 / 전사에서 들린 말)"
    }}
  ]
}}
규칙:
- 두루뭉술한 서술 금지("잘 다듬는다" X → "각도를 ~로 유지하며 압력을 줄인다" O).
- 근거 없는 항목 금지. 확실치 않으면 넣지 마라.
- 핵심 노하우를 빠뜨리지 마라."""


def build_messages(stt_text: str | None, frame_paths: list[str]) -> list[dict]:
    """Qwen2.5-VL chat 형식 messages. 이미지(프레임) + 텍스트."""
    content = [{"type": "image", "image": p} for p in frame_paths]
    content.append({"type": "text",
                    "text": INSTRUCTION.format(stt=(stt_text or "(없음)"))})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
