"""공통 프롬프트 — 세 조건(및 ablation 변형)에 동일 적용.

설계 교훈(1차 실측):
- 3B 모델이 약해서 JSON '스키마 안의 설명문/예시'를 그대로 베껴 적었다.
  → 스키마에 채울-빈칸 설명을 넣지 않는다. 예시는 본문 밖에서 1개만, 그것도
    실험 도메인(자전거)과 무관한 요리 예시로 둬서 키워드 누수를 막는다.
- 출력이 잘려 JSON 파싱 실패가 잦았다 → 항목 수를 제한(간결)하고 max_new_tokens는 config에서 올린다.

ablation 입력 대응:
- stt_text=None  → 영상만(video-only)
- frame_paths=[] → 발화만(text-only)
"""
from __future__ import annotations

SYSTEM = (
    "당신은 숙련 작업자의 작업을 분석해 '암묵지'를 뽑아내는 전문가다. "
    "암묵지는 단순 동작 나열이 아니라 말로 잘 드러나지 않는 판단·감각·요령"
    "(각도·압력·순서·타이밍·상태판단의 기준)이다. "
    "주어진 근거(영상 프레임/발화)에서 실제로 보이거나 들린 것에만 기반하라. "
    "없는 내용을 지어내지 마라. "
    "**반드시 한국어로만** 답하라(중국어·영어 금지). "
    "JSON 키는 정확히 action, tacit, evidence 만 쓴다(변형 금지)."
)

# 도메인 무관 예시 1개 (자전거 키워드 누수 방지) — 형식만 보여줌
_EXAMPLE = (
    '예시(다른 분야):\n'
    '{"knowledge_points":[{"action":"칼을 약간 눕혀 당기며 썬다",'
    '"tacit":"칼날을 15도쯤 눕혀야 재료가 안 으스러진다",'
    '"evidence":"손목 각도와 단면이 매끈한 프레임"}]}'
)


def _instruction(stt_text, has_frames):
    parts = ["이 작업 구간의 암묵지를 추출하라."]
    if has_frames and stt_text:
        parts.append("근거: 아래 프레임들 + 발화 전사.")
    elif has_frames:
        parts.append("근거: 아래 프레임들. (발화 정보는 주어지지 않음 — 영상만 보고 판단)")
    else:
        parts.append("근거: 아래 발화 전사만. (영상은 주어지지 않음)")

    if stt_text:
        parts.append(f"\n[발화 전사]\n{stt_text}")

    parts.append(
        "\n핵심 노하우 위주로 최대 4개만, 아래 JSON 형식으로만 답하라(설명·머리말 금지):\n"
        '{"knowledge_points":[{"action":"...","tacit":"...","evidence":"..."}]}\n'
        "- action: 관찰된 구체적 행동\n"
        "- tacit: 그 속의 판단/감각/요령 (구체적으로. '잘 한다' 같은 두루뭉술 금지)\n"
        "- evidence: 근거(프레임에서 본 것/전사에서 들린 말)\n"
        "확실치 않으면 넣지 마라. 핵심을 빠뜨리지 마라.\n\n" + _EXAMPLE
    )
    return "\n".join(parts)


def build_messages(stt_text: str | None, frame_paths: list[str]) -> list[dict]:
    """Qwen2.5-VL chat messages. frame_paths 비면 text-only, stt None이면 video-only."""
    content = [{"type": "image", "image": p} for p in frame_paths]
    content.append({"type": "text", "text": _instruction(stt_text, bool(frame_paths))})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
