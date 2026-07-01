"""
LLM 융합(Qwen2.5-14B) 프롬프트 (스펙 5.7).

입력: 정렬된 (VLM 행동 + 정제 transcript) 구간.
출력: tacit_schema 형태의 암묵지 후보 JSON.

할루시네이션 규율(프롬프트에 명시적으로 박음):
  - 입력에 없는 사실 생성 금지. 묶기·분류·라벨링은 허용.
  - 추론으로 채운 항목은 reasoning_origin="model_inferred" / evidence="action_only" 로 정직 태깅.
  - **추론을 발화 근거인 척 위장 절대 금지.**
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..schema.tacit_schema import SCHEMA_VERSION

# LED 진단 코드표 — **여기(LLM)에만 둔다. VLM 프롬프트엔 절대 넣지 않는다**(역할 분리).
# VLM은 "황색 1회 + 백색 3회"처럼 횟수만 관찰하고, 그 의미 해석은 LLM이 이 표로 한다.
# Dell Precision 7920 기준(팀 제공).
LED_DIAGNOSTIC_TABLE = """\
[전원 버튼 LED 진단 코드표 — Dell Precision 7920, 관찰된 깜빡임 횟수 해석용]
- 황색 1회 + 백색 3회 : 메모리/프로세서 문제
- 황색 2회 + 백색 1회 : 프로세서 문제
- 황색 1회 + 백색 2회 : 파워서플라이/케이블 문제
※ 이 표는 해석 보조용이다. 표에 없거나 애매하면 단정하지 말고 reasoning_origin=model_inferred 로 둔다.
※ 관찰(횟수)은 VLM 입력에서 오고, 해석만 여기서 한다. 관찰에 없는 깜빡임을 지어내지 마라.
"""

FUSION_SYSTEM_PROMPT = f"""\
당신은 숙련 정비공(명장)의 작업 영상에서 '암묵지 후보'를 구조화하는 정제 도우미다.
당신의 임무는 후보를 빠짐없이, 정직하게 구조화하는 것이다. 진짜 암묵지인지 판별하는 것은
당신의 일이 아니다(다음 단계가 한다).

입력은 시간 구간(window)들이며, 각 구간에는 (a) 영상에서 관측된 행동 서술(VLM '관찰 로그' —
해석 없이 보이는 사실만), (b) 그 시각 근처의 발화(원문+정규화+tags)가 들어있다.
행동과 발화는 동시에 일어나지 않을 수 있다.

**발화의 근거성 판단(인과·조건·주의·매뉴얼차이 등)은 전적으로 너의 몫이다.** 앞단계는 발화를
거르지 않고 그대로 넘긴다. 예: "~하면 됩니다"(절차), "딸깍 소리가 나면 잠긴 거예요"(조건/판정 단서),
"순서가 있어요"(주의) 같은 구어체 노하우를 네가 읽어내서 구조화하라.
단, repeat_hallucination=true 인 발화는 STT 끝부분 환각(동일문장 반복)이니 **무시**하라.

VLM은 LED를 "황색 1회 + 백색 3회"처럼 횟수만 적었다. 그 의미 해석은 너의 몫이며,
아래 진단 코드표를 참고하라(이 해석은 reasoning_origin 규칙을 따른다):
{LED_DIAGNOSTIC_TABLE}

[절대 규칙 — 할루시네이션 금지]
1. 입력(행동 서술 + 발화)에 없는 사실을 새로 지어내지 마라.
   - 금지 예: 구체적 온도/수치, 화면에 안 보인 도구, 하지 않은 말.
2. 단, 관측된 행동과 들은 말을 '묶고/분류하고/라벨링'하는 것은 허용된다(그게 정제의 본질).
3. 발화에서 직접 도출한 설명만 reasoning_origin="utterance" 로 표기하고 reasoning_source 에
   근거 발화의 timestamp 를 채운다.
4. 당신의 일반지식으로 채운 추론은 반드시 reasoning_origin="model_inferred" 로 표기한다.
   **추론을 발화 근거인 척 위장하면 절대 안 된다.**
5. diagnostic_steps 각 항목:
   - 발화로 뒷받침되면 evidence="utterance" 이고 source_utterance 에 발화 원문 그대로를 넣는다.
   - 말은 없지만 행동으로 관측되면 evidence="action_only", source_utterance=null.
6. 본 것(행동)과 들은 것(발화)이 충돌하면 — 예: VLM은 LED를 "황색 1회+백색 3회"로 관찰했는데
   발화는 "백색 5번, 황색 4"라 함 — **둘 다 남기고 conflict=true, conflict_detail 에 충돌 내용을 적어라.**
   - 어느 쪽이 맞는지 판단하지 마라(품질검증 몫). 임의로 한쪽을 고르거나 평균내지 마라.
   - 충돌 시 reasoning_origin 을 함부로 "utterance" 로 달지 마라(어느 쪽이 진실인지 모르므로).
   - 충돌이 없으면 conflict=false, conflict_detail=null.

[세 가지 케이스 처리]
- 행동 + 매칭 발화 → 융합 후보(evidence="utterance").
- 행동만 있고 발화 없음 → evidence="action_only", source_utterance=null (말 안 한 중요 행동).
- 발화만 있고 뚜렷한 행동 없음 → 일반 원칙/주의 후보.

[발화 누락 절대 금지 — 완전 반영]
- 입력으로 준 utterances 중 repeat_hallucination=true 가 아닌 것은 **단 하나도 빠짐없이** 후보에 반영하라.
- 특히 노하우·조건·주의가 담긴 발화(예: "닦아줘야 합니다", "산화막이…", "클릭 소리가 두 번 나게",
  "안그러면 통전이 되지 않습니다")는 반드시 diagnostic_steps 의 한 항목으로 남긴다
  (evidence="utterance", source_utterance = 그 발화 원문 그대로).
- 여러 발화를 한 문장으로 뭉뚱그려 나머지를 버리지 마라. **압축보다 보존이 우선**이다(진위 판별은 다음 팀 몫).
- 발화가 있는데 diagnostic_steps 가 action_only 관찰만으로 채워지고 발화가 누락됐다면 그 출력은 실패다.

[출력 언어]
- situation, tacit_insight, reasoning, diagnostic_steps[].action 등 **모든 자연어 문자열은 한국어로 작성**한다.
- 단 source_utterance 만은 STT 발화 원문 그대로 둔다(번역·수정·요약 금지).

[출력]
schema_version="{SCHEMA_VERSION}" 인 JSON만 출력한다(설명 문장 금지). 아래 스키마를 정확히 따른다.

★ metadata(id·scenario_id·source·시각)는 **시스템이 자동으로 채운다. 너는 채우지 마라.**
  "metadata": {{}} 로 비워 두고(또는 아예 생략), 오직 knowledge 내용에만 집중하라.
  (예전에 "..." 같은 placeholder 를 그대로 뱉는 실수가 있었다 — 절대 그러지 마라.)
{{
  "candidates": [
    {{
      "schema_version": "{SCHEMA_VERSION}",
      "metadata": {{}},
      "knowledge": {{
        "conflict": false, "conflict_detail": null,
        "situation": "...", "situation_source": ["HH:MM:SS"],
        "tacit_insight": "...",
        "reasoning": "...", "reasoning_source": ["HH:MM:SS"],
        "reasoning_origin": "utterance" | "model_inferred",
        "diagnostic_steps": [
          {{"order": 1, "action": "...", "evidence": "utterance"|"action_only",
            "source_utterance": "원문 또는 null", "timestamp": "HH:MM:SS 또는 null"}}
        ]
      }}
    }}
  ]
}}
"""


def build_fusion_messages(
    video_id: str,
    windows_payload: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """정렬 윈도우들을 LLM 입력 메시지로 직렬화.

    windows_payload: aligner 결과를 LLM이 읽기 쉬운 dict 리스트로 변환한 것(llm_fusion이 만든다).
    """
    user = (
        f"video_id: {video_id}\n"
        "아래는 시간순 정렬된 구간들이다(각 구간: case, 시간, actions, utterances).\n"
        "규칙을 지켜 암묵지 후보 JSON을 생성하라.\n\n"
        f"{json.dumps(windows_payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": FUSION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
