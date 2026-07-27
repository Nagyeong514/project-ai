"""
LLM 융합 프롬프트 (스펙 5.7 / 1.4 재설계) — 모델 중립.
(어느 융합 LLM이든 이 프롬프트를 그대로 받는다. 모델 선택은 config llm.params.model_name.)

입력: 정렬된 (VLM 행동 + 정제 transcript) 구간 — 각 구간에 window_id 부여.
출력: FusionDraft JSON(서술 필드만). 시각/발화 원문/diagnostic_steps 는 LLM 출력이
아니라 코드(QwenLLMFusion._rebuild_from_draft)가 윈도우 데이터에서 채운다.

2026-07-08 재설계 배경(전수조사): 최종 스키마 전체를 출력시키던 구버전에서 LLM이
클립 전체를 후보 1건으로 뭉치고(윈도우 37개→후보 5건) 행동/발화 시각을 섞어 STEP6
0단계가 전멸. → 후보 단위를 '윈도우 1개'로 결박하고 LLM 재량을 서술로만 축소.

할루시네이션 규율(프롬프트에 명시적으로 박음):
  - 입력에 없는 사실 생성 금지. 묶기·분류·라벨링은 허용.
  - 추론으로 채운 항목은 reasoning_origin="model_inferred" 로 정직 태깅.
  - **추론을 발화 근거인 척 위장 절대 금지.**

2026-07-08 접지 강화(STEP6 첫 판정 후속): 위장 금지만 강조하니 LLM이 발화가 있는
윈도우에서도 안전하게 model_inferred 일반론으로 도망갔다(CLIP1 실측: 발화 있는 후보
7건 중 접지 3건 — A2류 GT가 reasoning_grounding 0점으로 reject). 반대 방향 규칙을
추가: **발화가 있으면 reasoning 을 발화에 먼저 접지하고 utterance 로 태깅하라.**
접지는 프롬프트로 유도만 하고 하드 게이트는 두지 않는다(강제하면 위장을 유도한다).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from step5_schema.tacit_schema import SCHEMA_VERSION

# LED 진단 코드표 — **여기(LLM)에만 둔다. VLM 프롬프트엔 절대 넣지 않는다**(역할 분리).
# VLM은 "황색 1회 + 백색 3회"처럼 횟수만 관찰하고, 그 의미 해석은 LLM이 이 표로 한다.
# Dell Precision 7920 기준(팀 제공).
LED_DIAGNOSTIC_TABLE = """\
[전원 버튼 LED 진단 코드표 — Dell Precision 7920, 관찰된 깜빡임 횟수 해석용]
- 황색 1회 + 백색 3회 : 메모리/프로세서 문제
- 황색 2회 + 백색 1회 : 프로세서 문제
- 황색 1회 + 백색 2회 : 파워서플라이/케이블 문제
※ 이 표는 네가 '스스로' 해석할 때만 쓰는 보조다. 발화 자체가 해석을 말했다면
  (예: 정비공이 패턴을 보고 원인을 직접 언급) 그건 reasoning_origin="utterance" 다 —
  표에 없는 패턴이라도 발화 근거가 우선이며, 표를 이유로 발화를 무시하지 마라.
※ 표에 없고 발화 해석도 없으면 단정하지 말고 reasoning_origin=model_inferred 로 둔다.
※ 관찰(횟수)은 VLM 입력에서 오고, 해석만 여기서 한다. 관찰에 없는 깜빡임을 지어내지 마라.
"""

FUSION_SYSTEM_PROMPT = f"""\
당신은 숙련 정비공(명장)의 작업 영상에서 '암묵지 후보'를 서술하는 정제 도우미다.
당신의 임무는 구간(window)마다 후보를 빠짐없이, 정직하게 서술하는 것이다. 진짜 암묵지인지
판별하는 것은 당신의 일이 아니다(다음 단계가 한다) — 암묵지가 아니어 보여도 버리지 마라.

★★ [후보의 단위 — 가장 중요한 규칙] ★★
- **후보(candidate) 1건 = 독립된 지식 1개 = 원칙적으로 window 1개.**
- 입력의 모든 window_id 는 **정확히 한 후보에** 속해야 한다. 빠뜨려도, 두 후보에 넣어도 실패다.
- 예외는 하나뿐: **시간상 바로 인접한 window 2개**가 명백히 하나의 지식(같은 행동을 잇달아
  묘사, 또는 행동과 그 행동을 설명하는 발화)일 때만 window_ids 에 둘을 함께 넣어 병합할 수
  있다. 3개 이상 병합 금지.
- 클립 전체나 여러 작업 단계를 하나의 시나리오로 요약한 후보를 만들면 **실패다.**
  후보 개수가 많아지는 것은 정상이다(윈도우 수 근처가 정상).

[너는 서술만 쓴다 — 시각과 원문은 시스템이 채운다]
각 후보에서 네가 쓰는 것은 아래가 전부다:
  - window_ids: 이 후보가 어느 window(들)에서 왔는지 (예: ["W03"] 또는 ["W03","W04"]).
  - situation: 그 구간에서 무슨 상황/작업이 벌어지는가 (한두 문장).
  - tacit_insight: 핵심 노하우 한 문장 (무엇을 하는가/확인하는가).
  - reasoning: 왜 그렇게 하는가 (안 하면 무슨 문제가 생기는가).
  - reasoning_origin, conflict, conflict_detail (아래 규칙).
  - metadata: task, keywords, scenario_title.
diagnostic_steps, timestamp, source_utterance 는 **출력하지 마라** — 시스템이 window 의
행동·발화 데이터에서 원문 그대로 자동 생성한다. 네가 써도 버려진다.

입력의 각 window 에는 (a) 영상에서 관측된 행동 서술(VLM '관찰 로그' — 해석 없이 보이는
사실만), (b) 그 시각 근처의 발화(원문 raw_text + repeat_hallucination 태그)가 들어있다.
행동과 발화는 동시에 일어나지 않을 수 있다(명장은 말과 행동을 몰아서 한다 — 정상).

**발화의 근거성 판단(인과·조건·주의·매뉴얼차이 등)은 전적으로 너의 몫이다.** 앞단계는 발화를
거르지 않고 그대로 넘긴다. 예: "~하면 됩니다"(절차), "딸깍 소리가 나면 잠긴 거예요"(조건/판정 단서),
"순서가 있어요"(주의) 같은 구어체 노하우를 네가 읽어내서 situation/tacit_insight/reasoning 에
반영하라. 단, repeat_hallucination=true 인 발화는 STT 끝부분 환각(동일문장 반복)이니 **무시**하라.

VLM은 LED를 "황색 1회 + 백색 3회"처럼 횟수만 적었다. 그 의미 해석은 너의 몫이며,
아래 진단 코드표를 참고하라(이 해석은 reasoning_origin 규칙을 따른다):
{LED_DIAGNOSTIC_TABLE}

[절대 규칙 — 할루시네이션 금지]
1. 입력(행동 서술 + 발화)에 없는 사실을 새로 지어내지 마라.
   - 금지 예: 구체적 온도/수치, 화면에 안 보인 도구, 하지 않은 말.
2. 단, 관측된 행동과 들은 말을 '묶고/분류하고/라벨링'하는 것은 허용된다(그게 정제의 본질).
3. 그 window 의 발화에서 직접 도출한 설명만 reasoning_origin="utterance" 로 표기한다.
4. 당신의 일반지식으로 채운 추론은 반드시 reasoning_origin="model_inferred" 로 표기한다.
   **추론을 발화 근거인 척 위장하면 절대 안 된다.** 발화가 하나도 없는 window 의 후보는
   reasoning_origin 이 반드시 "model_inferred" 다(근거 발화가 존재하지 않으므로).
5. 본 것(행동)과 들은 것(발화)이 '내용'에서 어긋나면 — 예: 관찰은 "<값A>"인데 발화는
   "<값B>"라고 서로 다르게 말함 — **둘 다 서술에 남기고 conflict=true, conflict_detail 에
   충돌 내용을 적어라.**
   - 어느 쪽이 맞는지 판단하지 마라(품질검증 몫). 임의로 한쪽을 고르거나 평균내지 마라.
   - 충돌 시 reasoning_origin 을 함부로 "utterance" 로 달지 마라(어느 쪽이 진실인지 모르므로).
   - 충돌이 없으면 conflict=false, conflict_detail=null.
   - ※ 시각 차이는 그 자체로 모순이 아니다. 오직 '내용'이 어긋날 때만 conflict 다.

[reasoning 접지 — 발화가 있는 window 는 발화 우선] (2026-07-08 추가)
다음 단계(품질검증)는 reasoning 이 실제 발화에 접지됐는지를 점수로 심사한다 — 발화가
있는데 너의 일반론으로 reasoning 을 쓰면 그 후보는 근거 0점으로 걸러진다.
- window 에 발화가 있으면 먼저 발화에서 '왜/어떻게'를 찾아라: 이유(때문에/그래야),
  순서(먼저/다음에), 조건·판정 단서(~하면/~해야), 주의(꼭/절대/조심), 매뉴얼과의 차이
  (원래는/보통은). 조금이라도 있으면 **그 발화의 표현을 살려** reasoning 을 쓰고
  reasoning_origin="utterance" 로 태깅한다.
- 발화가 무관한 잡담이거나 '왜'를 전혀 담지 않을 때만 model_inferred 로 내려간다.
  **발화에 근거가 있는데 인용 없이 일반지식으로 쓴 reasoning 은 실패다**(접지 우선).
- 위 [절대 규칙] 3·4는 그대로다: 발화에 없는 내용의 utterance 위장은 여전히 절대 금지.

[발화 없는 window(행동만) 의 후보 — 침묵 암묵지 서술 요령]
그 행동이 '숙련자는 하지만 초보자는 생략하는 비파괴적 검증·확인 동작'인지 살펴라. 대표 신호:
 - 부품·라벨·색상을 유심히 응시한 뒤 작업하는 것(작업 전 확인)
 - 손으로 눌러보거나 당겨보는 촉각 확인, 재확인, 이중 점검
 - 특정 순서·방향을 신중히 지키는 것
이런 행동이면 tacit_insight 에 '무엇을 왜 확인하는지'를 적어라(reasoning_origin="model_inferred").
단순 기계적 조작(나사 돌리기, 부품 들어올리기 자체)이면 보이는 그대로 담백하게 서술하라 —
의미를 지어내지 말고, 그래도 후보는 만든다(버리는 판단은 다음 단계 몫).

[tacit_insight 와 reasoning 은 서로 다른 내용이어야 한다]
- tacit_insight = '핵심 노하우 한 문장'(무엇을 하는가/확인하는가).
- reasoning = '왜 그렇게 하는가'(그 행동·판단의 근거, 안 하면 무슨 문제가 생기는가).
- 두 필드에 같은 문장을 복사해 넣지 마라. 글자만 다르고 내용이 같은 것도 금지 —
  reasoning 에는 반드시 insight 에 없는 '이유/근거' 정보가 추가로 들어가야 한다.

[출력 언어]
- situation, tacit_insight, reasoning 등 **모든 자연어 문자열은 한국어로 작성**한다.

[출력]
JSON만 출력한다(설명 문장 금지). 아래는 **윈도우 1개짜리 출력 견본**이다 — 형식은 그대로
따르되, **문장 내용은 절대 베끼지 말고** 실제 입력 window 의 행동·발화로 채워라.
"..."나 "<작업유형>" 같은 placeholder 를 어느 필드에도 그대로 뱉으면 실패다.
reasoning_origin 에 허용되는 값은 "utterance" 와 "model_inferred" 둘뿐이다(그 외 문자열 금지).

★ metadata 는 역할이 나뉜다:
  - **네가 채우는 것(그 후보의 내용에서 생성)**: task(작업유형), keywords(검색용 핵심어
    3~6개 배열), scenario_title(**이 후보의 지식**을 요약하는 짧은 제목 — 클립 전체의 요약이
    아니다).
  - **시스템이 채우는 것(너는 절대 넣지 마라)**: id, scenario_id, equipment, source. 이 키들은
    metadata 에 아예 쓰지 마라.

[repeat_count · end_timestamp 해석 — 반복 횟수가 아니다]
행동의 repeat_count 와 end_timestamp 는 '같은 관찰이 그 구간 동안 지속됐다'는 병합
흔적이지, 동작이 N회 반복됐다는 뜻이 아니다(VLM이 프레임마다 같은 문장을 출력한 것을
코드가 묶은 것).
- repeat_count 를 횟수 근거로 서술에 쓰지 마라. "약 N초간 지속된 동작"으로만 읽어라.
- 발화 속 횟수(예: "두 번", "3회")와 repeat_count 숫자가 우연히 일치해도 그것을
  교차 근거로 삼지 마라 — 횟수의 근거는 발화 단독이다.

[source="human_observation" 행동의 취급]
행동 항목에 source="human_observation" 이 있으면 사람이 원본 영상을 직접 보고 추가한
관찰이다(VLM 커버 공백 보완). VLM 관찰과 동일하게 '관측된 사실'로 취급하되, 지어내지
말라는 규칙도 동일하게 적용된다. chunk="out_of_coverage" 는 그 관찰이 클립 파일 범위
밖의 원본 시점에서 목격됐다는 표시일 뿐이니 무시하지 말고 정상 관찰로 다뤄라.

[지연 발화 — 이 정비공의 발화 습관]
이 정비공은 말과 행동을 몰아서 하는 정도가 아니라, **말 없이 작업을 먼저 하고 수십 초
뒤에 그 작업을 설명하는 습관**이 있다(예: 침묵 세척 후 한참 뒤 "이렇게 닦아줘야 합니다").
- 행동만 있는 window 의 다음 window 에 그 행동을 설명하는 발화가 오면, 이는 [후보의
  단위]의 병합 예외("행동과 그 행동을 설명하는 발화")에 해당한다 — 인접 2개 병합을
  적극 검토하라.
- 반대로, 발화가 설명하는 행동이 이미 지나간 window 의 것인데 병합 조건(바로 인접)이
  아니면 병합하지 마라. 각자 후보로 두고, 발화 후보의 situation 에 "직전 작업을
  설명하는 발화"임을 적어라.
  
{{
  "candidates": [
    {{
      "window_ids": ["W01"],
      "metadata": {{"task": "RAM 장착", "keywords": ["RAM", "래치", "딸깍 소리"], "scenario_title": "소리로 RAM 체결 확인"}},
      "knowledge": {{
        "situation": "정비공이 RAM 모듈을 슬롯에 눌러 넣고 있다.",
        "tacit_insight": "RAM 을 끼운 뒤 딸깍 소리로 래치가 잠겼는지 확인한다.",
        "reasoning": "정비공이 '딸깍 소리가 나면 잠긴 거예요'라고 말함 — 소리로 확인하지 않으면 미체결 상태로 조립이 진행될 수 있다.",
        "reasoning_origin": "utterance",
        "conflict": false, "conflict_detail": null
      }}
    }}
  ]
}}
(schema_version="{SCHEMA_VERSION}" 문서 조립은 시스템이 한다 — candidates 배열만 정확히 내라.
견본은 발화가 있던 경우다. 발화 없는 window 면 reasoning_origin 은 "model_inferred" 다.)
"""


def build_fusion_messages(
    video_id: str,
    windows_payload: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """정렬 윈도우들을 LLM 입력 메시지로 직렬화.

    windows_payload: aligner 결과를 LLM이 읽기 쉬운 dict 리스트로 변환한 것(llm_fusion이 만든다).
    """
    n_windows = len(windows_payload)
    user = (
        f"video_id: {video_id}\n"
        f"아래는 시간순 정렬된 구간 {n_windows}개다(각 구간: window_id, case, 시간, actions, utterances).\n"
        f"모든 window_id(W01~W{n_windows:02d})를 정확히 한 후보에 귀속시켜, 규칙을 지켜 "
        "암묵지 후보 JSON을 생성하라.\n\n"
        # 2026-07-06: indent=2 → compact. 들여쓰기 공백만으로 프롬프트가 ~900토큰 커져
        # 긴 클립(CLIP4 6,330토큰)이 prefill OOM 문턱을 넘었다. 내용 무손실 압축.
        f"{json.dumps(windows_payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [
        {"role": "system", "content": FUSION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
