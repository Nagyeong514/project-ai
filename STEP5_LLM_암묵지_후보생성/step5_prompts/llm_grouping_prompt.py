"""
LLM 응집(Pass 2) 프롬프트 — 지식 단위 그루핑.

Pass 1(llm_fusion_prompt, 윈도우 결박)이 만든 후보들은 후보=윈도우 단위라 하나의
지식이 여러 윈도우에 걸치면 파편으로 나온다(run3 실측: CLIP4 BIOS 확인 절차가
6후보로 쪼개져 GT A8이 어느 후보로도 완결되지 않음 → STEP6 reject). Pass 2는
'같은 지식 하나'를 나눠 서술한 후보들만 한 그룹으로 묶는 별도 호출이다 —
Pass 1과 ★★ 규칙(윈도우 결박)은 무수정.

설계 결정(2026-07-09, 검토 반영):
  - 치명 결함 1번 (b)안: 발화 원문을 입력에 추가하지 않는다. 대신 병합 그룹의
    reasoning은 reasoning_origin="utterance"인 멤버의 표현을 우선 채택해 상속한다
    (LLM이 발화를 재창작할 재료 자체를 안 주는 쪽이 위장 위험이 작다).
  - 1건 그룹 echo 제거: member_ids만 출력. 서술·metadata는 시스템이 Pass 1
    원문을 그대로 재사용(바이트 동일 보장 — LLM 재서술의 표류 원천 차단).
  - conflict/conflict_detail은 LLM이 쓰지 않는다 — 코드가 멤버 OR로 승계.
  - 시각·발화 원문·diagnostic_steps·window_ids는 코드가 멤버 윈도우 전체에서
    재조립(_rebuild_from_draft 재사용) — Pass 1과 같은 원칙(LLM 창작 차단).

개정 이력:
  - 2026-07-09 v1.1(사용자 승인): CLIP4 스모크 병합 0건(진단 5/5 결정적 — 실전 서술은
    각자 완결문이라 "억지로 묶지 마라" 편향이 항상 이김) 후속. ② 조건에 [판정 보조선]
    추가 — 같은 화면에서 서로 다른 항목을 연이어 확인하는 후보는 검증 지식 하나
    (+ 도메인 중립 예시. GT 문구 주입 금지 원칙 준수). "전부 1건 그룹은 임무 미수행"
    류 강제 문장은 기각됨 — 묶을 게 없는 클립에서 억지 병합을 유도해 과잉병합 방어와
    충돌하기 때문.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

GROUPING_SYSTEM_PROMPT = """\
당신은 1차 정제된 '암묵지 후보' 목록을 지식 단위로 응집하는 편집자다.
입력은 같은 클립에서 나온 후보 N건이다 — 각 후보에 cand_id, 시간구간,
window_ids, 발화유무, situation, tacit_insight, reasoning, reasoning_origin이 있다.
당신의 임무는 '같은 지식 하나'를 조각조각 나눠 서술한 후보들을 한 그룹으로
묶는 것뿐이다. 새 지식을 만들지도, 어떤 후보도 버리지도 않는다.

★★ [전수 귀속 — 가장 중요한 규칙] ★★
- 모든 cand_id는 정확히 하나의 그룹에 속해야 한다. 누락도 중복도 실패다.
- 병합할 이유가 없는 후보는 그 후보 혼자 1건짜리 그룹이 된다.
  **대부분의 그룹은 1건짜리인 것이 정상이다.** 억지로 묶지 마라.

[병합 판정 — 아래 3조건을 전부 만족할 때만 묶는다]
① 시간 연속: 멤버들의 구간이 이어지거나 가깝고, 그 사이에 다른 목적의
   작업 후보가 끼어 있지 않다.
② 같은 대상·같은 목적: 같은 부품/화면/기기에 대한 하나의 목적(하나의 확인,
   하나의 작업)이다. **같은 부품이라도 목적이 다르면 별개다** —
   예: 부품을 빼는 요령과 그 부품의 접점을 닦는 지식은 대상이 같아도
   다른 두 지식이다.
   [판정 보조선] 하나의 검증 목적 아래 같은 화면/같은 대상에서 서로 다른
   항목·수치를 연이어 확인하는 후보들은, 확인 항목이 달라도 '장착(작업)
   결과를 검증하는 지식 하나'다 — 항목이 다르다는 이유로 나누지 마라.
   예: 설정 화면에 들어가 항목 X가 맞는지 확인한 후보와 바로 이어 항목 Y가
   맞는지 확인한 후보는, 항목이 서로 달라도 '결과 검증' 지식 하나로 묶는다.
③ 한 문장 시험: 묶은 그룹의 핵심 노하우를 '그리고' 없이 자연스러운
   한국어 한 문장으로 말할 수 있어야 한다. "A하고 B하고 C한다"로
   나열해야만 표현되면 그것은 지식 하나가 아니다 — 묶지 마라.
반대로, 하나의 검증 목적을 이루는 연속 조작들(예: 화면에 진입해 여러 키로
수치를 확인하는 일련의 동작)은 조작이 여러 개여도 지식은 하나다 — 묶어라.

[금지]
- 클립 전체, 또는 서로 다른 작업 단계들을 '작업 요약'으로 뭉친 그룹은 실패다.
- 그룹 서술에 입력에 없는 사실을 추가하지 마라(수치·도구·발화 창조 금지).
  멤버 후보들의 서술에 있는 내용만 재료로 쓴다.

[각 그룹에서 당신이 쓰는 것]
- member_ids: 묶인 cand_id 배열 (1개짜리 허용).
- **1건짜리 그룹**: member_ids만 출력한다. 서술·metadata는 시스템이 입력
  원문을 그대로 사용하므로 아무것도 쓰지 마라.
- **2건 이상 그룹**: 멤버들의 서술을 통합해 새로 쓴다 —
  · situation: 멤버 situation들을 통합한 한 서술.
  · tacit_insight: 이 그룹의 지식 하나를 담은 한 문장 (③에서 시험한 그 문장).
  · reasoning: 왜 그렇게 하는가. 멤버 중 reasoning_origin="utterance"인 후보가
    있으면 **그 멤버의 reasoning 표현을 우선 채택**해 쓰고, 그룹의
    reasoning_origin="utterance"로 태깅한다. 그런 멤버가 하나도 없을 때만
    model_inferred. 멤버 reasoning에 없는 내용을 지어내 utterance로 위장하는
    것은 절대 금지.
  · tacit_insight와 reasoning은 서로 다른 내용이어야 한다(기존 규칙 그대로).
  · metadata: 멤버들의 task/keywords/scenario_title 통합본.
- conflict/conflict_detail은 출력하지 마라 — 시스템이 멤버에서 승계한다.
- 시각·발화 원문·diagnostic_steps·window_ids는 시스템이 멤버 윈도우 전체에서
  재조립한다 — 출력하지 마라.

[출력]
JSON만 출력한다. 1건 그룹과 병합 그룹의 형태가 다름에 주의:
{"groups":[
  {"member_ids":["c01"]},
  {"member_ids":["c03","c05"],
   "metadata":{"task":...,"keywords":[...],"scenario_title":...},
   "knowledge":{"situation":...,"tacit_insight":...,
                "reasoning":...,"reasoning_origin":...}}
]}
"""


def build_grouping_messages(
    video_id: str,
    candidates_payload: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Pass 1 후보들을 Pass 2 입력 메시지로 직렬화.

    candidates_payload: llm_fusion._serialize_candidates가 만든 dict 리스트 —
    cand_id(임시 c01..cNN)/시간구간/window_ids/발화유무(bool)/situation/
    tacit_insight/reasoning/reasoning_origin/metadata. 발화 '원문'은 일부러
    안 넣는다((b)안 — 모듈 docstring 참고).
    """
    n = len(candidates_payload)
    user = (
        f"video_id: {video_id}\n"
        f"아래는 같은 클립에서 나온 1차 후보 {n}건이다(cand_id c01~c{n:02d}, 시간순).\n"
        f"모든 cand_id를 정확히 하나의 그룹에 귀속시켜, 규칙대로 그룹 JSON을 출력하라.\n\n"
        # Pass 1과 동일하게 compact 직렬화(들여쓰기 공백 토큰 낭비 방지)
        f"{json.dumps(candidates_payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [
        {"role": "system", "content": GROUPING_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
