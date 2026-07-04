정답지 기반 암묵지 최종 JSON 9건 (step6 실제 로직 기준 재계산, 벡터 DB 적재용)
====================================================================
주제: Dell Precision 7920 조립 후 부팅 실패 시나리오
출처: 촬영 시 만든 암묵지 정답지 9행을 그대로 매핑

⚠️ 2026-07-04 정정: 이 파일들은 실제 STEP3에서 VLM이 생성한 데이터가 아니라 손으로
만든 "예시"다(당시 VLM 0건 버그로 실제 데이터를 못 뽑아서 임시로 만듦). 아래 서술은
전부 이 예시 픽스처에 한정된 것이고, 실제 STEP6_품질검증 코드(read-only, 절대 수정 안 함)를
직접 돌린 결과가 아니다 — 그 코드가 실제로 하는 계산 방식(가중치·임계값·기본값)만
최대한 그대로 흉내 냈을 뿐이다.

[구조] 각 파일 = LLM 최종 스키마(1.3) + verification 블록(6-9, step6 실행 결과를 흉내낸 것)
  - knowledge.tacit_insight 를 임베딩해 벡터 DB에 적재하면 됨
  - verification 은 "step6를 실제로 돌리면 이렇게 나올 것"이라는 추정 기록

[action_only vs utterance]  ← 너희 기준 그대로 반영
  침묵 동작(말 없이 한 동작) = evidence:"action_only", source_utterance:null,
                              reasoning_origin:"model_inferred", timestamp=VLM 것
  발화 동반 동작           = evidence:"utterance", source_utterance=STT 원문(verbatim),
                              reasoning_origin:"utterance", timestamp=STT 것

[정정] "설계원칙 4: action_only 보호"는 실제 STEP6 코드에 없었음 — 제거함
  예전 버전은 침묵 동작(action_only) 후보를 발화전제 검사(reasoning_grounding/
  step_grounding_ratio/utterance_signal)에서 통째로 "면제"하고 action_reason_consistency
  하나로 가중치를 재정규화해 accept로 밀어줬다. 그런데 실제 STEP6_품질검증/graph.py·
  rules.py·llm_utils.py를 직접 읽어보니 이런 "면제" 로직이 어디에도 없었다:
    - step_grounding_ratio: 발화 스텝 0건이면 1.0으로 폴백(graph.py) — 면제 아니라 기본값.
    - utterance_signal: 발화 스텝 0건이면 rule_score=0.0(rules.py, 결정론적) 위에 LLM 보정.
    - reasoning_grounding: 스니펫이 없어도 LLM judge가 그대로 판단, 면제 없음.
  그래서 면제 정책을 걷어내고 이 3가지를 실제 코드 기준값(001/002는 ESTIMATED 표시,
  실제 LLM judge를 안 불러서 정확한 값은 모름 — 아래 참고)으로 채워 재계산했다.
  T_high/T_low도 예전엔 0.62/0.42라는 임의값을 썼는데, 실제 config.py 값인 0.70/0.40으로
  정정했다.

  결과: action_only 2건(001, 002)이 accept(0.87~0.88)에서 **hold(0.51 근처)로 바뀜.**
  발화 근거가 하나도 없는 후보를 "그냥 봐준다"는 예전 가정이 실제 코드엔 없었으므로,
  이게 지금 STEP6 코드 기준으로 더 정확한 추정치다.

  ※ reasoning_grounding/utterance_signal은 실제 LLM judge를 부르지 않고 만든 값이라
  각 JSON의 verification.confidence.estimated_scores에 어떤 필드가 추정치인지 표시해뒀다.
  STEP3 VLM 버그가 고쳐져서 실제 데이터가 나오면, 이 파일들 대신 진짜 STEP6 실행 결과로
  교체할 것 — 지금은 어디까지나 벡터 DB 파이프라인 테스트용 예시다.

[9건 요약]  (7건 accept, 2건 hold — 위 정정 반영)
  001 ch_preread       action_only  delta  0.511 hold    채널 사전 판독(삽입 전 라벨·색 확인)
  002 tactile_conn     action_only  delta  0.509 hold    커넥터 촉각 재확인
  003 led_read         utterance    delta  0.926 accept  LED 패턴 판독 -> 원인 계열 좁히기
  004 systematic_check utterance    delta  0.872 accept  의심 전 전체 체계적 점검
  005 latch_removal    utterance    novel  0.897 accept  래치 양쪽 동시 가압 탈거
  006 terminal_clean   utterance    novel  0.915 accept  단자 산화막 세척 우선(교체보다)
  007 latch_confirm    utterance    delta  0.901 accept  래치 잠금 청각+촉각 이중 확인
  008 channel_id       utterance    delta  0.900 accept  색·라벨로 채널 즉시 식별(6채널)
  009 bios_verify      utterance    delta  0.926 accept  BIOS 용량·채널 수치 최종 확인

[재생성] python3 make_gold_records.py  (서브스코어/임계값 조정은 이 파일 상단에서)
  ⚠️ 출력 경로 버그도 같이 고침: 예전엔 스크립트가 gold_records/gold_records/라는
  중첩 폴더에 새로 썼고 원본 파일은 안 덮어썼다. 이제 스크립트와 같은 폴더에 바로 쓴다.
