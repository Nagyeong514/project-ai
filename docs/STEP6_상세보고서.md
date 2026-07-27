# STEP6(암묵지 품질 검증) 상세 기술 보고서

> 목적 (앞 보고서들과 동일)
> 1. **재디벨롭 참고 문서** — 게이트·가중치·임계값·손잡이를 바로 찾는다.
> 2. **트러블슈팅 아카이브** — 어댑터 충돌·침묵 후보 accept 불가·접지 위장 통과 등의 사건과 해결.
>
> **작성 규칙(엄수):** 코드를 실제로 따라가 확인된 것만 기술. **사실과 추론을 구분 표기**
> (추론은 "추론:"/"추정:"). 확인 불가는 "**(확인 불가)**" 명시 후 §8에 재수집. 모든 주장에
> 근거 위치(파일:함수:라인 / config / git 커밋). **코드 수정 없음(조사·평가 전용).**
> 작성 기준 시점: 2026-07-16. 기준 커밋 `45be634`. 대상 폴더: `STEP6_암묵지_품질검증/`(안나경 내부).

> **이 보고서의 세 축(요청):** ① 원본 설계 / 수정 사항 구분(§5) ② STEP5↔STEP6 어댑터
> 경계면 감사(§2) ③ 판정 로직에 정답지(GT) 무유입 무결성(§4).

---

## §1. 전체 구조 개관

### 1-1. 엔트리포인트 → 판정 순서도

```
[입력] step6_adapter가 만든 폴더:  <candidate>.json (후보 1건=파일 1개) + transcripts/<id>_segments.json
        │
        ▼  main.run()  (main.py:137-218)
   preflight(무거운 모듈 import 전 5초 검증) → 통과 후 embedding/llm/graph import
   임베딩(bge-m3)·매뉴얼 벡터스토어(Chroma)·LLM(judge) 1회 로드
        │  후보마다 graph.invoke(init_state)   (graph.py build_graph)
        ▼
   [0단계] timestamp_check      (rules.check_timestamp_validity — 결정론, 탈락조건)
        │   FAIL → reject_terminal (STOP, confidence 계산 없이 reject)
        ▼ PASS
   [1단계] gate_a_search         (embedding_utils.search_manual — 매뉴얼 RAG)
        │   1차 쿼리 "equipment task keywords" → top1 유사도 < 0.45 면
        │   tacit_insight로 1회 재검색(더 높으면 교체)
        ▼
   [2단계] gate_a_judge          (llm_utils.judge_manual_relation — LLM, 탈락조건)
        │   relation ∈ {same, delta, novel};  same → reject_terminal (STOP)
        ▼ delta/novel
   [3단계] gate_bc_scores        (LLM judge 점수 4종, 서로 독립)
        │   Gate B: action_reason_consistency
        │   Gate C-1: reasoning_grounding
        │   Gate C-2: step_grounding_ratio (utterance 스텝만)
        │   Gate C-3: utterance_signal (rules 마커규칙 + LLM 보정)
        ▼
   [4단계] confidence_route      (가중합 → 라우팅)
        │   silent 트랙 여부 판정 → 가중합 → confidence
        │   confidence ≥ T_high(0.70) → accept / ≥ T_low(0.40) → hold / else → reject
        ▼
[출력] output_dir/<id>_result.txt     (0~4단계 전체 로그 + 최종 요약)
       output_dir/<id>_verified.json  (원본 + "verification" 블록: decision/confidence/사유/점수)
```

### 1-2. 설계 원칙 4종의 코드적 실체 (검증)

| # | 원칙 | 코드적 실체 | 판정 |
|---|---|---|---|
| ① | 탐색만, 생성 금지 — 증거 부족이면 채우지 않고 conf 낮춰 hold | 모든 judge 프롬프트 상단에 `HALLUCINATION_GUARD`("입력에 없는 사실 추론·생성 금지, 근거 부족이면 낮은 점수", llm_utils.py:114-119). 낮은 점수 → conf 하락 → T_low~T_high 사이면 hold(graph.py:279-284) | **성립(사실)** |
| ② | 탈락조건(결정론) vs 점수항목(LLM judge) 분리 | **탈락조건 2개**: timestamp_validity(rules.py, **결정론**) + Gate A same(llm_utils, **LLM**). **점수항목**: Gate B/C 4종(LLM). 즉 축은 "결정론 vs LLM"이 아니라 **"탈락 vs 점수"** — Gate A same은 LLM 기반 탈락조건이라 "결정론=탈락"이라는 단순 대응은 부정확 | **부분 성립(사실)** — §3-1 정밀 서술 |
| ③ | 비용순 실행 — 결정론 먼저, LLM judge 뒤, 탈락 확정 시 중단 | 그래프 순서: 0단계 결정론(LLM 없음) → 1단계 임베딩 검색 → 2단계 LLM judge, `same`/`timestamp FAIL`이면 `reject_terminal`로 조기 종료(graph.py:332-343). 조기 종료 시 3·4단계 LLM 점수 계산 건너뜀 | **성립(사실)** |
| ④ | action_only 보호 — 발화 전제 검사는 evidence=utterance에만 | timestamp_validity: `evidence=="utterance"`만 실재 대조, action_only는 면제(rules.py:123-138). step_grounding_ratio: utterance 스텝만 분모(graph.py:213). utterance_signal: utterance 스텝만 마커 검사(rules.py:150) | **성립(사실)** |

### 1-3. STEP5↔STEP6 역할 경계 재감사 (이중 필터 우려)

- **사실**: STEP5는 후보를 **가치로 탈락시키지 않는다.** STEP5의 게이트(`_check_window_binding`=전수
  귀속·병합 상한, `_ungrounded_utterance_claims`=접지 정직 태깅)는 **형식·정직성 강제**이지
  accept/reject 판정이 아니다 — 실패해도 재시도·태깅 교정을 할 뿐 후보를 버리지 않는다(STEP5
  보고서 §3-4). "진짜 암묵지인지 판별은 다음 단계" 원칙이 프롬프트·러너에 명시(llm_fusion_prompt.py:49-50).
- **사실**: STEP6는 STEP5의 `case` 필드를 **읽지 않는다**(grep 결과 STEP6 소스에 참조 0건). STEP6는
  후보를 raw dict로 받아 자체 게이트로만 판정한다.
- **추론:** 이중 필터는 아니다. 다만 **커플링은 있다** — STEP5가 접지를 정직 태깅
  (reasoning_origin=utterance/model_inferred)한 결과가 STEP6 reasoning_grounding 채점의 입력이 된다.
  이는 "필터→필터"가 아니라 "정직 태깅→채점"이라 분리 원칙 위반은 아니지만, STEP5의 태깅 성향이
  STEP6 점수에 영향을 준다는 점은 재디벨롭 시 인지할 지점이다(§7).

**근거 위치(§1):** `main.py:137-218`, `graph.py:70-346`, `rules.py:56-140`, `llm_utils.py:114-119`,
`config.py`, `STEP5_.../llm_fusion.py`(그라운딩 게이트), grep(case 미참조).

---

## §2. 어댑터 경계면 감사 (step6_adapter)

> STEP5→STEP6 경계 어댑터는 **`STEP5_LLM_암묵지_후보생성/step6_adapter.py`**(STEP5 소유)이다.
> STEP6→STEP7 어댑터는 `STEP6_암묵지_품질검증/step7_adapter.py`(별개, §2-5).

### 2-1. 견본 개발 → 실물 첫 만남의 스키마 불일치 (사실)

STEP6는 견본 파일(`input/tk_dell7920_mem_boot_001.json`)로 개발됐고, STEP5 실물과 처음 만났을 때
두 층위의 스키마 불일치가 드러났다(step6_adapter.py:4-19):

| 층위 | STEP6 전제 | STEP5 실물 |
|---|---|---|
| 파일 구조 | "파일 하나 = 후보 하나" | 영상 하나당 `{"candidates":[...]}` 배열 |
| transcript | `{"segments":[{"timestamp","end","text"}]}` | `{"utterances":[{"start","end","raw_text",...}]}` |

**"STEP6 코드 무수정" 원칙**(README.md:4 "생성/수정하지 않고 판정만") 하에, 차이는 **전량 어댑터가
흡수**한다(step6_adapter.py:8-10 "STEP6 코드는 한 줄도 안 건드리고 우리 쪽에서 메꾼다").

### 2-2. 어댑터 변환 전수 목록 (사실)

| 종류 | 변환 | 위치 |
|---|---|---|
| 파일 분리 | `candidates[]` 배열 → 후보마다 `<id>.json` 1파일 | `split_candidates`(step6_adapter.py:102-115) |
| 필드 매핑(transcript) | `utterances[].start`→`segments[].timestamp`, `.end`→`.end`, `.raw_text`→`.text` | `convert_transcript`(step6_adapter.py:44-57) |
| 타입 변환 | `start`(float 초) → `"HH:MM:SS"`(round 후) | `seconds_to_hhmmss`(step6_adapter.py:33-41) |
| 기본값 주입 | `metadata.source.transcript_ref` = `transcripts/<id>_segments.json`로 덮어씀 | step6_adapter.py:112-113 |
| 버리는 필드 | `normalized_text`, `repeat_hallucination`(STEP6은 모름 — segment엔 timestamp/end/text만) | step6_adapter.py:52-57 |
| 경계 겹침 보정 | 앞 segment `end`를 다음 `start`와 안 겹치게 깎음 | step6_adapter.py:59-74 (§2-4) |

- 후보 본문(id/metadata/knowledge)은 **그대로 통과**(deep copy 후 transcript_ref만 수정,
  step6_adapter.py:111-113). 원본 STEP5 tacit_json·STEP3 transcript는 **읽기만**(무변경, step6_adapter.py:12-13).
- 변환 직후 **전수 대조 검증**(건수·원문·시각) + write→read 왕복 검증(step6_adapter.py:76-94,142-144).
  하나라도 안 맞으면 `AssertionError`로 즉사(조용한 손실 방지).

### 2-3. 키 계약 현황표 — "STEP6이 읽는데 어댑터에 없는 키"가 0인가 (사실)

STEP6가 후보/transcript에서 실제로 읽는 키(중첩 포함)를 코드에서 전수 추출한 결과:

| STEP6가 읽는 경로 | 읽는 위치 | 어댑터/STEP5가 제공? |
|---|---|---|
| `id` | main.py:190 | ✅ (통과) |
| `metadata.source.clip_start/clip_end/transcript_ref` | rules.py:63-65, main.py:48 | ✅ (transcript_ref는 어댑터가 덮어씀) |
| `metadata.equipment/task/keywords` | graph.py:99,105 | ✅ |
| `knowledge.situation/situation_source` | graph.py:203, rules.py:115 | ✅ |
| `knowledge.reasoning/reasoning_source/reasoning_origin` | graph.py:203, llm_utils.py:247,276,279, rules.py:119-120 | ✅ |
| `knowledge.tacit_insight` | graph.py:120,143 | ✅ |
| `knowledge.diagnostic_steps[].order/action/evidence/source_utterance/timestamp` | rules.py:124-131, graph.py:213-217 | ✅ |
| transcript segment `.timestamp/.end/.text` | rules.py:30-34 | ✅ (convert_transcript 생성) |

**결론(사실)**: 현재 STEP5 tacit_schema 1.4 + step6_adapter 조합에서 **"읽는데 없는 키"는 0**이다.
- STEP5는 `utterance_timestamp`도 출력하나 STEP6는 `timestamp`만 읽어 무해(잉여 키).
- **주의(사실)**: STEP6 자체 `schemas.py`(pydantic)는 **어디서도 import되지 않는다**(grep 확인).
  STEP6는 후보를 raw dict로 처리하므로 입력 스키마 검증이 런타임에 없다 — 계약 위반은
  스키마 예외가 아니라 **`KeyError`로 늦게** 드러난다(예: `candidate["knowledge"]` 직접 접근, rules.py:112).

### 2-4. timestamp round() 어긋남의 현재 처리 (사실)

- STEP5 `_rebuild_from_draft`는 발화 시각을 `round()`로 정수초화(llm_fusion.py:549). 어댑터
  `seconds_to_hhmmss`도 `round()`(step6_adapter.py:33-41). → **양쪽 round()로 초 단위 일치**.
- 과거 문제: STT 발화가 빈틈없이 이어져 반올림 후 앞 segment `end == 뒤 start` → STEP6
  `find_transcript_segment`가 경계포함(`seg_start≤ts≤seg_end`)+최초매칭이라 앞 segment가 먼저
  잡혀 원문 불일치 false reject(CLIP1 0단계 FAIL 4건).
- **현재 처리(해결됨)**: 어댑터가 앞 segment `end`를 다음 `start` 직전으로 깎음(step6_adapter.py:59-74,
  git 5cc482b). 발화 원문·시작 시각(검증 대상)은 불변, `end`(구간 판정용)만 조정.

### 2-5. STEP6→STEP7 어댑터 (step7_adapter.py, 참고)

STEP6 `verification`을 STEP7이 읽는 2필드로 매핑: `decision`→`verification.routing`,
`confidence(float)`→`verification.confidence.score`(step7_adapter.py:40-72). gate_a/b/c 점수는
참고용으로만 보존(STEP7 미독). 표준 라이브러리만 사용.

**근거 위치(§2):** `STEP5_.../step6_adapter.py:4-115,142-144`, `STEP6/rules.py:30-34,63-65,124-131`,
`graph.py:99-105,203-217`, `main.py:48,190`, `schemas.py`(미import), `step7_adapter.py:40-72`, git 5cc482b.

---

## §3. 게이트·채점 체계 전수 해부

### 3-1. 게이트 항목 전수표 (탈락조건 / 점수항목 구분)

| 단계 | 항목 | 유형 | 검사 내용 | 근거 위치 |
|---|---|---|---|---|
| 0 | timestamp_validity | **탈락조건(결정론)** | source·utterance 스텝 시각이 clip 범위 안 + transcript에 원문 실재. utterance만, action_only 면제 | rules.py:56-140 |
| 1 | Gate A search | (전처리) | 매뉴얼 top-k RAG. top1<0.45면 tacit_insight로 1회 재검색 | graph.py:95-134, embedding_utils.py:160-170 |
| 2 | Gate A relation | **탈락조건(LLM)** | same/delta/novel. **same → reject STOP**(매뉴얼 수준 후보 강등) | graph.py:140-168, llm_utils.py:125-228 |
| 3 | action_reason_consistency (Gate B) | 점수(LLM) | 행동↔이유 인과 일치도 0~1 | llm_utils.py:234-262 |
| 3 | reasoning_grounding (Gate C-1) | 점수(LLM) | reasoning이 실제 발화에서 도출 가능한가 0~1 | llm_utils.py:268-291 |
| 3 | step_grounding_ratio (Gate C-2) | 점수(LLM) | utterance 스텝 중 source_utterance가 action을 뒷받침하는 비율 | graph.py:211-228, llm_utils.py:297-314 |
| 3 | utterance_signal (Gate C-3) | 점수(규칙+LLM) | 근거 발화에 인과/이탈/주의/부정 마커 포함 비율(규칙) + LLM 보정 | rules.py:143-178, llm_utils.py:320-342 |
| 4 | confidence route | 라우팅 | 가중합 → accept/hold/reject | graph.py:247-307 |

### 3-2. Gate A (매뉴얼 대조) 상세 (사실)

- **유사도 임계값**: `manual_similarity_threshold = 0.45`(config.py:56). 1차 쿼리 top1 미만이면
  `tacit_insight`로 재검색해 더 높으면 교체(graph.py:117-128).
- **매뉴얼 문서**: `manuals/dell_precision_7920_manual.txt`(config `manual_dir`). **현재 실물은
  6,193바이트의 실질 발췌**("조립 후 부팅(POST) 실패 진단" 시나리오) — README는 "예시용 더미"라
  하지만 현재 파일은 내용이 채워져 있다(사실). Chroma in-memory, 매 실행 재색인(embedding_utils.py:108-123).
- **거리→유사도 변환**: Chroma cosine `distance` → `1 - distance`로 유사도 복원(embedding_utils.py:88).
  임계값 0.45는 cosine 유사도 기준(config.py:44-45).
- **"매뉴얼 수준 REJECT 강등"**: relation=`same`이면 confidence 계산 없이 `reject`(graph.py:160-163).
  판정 기준은 3차에 걸쳐 개정됨(§5-3).

### 3-3. confidence 산출 공식 (사실)

**발화 트랙**(silent 아님, graph.py:270-276):
```
confidence = 0.35·reasoning_grounding + 0.20·step_grounding_ratio
           + 0.15·action_reason_consistency + 0.30·utterance_signal   (weight_a)
```
**침묵 트랙**(silent, graph.py:263-268):
```
confidence = 0.60·step_grounding_ratio + 0.40·action_reason_consistency   (weight_a_silent)
```
- weight_a 값(config.py:63-68): rg=0.35, us=0.30, sgr=0.20, arc=0.15 (2026-07-10 개정:
  us 0.15→0.30, sgr·arc 재배분, config.py:61-62).
- **weight_a_silent 존재 이유(코드·주석에 명시, 사실)**: config.py:69-74 —
  > "발화 참조가 전혀 없는 후보(utterance step 0 AND reasoning_source 빈 배열)는 rg/us가 데이터
  > 부재로 0에 깔려 이론 상한 0.50 < T_high 0.70 — 구조적으로 accept 불가였다(run3 실측: 침묵 13건
  > 전원 rg=us=0, 최대 conf 0.420). … 침묵 암묵지가 이 프로젝트의 핵심 타깃인데 발화 전제 항목
  > 때문에 감점되는 모순 해소."

  침묵 트랙은 rg/us를 빼고 sgr·arc만 재정규화(합=1.0)해 만점 시 conf=1.0 가능.

### 3-4. case별 심사 분기 — 실제 구조 (사실, 요청 프레임 정정)

STEP6에는 both/action/utterance라는 **"case 필드 기반 분기 매트릭스가 없다"**(STEP5 `case` 미독,
§1-3). 대신 두 축으로 분기한다:

| 분기 축 | 판정 기준(데이터 유도) | 영향 |
|---|---|---|
| **스텝별 evidence** | `evidence=="utterance"` vs `"action_only"` | action_only는 timestamp 실재검사·step_grounding·utterance_signal 분모에서 **면제**(원칙 ④) |
| **silent 트랙** | (utterance 스텝 0개) **AND** (reasoning_source 빈 배열) | silent면 weight_a_silent(rg/us 제외), 아니면 weight_a(4종) |

- **추론:** 사용자가 말한 "case별 분기"는 실제로는 이 evidence/silent 이원 분기로 구현돼 있다. GT나
  STEP5 case 태그가 아니라 **후보 데이터 자체**로 트랙이 갈린다(graph.py:256-261 "트랙은 GT 라벨이
  아니라 후보 데이터로 갈린다").

### 3-5. 임계값·트랙 (사실)

| | T_high | T_low | 상태 |
|---|---|---|---|
| 트랙 A(휴리스틱, hold 넓게) | 0.70 | 0.40 | **현재 가동**(config.track="A") |
| 트랙 B(라벨 50건+ PR curve) | 0.725 | 0.391 | **미가동**(상수만 정의, 더미 시뮬 값) |

- 현재 코드에 **가동 중인 것은 트랙 A뿐**이다(config.py:59 `track="A"`, thresholds()/weights()가
  track으로 분기). 트랙 B는 weight_b/t_high_b/t_low_b 상수가 존재하나 `track="B"`로 바꿔야 활성화되고,
  **weight_b_silent는 아예 없다**(TODO, config.py:75-76,146-149) — 트랙 B 전환 시 라벨로 재산정 필요.
- **추론:** 즉 현재는 "라벨 0~20건 가정"의 고정 가중치·휴리스틱 임계값 체계이며, PR-curve 기반
  자동 임계값은 미구현 상태다.

### 3-6. LLM judge (사실)

- **모델**: `Qwen/Qwen3-14B`(config.py:34, 2026-07-12 Qwen2.5-14B에서 교체). `device_map="auto"`,
  **float16**(Turing bf16 불가, llm_utils.py:54-57), thinking 자동 차단(qwen3 감지 시
  `enable_thinking=False`, llm_utils.py:43-51). `max_new_tokens=800`, `temperature=0.0`.
- **프롬프트 구성**: 모든 judge에 공통 `HALLUCINATION_GUARD`(system) + 게이트별 user 프롬프트.
  각 프롬프트에 **들어가는 것 / 안 들어가는 것**:

| judge | 프롬프트에 들어가는 것 | 안 들어가는 것 |
|---|---|---|
| judge_manual_relation | tacit_insight, (reasoning 참고), 매뉴얼 top-k 발췌 | GT, decision, 다른 후보 |
| score_action_reason_consistency | situation, diagnostic_steps(action), reasoning, tacit_insight | 매뉴얼, transcript 원문, GT |
| score_reasoning_grounding | reasoning, reasoning_origin, reasoning_source 근처 transcript 발췌 | GT |
| judge_step_grounded | 스텝 action, source_utterance | GT, 다른 스텝 |
| score_utterance_signal | 규칙기반 마커 탐지 통계 + 스텝별 발화/마커 | GT |

- 출력은 전부 `{"score"/"relation"/"grounded", "justification"}` JSON. `_extract_json`이 코드펜스
  제거 후 최외곽 `{...}` 파싱, 실패 시 `{"parse_error":True}`(llm_utils.py:100-111).
  **추론:** parse_error 시 각 함수가 `setdefault`로 기본값(score 0.0/relation "novel"/grounded False)을
  넣으므로, **파싱 실패는 보수적(낮은 점수/novel/미접지)으로 흐른다** — 조용히 통과시키지 않음.

**근거 위치(§3):** `config.py:34-124,143-154`, `graph.py:95-307`, `rules.py:143-178`,
`llm_utils.py:43-342`, `embedding_utils.py:88,160-170`, `manuals/dell_precision_7920_manual.txt`.

---

## §4. 무결성 검증 — 판정에 정답지(GT) 무유입

### 4-1. GT 무유입 입증 (사실)

- **import·파일 접근 추적**: STEP6가 판정 실행 중 여는 파일은 (a)후보 JSON (b)transcript segment
  (c)`manuals/*.txt/.md` 셋뿐이다(main.py:42-85, embedding_utils.load_manual_documents:50-61).
  answer_key/정답지/GT 성격 파일을 여는 코드가 **없다**.
- **grep 결과(사실)**: STEP6 소스(.venv/pycache 제외)에서 `answer_key`/`ground_truth`/`정답`/`gold` 검색 →
  판정에 쓰이는 GT 데이터 참조 **0건**. 히트는 (1)`rules._check_one`의 `label` 파라미터(체크 항목명,
  GT 아님) (2)preflight의 루프 변수 `label` (3)`step7_adapter.py:48`의 주석 "gold_records 형식에
  가깝게"(주석일 뿐 GT 데이터 없음)뿐.
- **프롬프트 조립 경로**: §3-6 표대로 어떤 judge 프롬프트에도 GT/기대판정이 들어가지 않는다.
- **추론:** GT(정답지)는 이 리포에서 STEP6 **밖**의 채점(eval) 자료로만 존재하며(docs의 GT v2 등,
  git df21e4e), 판정 파이프라인은 GT를 import도 파일 접근도 하지 않는다 → **무결성 성립**.

### 4-2. 판정 재현성 (사실)

- LLM judge: `temperature=0.0`(config.py:38) → `do_sample = temperature>0 = False`(llm_utils.py:75) →
  **그리디 디코딩**. `temperature`는 `do_sample=False`일 때 `None`으로 전달(llm_utils.py:80).
- **추론:** 같은 입력·모델·하드웨어에서 그리디는 결정론적이므로 판정도 재현된다(부동소수점·라이브러리
  버전 차이는 예외). 단 **모델 교체는 판정 기준 자체의 변화**다 — config.py:32-33이 명시:
  "run6은 Qwen2.5 기준 산출물이므로 새 실행과 직접 비교하지 말 것". 실제로 judge 교체 실험에서
  분포가 크게 달라짐(§5-4).

### 4-3. 판정 사유 기록 (사실)

- `_verified.json`의 `verification` 블록(main.py:106-134): `decision`, `confidence`, `reject_reason`,
  `timestamp_validity.passed`, `gate_a_manual_comparison`(relation/justification/query/top_score/retry),
  `scores`(4종), `weight_track`, `weights_used`, `thresholds_used`, `track`.
- `_result.txt`: 0~4단계 전체 로그(각 게이트 justification 포함) + 최종 요약(main.py:88-103,192-196).
- **추론:** justification·점수·트랙이 전부 남으므로 전문가 검증(6-8)과 트랙 B 라벨 축적의 재료가 된다.
  다만 **라벨(전문가 판정) 자체를 수집·적재하는 코드는 STEP6에 없다** — verified.json이 재료일 뿐,
  라벨링·PR-curve 재산정 파이프라인은 미구현(§3-5, §7).

**근거 위치(§4):** `main.py:42-85,106-134`, `embedding_utils.py:50-61`, `llm_utils.py:66-83`,
`config.py:32-38`, grep(GT 0건), git df21e4e.

---

## §5. 원본 대비 수정 이력 (설계 역사)

> **방법(사실)**: 사용자 지시로 외부 원본(김보나 버전)은 대조하지 않았다. 아래는 **내 폴더 안의
> git log + 코드 주석의 날짜 annotation**만을 근거로 재구성한 것이다. 따라서 "원본 → 수정"의
> 정확한 diff가 아니라 **"주석·커밋에 기록된 변경"** 이다(원본 상태는 주석 진술로만 추정).

### 5-1. 주요 수정 5건 (사실 = 코드/주석/커밋, 추론 = 영향 해석)

| # | 수정 | 수정 전(주석 진술) → 후 | 왜(근거) | 판정분포 영향 |
|---|---|---|---|---|
| 1 | **case/트랙 분기 신설 (침묵 트랙)** | rg/us 포함 4종 가중치 단일 → 발화 무참조 후보는 sgr·arc만(weight_a_silent) | run3 침묵 13건 전원 conf≤0.420, accept 불가(config.py:69-74) | **추론:** 침묵 암묵지 accept 경로 개방 |
| 2 | **weight_a_silent 신설** | (없음) → `{sgr:0.60, arc:0.40}` | 위와 동일 세트(config.py:77-80), git af73e77 | **추론:** 무발화 후보 conf 상한 0.5→1.0 |
| 3 | **Gate A(매뉴얼 게이트) same 기준 3차 개정** | "주제·항목 존재만으로 same" → "구체 패턴·수치·방법의 새 정보 유무"로 판정 + reasoning 참고 입력 | run5/run3/run4 오판(llm_utils.py:127-151), git fe52e12 | **추론:** same 오탐(진짜 암묵지 사살)·미탐(잡것 생존) 양방향 진동 후 수렴 |
| 4 | **발화 마커·가중치 조정** | 문어체 마커(히트율 0.0) → 실발화 역추출 구어 마커, us 0.15→0.30 | motion_obs_run1 24건 전수 대조(config.py:98-124) | **추론:** ACCEPT 목표 후보 마커 히트율↑, "하면" 삭제로 접지 위장 오탐↓ |
| 5 | **모델 교체(judge)** | Qwen2.5-14B → Qwen3-14B(+thinking off) | STEP5·STEP8과 계열 통일(config.py:30-34), 2026-07-12 | **사실:** run7(qwen3) accept9/hold6/reject9 vs run6(qwen2.5) accept6/hold4/reject14 |
| (+) | 인프라: Qdrant→Chroma, dataclass→pydantic config, preflight 신설 | — | 팀 표준 통일(config.py:41-46), 조기 검증(config.py:7-11) | **추론:** 판정 로직 불변, 운영 안정성↑ |

- git 커밋(STEP6 관련): `fe52e12`(Gate A same 강화), `af73e77`(침묵 트랙), `5cced4f`(Chroma 전환).
  나머지는 코드 주석의 날짜 annotation으로만 남음(폴더 통합 시점 이전 이력이 단일 커밋에 묶임).
- 참조 문서(리포 내): `docs/report_packet/보고서_STEP5재설계_STEP6첫판정_20260708.md`,
  `docs/공유_STEP5-STEP6_스키마_어댑터.md`.

### 5-2. 실측 판정 분포 (사실 — 현존 산출물)

| 런 | judge | 분포(24건) |
|---|---|---|
| motion_obs_run6 | Qwen2.5(추정, config 주석 기준) | accept 6 / hold 4 / reject 14 |
| motion_obs_run7_qwen3judge | Qwen3 | accept 9 / hold 6 / reject 9 |
| motion_obs_run8_manualaug | (매뉴얼 증강) | accept 7 / hold 6 / reject 11 |
| motion_obs_run9_qwen25judge | Qwen2.5 | accept 5 / hold 4 / reject 15 |

- **사실:** judge 모델만 바꿔도 accept가 5→9로 벌어진다. (run6이 발표 기준런, run7~9는 미마감 실험 —
  섞어 비교 금지, 기존 메모리 [[project_step6_run6_baseline]].)

### 5-3. Gate A same 기준의 진동 (사실, 주목)

llm_utils.py:127-151 주석에 **같은 함수를 4번(07-09~07-10) 개정한 이력**이 그대로 남아있다:
07-09(패턴 보호) → 07-10 1차(same 회피 억제) → 2차(과잉교정으로 same 남발→진짜 암묵지 사살) →
3차(핀포인트 2건). **추론:** Gate A same은 이 파이프라인에서 **가장 손이 많이 간, 가장 불안정한
판정**이다 — LLM judge의 same/delta 경계가 프롬프트 문구에 민감하게 앵커링돼 양방향으로 진동했다.

### 5-4. CLI 평가 — 가장 취약한 고리 (추정, 근거 명시)

- **추정 ①(가장 취약): LLM judge 점수·판정의 분산.** §5-2에서 judge 모델 교체만으로 accept가
  6→9(+50%)로 흔들렸다. Gate A same은 프롬프트 문구에 민감해 4차 개정을 겪었다(§5-3). temperature=0
  으로 재현성은 있으나, **"어느 모델/프롬프트냐"가 판정 분포를 지배**한다 → 임계값·가중치 튜닝보다
  judge 자체가 결과를 좌우. 근거: run6/7/8/9 분포 실측, llm_utils.py:127-151 개정 이력.
- **추정 ②: 라벨 부재 상태의 휴리스틱 임계값.** 트랙 A의 T_high=0.70/T_low=0.40은 "hold 20%/60%/20%
  목표"의 **휴리스틱**이다(config.py:91). PR-curve 근거가 없어 accept/reject 경계가 원리적 최적점이
  아니다. 트랙 B는 미가동·라벨 미수집. 근거: config.py:59,91-96, §3-5.
- **추정 ③: 매뉴얼 커버리지 의존.** Gate A는 매뉴얼 발췌 1개 파일(6KB)에 의존한다. 매뉴얼이 얇으면
  novel이 과다 판정돼 same 강등이 무력화될 수 있다. 근거: manuals/ 단일 파일, run8_manualaug 실험 존재.
- **추정 ④: 침묵 트랙 2항목 가중합의 좁은 근거.** silent 후보 conf가 sgr·arc **단 2개**로 결정돼,
  둘 중 하나의 judge 오차가 그대로 accept/reject를 가른다(발화 트랙은 4개로 분산). 근거: graph.py:263-268.

**근거 위치(§5):** `config.py:30-124`, `llm_utils.py:127-151`, git fe52e12/af73e77/5cced4f/df21e4e,
`output/motion_obs_run6~9/`, `docs/report_packet/보고서_STEP5재설계_STEP6첫판정_20260708.md`.

---

## §6. 트러블슈팅 아카이브

### 6-1. 견본 개발 → 실물 첫 만남의 스키마 충돌 (어댑터 탄생)

- **증상**: STEP6가 견본 파일로 개발돼 STEP5 실물(candidates 배열, utterances 스키마)을 못 읽음.
- **원인**: 파일 구조·transcript 스키마 이중 불일치(§2-1).
- **해결**: "STEP6 코드 무수정" 원칙 하에 `step6_adapter.py`가 전량 흡수(파일 분리+필드 매핑+타입 변환).
- **남은 방어**: 어댑터 전수 대조 + write→read 왕복 검증(AssertionError로 즉사, step6_adapter.py:76-94).
- **재발 시**: STEP5 스키마 변경 시 §2-3 키 계약표 재점검(STEP6는 KeyError로 늦게 드러남 — schemas.py 미가동).

### 6-2. 무발화 후보 accept 불가 (침묵 트랙 신설)

- **증상**: run3 침묵 후보 13건 전원 conf≤0.420, accept 0건 — 구조적 불가.
- **원인**: rg/us가 발화 전제라 무발화 시 0에 깔림 → 이론 상한 0.50 < T_high 0.70(config.py:69-74).
- **실패한 시도(추론)**: 없음 — 원인이 "발화 전제 항목의 존재" 자체라 임계값·마커 튜닝으로 해결 불가.
- **해결**: 데이터로 silent 트랙 판정(utt 스텝 0 AND reasoning_source 빈 배열) → weight_a_silent
  (sgr·arc만, 재정규화)로 만점 시 conf=1.0(graph.py:256-268, git af73e77).
- **남은 방어**: 트랙을 GT가 아니라 후보 데이터로 가름(graph.py:256). weight_track을 verified.json에 기록.
- **재발 시**: silent 판정 AND 조건, weight_a_silent 합=1.0 확인.

### 6-3. 접지 위장 후보 통과 (Gate A + 마커 보강)

- **증상**: 발화를 인용하지만 결론과 무관한 후보(C1_001~003 류)가 conf를 밀어올려 통과.
- **원인**: (a)마커 `"하면 "`이 공백 제거 후 부분일치라 "세팅하면 됩니다" 같은 평범 발화에도 걸림
  (config.py:102-104). (b)Gate A same 기준이 느슨/과엄 사이 진동(§5-3).
- **해결**: (a)`"하면"` 마커 삭제(ACCEPT 목표는 "나면"/"먼저"로 잡힘, config.py:102-104).
  (b)Gate A same 3차 개정 — 구체 패턴·방법·비자명 위험만 새 정보로 인정(llm_utils.py:145-151).
  (c)STEP5 측 접지 내용검증(_ungrounded_utterance_claims)이 상류에서 위장 태깅 교정(STEP5 §3-4).
- **남은 방어**: 마커 화이트리스트 실발화 역추출 기준 유지, Gate A justification에 새 정보 지목 강제
  (llm_utils.py:188-191, 못 찾으면 same).
- **재발 시**: 마커 오탐(평범 발화 히트), Gate A justification이 실제 매뉴얼 밖 정보를 지목하는지.

### 6-4. 그 외 (git log·주석)

| 사건 | 내용 | 근거 |
|---|---|---|
| bf16 → fp16 | Turing(sm75) bf16 미지원 — judge 로딩 dtype 교체 | llm_utils.py:54-57 (2026-07-03) |
| dataclass → pydantic config | track/device 오타를 import 즉시 예외로(조용한 오작동 방지) | config.py:7-11,129-141 |
| preflight 신설 + 무거운 import 지연 | langgraph 최상단 import로 run 진입 전 사망(2026-07-03 사고) → preflight 후 import | main.py:36-39, preflight.py:8-15 |
| Qdrant → Chroma | STEP7/8 표준 통일, distance→(1-cos) 어댑터 | embedding_utils.py:64-89 (2026-07-14) |
| Qwen2.5 → Qwen3 judge | 계열 통일 + thinking off | config.py:30-34, llm_utils.py:43-51 (2026-07-12) |

**근거 위치(§6):** `step6_adapter.py:76-94`, `config.py:7-11,69-124`, `graph.py:256-268`,
`llm_utils.py:43-57,145-151`, `main.py:36-39`, git af73e77/fe52e12/5cced4f.

---

## §7. 재디벨롭 가이드

### 7-1. 재실행 커맨드

```bash
# STEP5 산출물을 STEP6 입력으로 변환(원본 무변경)
cd project-ai/STEP5_LLM_암묵지_후보생성
python3 step6_adapter.py --tacit-dir output/tacit_json --output output/step6_adapter_input/<tag>

# STEP6 판정 실행(자기 venv)
cd project-ai/STEP6_암묵지_품질검증
srun -p RTX6000 -w n5 --gres=gpu:1 .venv/bin/python3 main.py \
  --input ../STEP5_LLM_암묵지_후보생성/output/step6_adapter_input/<tag> \
  --output_dir output/<run_tag>
# 오프라인 구조 테스트(모델 로드 없이): python3 main.py --mock
# judge 격리 실험: STEP6_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct" python3 main.py ...
# 원샷(STEP3~7): 파이프라인_통합실행/main.py --judge-model ...

# 기대 판정표(GT) 대조 채점은 판정 '후' 별도 수행 — 판정 파이프라인에 GT를 넣지 말 것(§4 불변식).
```

### 7-2. 개선 후보

| 항목 | 내용 | 근거 |
|---|---|---|
| **트랙 B 전환** | 라벨 50건 축적 시: (1)라벨 수집 파이프라인 신설(현재 없음) (2)로지스틱 회귀로 weight_b **및 weight_b_silent**(현재 미정의) 재산정 (3)PR-curve로 t_high_b/t_low_b 재산출 (4)config.track="B" | config.py:75-76,146-149, §3-5 |
| **judge 안정화** | §5-4 최취약 고리 — judge 모델·프롬프트 고정, Gate A same 기준 회귀 테스트(run6~9 분포 추적) | §5-2~5-4 |
| **hold 폭 조정** | 트랙 A는 휴리스틱 — 전문가 라벨 확보 후 T_high/T_low 재조정. hold를 좁히면 accept/reject 오판↑ | config.py:91-93 |
| **매뉴얼 확충** | Gate A는 매뉴얼 커버리지에 의존. 확충 시 novel→same/delta 재분류로 판정 분포 변동 → 확충 후 재판정·재대조 필요 | run8_manualaug, §5-4 |
| **STEP6 schemas.py 가동** | 현재 미import — 입력 계약을 pydantic으로 강제하면 KeyError를 이른 스키마 예외로 전환 | §2-3 |

### 7-3. 건드리면 위험한 불변식

1. **탈락조건 / 점수항목 분리.** 점수항목(Gate B/C)을 탈락조건으로 바꾸거나 그 반대로 섞으면
   confidence 가중합의 의미가 붕괴한다. 탈락은 0/2단계 STOP, 점수는 4단계 가중합으로만(graph.py).
2. **action_only 보호.** 발화 전제 검사(timestamp 실재·step_grounding·utterance_signal)를 action_only에
   적용하면 침묵 암묵지가 전멸한다(rules.py:123-138, graph.py:213, 원칙 ④).
3. **GT 격리.** 판정 경로(main/graph/rules/llm_utils/embedding_utils)에 GT/기대판정을 import·파일
   접근·프롬프트 주입하지 말 것 — 무결성의 근거(§4).
4. **silent 트랙 판정 조건.** silent = (utt 스텝 0) AND (reasoning_source 빈 배열). OR로 바꾸면 발화
   일부 후보가 침묵 트랙으로 새 rg/us 감점을 피하는 우회로가 생긴다(graph.py:260).
5. **어댑터 검증 assert 유지.** transcript 변환 전수 대조를 끄면 발화 누락·시각 어긋남이 조용히
   false reject로 이어진다(step6_adapter.py:76-94).
6. **judge temperature=0 유지.** 올리면 재현성이 깨져 같은 후보가 실행마다 다르게 판정된다(§4-2).

**근거 위치(§7):** `config.py:75-93,146-149`, `graph.py:213,256-284`, `rules.py:123-138`,
`main.py`, `step6_adapter.py:76-94`, `llm_utils.py:75-80`.

---

## §8. "확인 불가" 항목 + §5 사실/추론 구분

### 8-1. 확인 불가 항목

1. **김보나 원본과의 정확한 diff** — 사용자 지시로 외부 원본을 대조하지 않았다. §5는 내 폴더 안
   git log·주석 annotation 기반이라 "원본 → 수정"의 라인 단위 diff가 아니다(주석 진술로만 추정).
2. **run6의 judge 모델** — config 주석은 "run6=Qwen2.5 기준"이라 하나(config.py:32-33), 현재 config는
   Qwen3다. run6 산출 시점의 실제 모델은 산출물에 기록돼 있지 않아 주석 진술로만 판단.
3. **트랙 B 실효성** — 코드에 상수만 있고 가동된 적이 없어(라벨 수집 파이프라인 부재) 실제 동작·성능
   확인 불가.
4. **판정 재현성의 하드웨어 독립성** — temperature=0 그리디는 결정론이나, GPU/라이브러리 버전에 따른
   부동소수점 차이가 판정을 뒤집는지는 코드로 확인 불가(실측 안 함).
5. **schemas.py의 의도** — 미import 상태가 "문서용"인지 "가동 예정이었으나 누락"인지 코드로 확정 불가.
6. **매뉴얼 파일의 권위성** — 현재 `dell_precision_7920_manual.txt`가 실제 Dell 매뉴얼 발췌인지
   팀 작성 요약인지 코드로 확인 불가(README는 "예시용 더미"라 하나 내용은 실질적).

### 8-2. §5 사실 / 추론 구분

**사실(코드·주석·커밋·산출물에서 직접 확인):**
- 수정 5건의 존재와 방향(침묵 트랙/weight_a_silent/Gate A 3차 개정/마커 조정/모델 교체) — config·llm_utils
  주석 + git af73e77/fe52e12/5cced4f.
- run6~9 판정 분포 수치(현존 verified.json 집계).
- Gate A same 함수의 4차 개정 이력(llm_utils.py:127-151 주석).
- 트랙 A만 가동, 트랙 B 상수만 존재, weight_b_silent 부재.

**추론/추정(내 해석):**
- 각 수정이 판정 분포에 미친 영향(§5-1 우측 열) — "추론:" 표기. 실측 대조는 부분적.
- §5-4 "가장 취약한 고리" 4개(judge 분산 / 휴리스틱 임계값 / 매뉴얼 의존 / 침묵 2항목) — 전부 "추정:"
  표기, 근거(런 분포·개정 이력·코드 구조)를 병기했으나 인과 확정은 아님.
- §1-3 STEP5↔STEP6 커플링이 "이중 필터 아님"이라는 판단 — 추론(STEP5가 후보를 안 버린다는 사실에 근거).

---

*(끝) 본 보고서는 조사·설명·평가 전용이며 코드를 수정하지 않았다.*
