# STEP7_DB · STEP8_RAG서비스 — 심층 분석 보고서

> 대상 코드: `project-ai/STEP7_DB/` (6개 .py, 총 590줄) + `project-ai/STEP8_RAG서비스/` (6개 .py, 총 657줄 + `static/index.html` 124줄) + 원본 참고 `voice-rag-upload/`(서시은).
> 분석 방법: 전체 소스 전수 정독 + 실제 산출물(`qdrant_db/` 컬렉션, `gold_records/*.json` 9건, `통합_작업메모.md` 실측 로그) 대조. `docs/면접대비_파이프라인_심층분석.md`와 동일 표기 규칙 — `파일명:줄번호`는 실제 소스 라인.
> 작성: 2026-07-05.

---

## 0. 한눈에 보는 시스템 개요

STEP7/8은 파이프라인의 **소비 측**이다. STEP3~5(추출)→STEP6(판정)이 만든 암묵지 최종 JSON(schema v1.3)을 받아, STEP7이 **벡터 DB로 적재+검색**을, STEP8이 그 위에 **음성 질의응답 웹서비스**를 얹는다.

```
STEP5 tacit_json (schema 1.3)
   └→ STEP6 품질검증 (accept/hold/reject 판정만, 내용 수정 없음)
        └→ [현재는 우회: 손으로 만든 gold_records 9건] ──┐
                                                          ▼
   STEP7_DB   : JSON → bge-m3 임베딩 → Qdrant(로컬 디스크) → CLI 검색 + [핵심/이유/절차/출처] 템플릿 조립
   STEP8_RAG  : 같은 JSON → 자체 Qdrant → FastAPI 서버
                브라우저 STT → POST /ask → 검색(임계값 필터) → Ollama LLM 답변 → 브라우저 TTS
```

두 폴더의 관계: **STEP7은 안나경의 CLI 파이프라인(방어 로직 강함, LLM 없음), STEP8은 서시은의 voice-rag-upload(웹서비스 뼈대, 방어 약함)를 "뼈대는 서시은 + 안전장치는 안나경" 원칙으로 통합한 것**이다(`docs/STEP8_RAG서비스_통합보고서.md` §1). STEP8이 사실상 STEP7의 후속 통합판이지만, STEP7은 독립 CLI로 여전히 동작하며 **벡터 DB도 각자 따로 갖고 있다**(§3-2에서 이 중복의 함의를 다룸).

핵심 설계 철학(코드 주석에 반복 등장):
- "환각 금지 — 관련도가 낮으면 지어내지 않고 '못 찾았다'고 답한다" (`search_tacit.py:8-9`, `app.py`의 이중 방어)
- "에러 없이 끝남 = 성공이라는 착각 방지 — 적재 후 개수까지 assert" (`build_db.py:43`, `ingest.py:170`)
- "무거운 모델 로드 전 5초 검증(preflight)" — STEP3/6과 동일 원칙을 양쪽 다 이식

---

## 1. STEP7_DB 심층 분석

### 1-1. 파일 구성과 실행 흐름

| 파일 | 줄수 | 역할 |
|---|---|---|
| `config.py` | 53 | pydantic Config — 값 오타 시 import 시점 즉사 |
| `preflight.py` | 116 | 4단계 사전 검증(패키지/환경/config/경로). 표준 라이브러리만 사용 |
| `vectordb_utils.py` | 160 | JSON 로드 → 임베딩 텍스트 조립 → Qdrant upsert → 검색 |
| `build_db.py` | 55 | 적재 진입점(+검색 스모크 assert) |
| `search_tacit.py` | 151 | 검색 + [핵심/이유/절차/출처] 답변 템플릿 조립(LLM 없음) |
| `query_tacit.py` | 55 | 질의 진입점 |
| `gold_records/` | 9건+2 | ⚠️ 손으로 만든 예시 데이터(§1-5) + 재생성 스크립트 |

실행 흐름: `build_db.py` → `run_preflight()` → bge-m3 로드(cpu) → `build_or_load_vectorstore()` → count assert → 검색 스모크. 질의는 `query_tacit.py "화면이 안 떠요"`.

### 1-2. 스위치·옵션 전수 목록

| # | 스위치 | 위치 | 기본값 | 작동 방식 |
|---|---|---|---|---|
| 1 | `input_dir` | `config.py:24` | `gold_records/` | 적재할 JSON 폴더. CLI `--input_dir`로 덮어씀(`build_db.py`). **실데이터 전환 시 바꿀 단 하나의 지점** |
| 2 | `embedding_model_name` | `config.py` | `BAAI/bge-m3` | STEP6과 동일 모델 재사용(스택 통일) |
| 3 | `device` | `config.py:28` | `"cpu"` | 문서 9건이라 GPU 불필요 — **로그인 노드에서 srun 없이 실행 가능**하게 한 의도적 결정. validator가 cuda/cpu 외 값 즉사 |
| 4 | `mock_mode` | `config.py` | `False` | `True`면 `MockEmbeddings`(`vectordb_utils.py:26`, sha256 기반 64차원 더미) — 오프라인 구조 테스트용 |
| 5 | `qdrant_path`/`qdrant_collection` | `config.py` | `qdrant_db/` / `tacit_knowledge` | `QdrantClient(path=...)` 로컬 모드(`vectordb_utils.py:134`) — 서버 프로세스 불필요, 단 **동시에 한 프로세스만 접근 가능**(.lock) |
| 6 | `top_k_default` | `config.py` | 3 | CLI `--top_k`로 덮어씀 |
| 7 | `min_similarity_threshold` | `config.py:41` | **0.40** | 이 미만이면 답변 조립 안 하고 NOT_FOUND(§1-4). 실측으로 정한 값(§1-4 임계값 사건) |
| 8 | `accept_only_default` / CLI `--accept_only` | `config.py:43`, `search_tacit.py:98` | `False` | routing==accept만 검색하는 Qdrant Filter 훅. **기본 꺼짐** — 실데이터·실제 STEP6 판정이 나오기 전까지 보류(D5) |
| 9 | `--query` | `build_db.py` | "래치 두 개를 동시에 눌러야 안전하다" | 적재 직후 스모크 검색 쿼리 |

조건부 실행 경로: 스키마 안 맞는 JSON은 건너뛰고 경고만(`vectordb_utils.py` `load_candidate_documents` — "관용적 실패"), timestamp 범위 밖은 경고만 하고 폐기 안 함(`search_tacit.py:48` — "검색 결과라 후보를 폐기할 권한이 없다"는 주석).

### 1-3. 임베딩 설계 결정(D1) — 무엇을 벡터로 만드나

**결정: `[상황] situation \n[노하우] tacit_insight \n[이유] reasoning` 합본을 임베딩**(`vectordb_utils.py:112`).

근거(실측): `tacit_insight` 단독 임베딩이었을 때 신입의 "증상 언어" 질의("화면이 안 떠요")와 유사도가 **0.27대**로 낮았다. situation이 증상 질의와 붙는 접점 역할을 하도록 합본으로 바꾸자 top1 유사도가 **0.44~0.61대**로 상승(README). `diagnostic_steps.action`은 "절차 나열이라 노이즈"로 판단해 임베딩엔 안 넣고 payload로만 보관(답변 조립용).

멱등 적재: 암묵지 `id`를 고정 네임스페이스 UUID5로 변환해 Qdrant point ID로 사용(`vectordb_utils.py:23,60`) → `build_db.py`를 몇 번 돌려도 같은 id는 덮어써져 중복이 안 쌓인다.

### 1-4. 검색과 답변 조립 (D3/D4/D5)

- **답변 템플릿**(`search_tacit.py:73` `format_answer`): `[유사도 x.xxx]` / 핵심(tacit_insight) / 이유(reasoning) / 절차(diagnostic_steps.action 순서대로) / 출처(video_id + clip 구간) / (있으면) `⚠ 초보자 함정`("초보자"가 들어간 문장 추출, `search_tacit.py:39`). **LLM 재작성은 의도적으로 안 함** — "이번 단계는 템플릿 조립까지만"(코딩지시 4절), 자연어화는 STEP8의 몫.
- **임계값 사건(중요)**: 처음 0.30으로 뒀더니 완전 무관한 질의("오늘 점심 뭐 먹지")도 bge-m3 특성상 baseline 유사도가 **0.30~0.36대**로 나와 안 걸러졌다. 실측 결과 무관 질의군(0.30~0.36)과 관련 질의군(0.44+) 사이 뚜렷한 갭 확인 → **0.40으로 상향**(`config.py:37-41` 주석). 이 값이 이후 STEP8의 `SIM_THRESHOLD` 기본값으로 그대로 이식됐고, 2026-07-04 팀 결정으로 확정됨.
- **timestamp 검증(D4)**: step timestamp가 clip_start~clip_end 밖이면 경고(STEP6 `timestamp_validity`와 같은 규칙, 단 폐기 권한 없음).
- **accept 필터(D5)**: langchain_qdrant는 payload를 `{"metadata": {...}}` 로 중첩 저장하므로 필터 키가 `metadata.routing`(`search_tacit.py:98`) — STEP8의 플랫 payload(`routing`)와 **필터 키가 다르다**(§3-1 비교표).

### 1-5. gold_records의 정체 — 정직하게 기록된 캐비앗 (필독)

`gold_records/README.txt`가 이 프로젝트에서 가장 정직한 문서다. 요지:

1. **이 9건은 실제 파이프라인 산출물이 아니라 손으로 만든 예시다.** 당시 STEP4 VLM 0건 버그로 실데이터를 못 뽑아서, 촬영 시 만든 암묵지 정답지 9행을 스키마 1.3 형태로 수작업 매핑했다. verification 블록도 "STEP6을 돌리면 이렇게 나올 것"이라는 **추정치**다.
2. **"action_only 보호" 정정 사건**: 예전 버전은 침묵 동작(action_only) 후보를 발화 전제 검사에서 면제해 accept(0.87~0.88)로 밀어줬는데, 실제 STEP6 코드(graph.py/rules.py/llm_utils.py)를 정독해보니 그런 면제 로직이 없었다. 실제 코드 기준으로 재계산하자 **action_only 2건(001, 002)이 accept→hold(0.51)로 강등**됐다. 잘못된 가정을 발견하고 데이터를 실제 코드 기준으로 정정한 사례.
3. 현재 분포: **7건 accept, 2건 hold** (001 채널 사전판독, 002 커넥터 촉각 확인 — 공교롭게도 둘 다 '침묵 암묵지'라 이 프로젝트가 가장 잡고 싶어 하는 유형이 hold로 밀려 있음).

**함의**: STEP7/8의 검색 품질 수치(Recall 9/9 등)는 전부 이 수작업 예시 기준이다. 실데이터(STEP4/5 수정 후 산출물)로 교체되면 재측정이 필수다.

### 1-6. 허점·리스크 전수

| # | 허점 | 위치 | 영향 |
|---|---|---|---|
| 1 | **`query_tacit.py`가 질의 때마다 전체 재적재를 한다.** 검색만 하면 되는데 `build_or_load_vectorstore()`(`query_tacit.py:37`)를 불러 매번 9건 재임베딩+upsert. 멱등이라 데이터는 안 망가지지만, 문서가 수백 건이 되면 질의 한 번에 전량 재임베딩 | `query_tacit.py:37` | 성능(현재는 9건이라 무해) |
| 2 | `search_tacit()` 함수 시그니처의 `min_similarity=0.30` 기본값이 config(0.40)과 불일치 — 호출부는 항상 CONFIG를 넘겨서 실사용 문제는 없지만, 함수만 따로 쓰면 옛 임계값이 살아난다 | `search_tacit.py:113` | 잠복 버그 |
| 3 | 유사도 변환식 `score if score <= 1.0 else 1.0/(1.0+score)`(`vectordb_utils.py:158`) — cosine 유사도와 거리를 휴리스틱으로 겸용 처리. langchain_qdrant 버전이 바뀌어 반환 의미가 바뀌면 임계값 0.40의 의미도 조용히 바뀐다 | `vectordb_utils.py:158` | 버전 취약성 |
| 4 | Qdrant 로컬 모드 `.lock` — STEP7 CLI와 STEP8 서버가 **같은 DB를 못 쓴다**(그래서 STEP8이 자체 qdrant_db를 복제 운영). 실서비스에선 Qdrant 서버 모드 전환 필요 | `vectordb_utils.py:134` | 배포 구조 |
| 5 | `⚠ 초보자 함정` 추출이 "초보자" 문자열 포함 여부라는 초단순 규칙(`search_tacit.py:39`) — insight 문구가 바뀌면 조용히 사라짐 | `search_tacit.py:39` | 견고성 낮음(문서화된 의도적 간이 구현) |

---

## 2. STEP8_RAG서비스 심층 분석

### 2-1. 통합 배경과 구조

같은 목표(암묵지 RAG)를 두 팀원이 독립 구현한 걸 합쳤다: **뼈대(FastAPI+Ollama+환경변수 스위치+Recall 평가)는 서시은 voice-rag-upload, 안전장치(preflight+pydantic+assert+timestamp 검증+accept 필터)는 안나경 STEP7**. 입력 데이터는 복제하지 않고 `../STEP7_DB/gold_records`를 참조(`config.py`). 원본 `voice-rag-upload/`는 그대로 남아 있으며 ChromaDB 잔재(`knowledge_db/`, `inspect_db.py`)는 죽은 코드로 판정해 안 가져왔다.

서비스 파이프라인(음성→음성): 브라우저 Web Speech API STT(`static/index.html:107`, Chrome 전용 `webkitSpeechRecognition`) → `POST /ask` → `search.search()`(임베딩+Qdrant+임계값) → Ollama LLM(`qwen2.5:14b-instruct`) → 브라우저 SpeechSynthesis TTS(출처 줄은 음성에서 제외, `index.html:56`).

### 2-2. 스위치·옵션 전수 목록 (환경변수 중심 — 서시은 방식 유지)

| # | 스위치 | 위치 | 기본값 | 작동 방식 |
|---|---|---|---|---|
| 1 | `EMBED_MODE` | `config.py:29` | `full` | `full`(작업+상황+노하우+이유+**절차+키워드**, `ingest.py:67`) vs `insight`(단독). 컬렉션명도 `tacit_knowledge_{mode}`로 자동 분리(`config.py:84`) — 비교 실험 구조 |
| 2 | `COLLECTION` | `config.py` | ""(자동) | embed_mode 기반 자동 결정을 강제로 덮는 용도 |
| 3 | `TOP_K` | `config.py` | 3 | |
| 4 | `SIM_THRESHOLD` | `config.py:41` | **0.40** | 서시은 원본 0.35 → STEP7 실측값 0.40으로 상향(§2-4 사건) |
| 5 | `ACCEPT_ONLY` | `config.py:43` | `false` | routing==accept 필터. payload가 플랫이라 필터 키가 `routing`(`search.py:40`) — STEP7(`metadata.routing`)과 다름 |
| 6 | `LLM_MODEL` | `config.py:48` | `qwen2.5:14b-instruct` | 원본 기본값 3b는 이 환경에 미설치, 이미 pull된 14b로 변경 |
| 7 | `OLLAMA_URL` | `config.py` | `http://localhost:11434/api/chat` | GPU 노드에서 `OLLAMA_HOST=0.0.0.0`으로 띄워야 접근 가능(§2-5) |
| 8 | CLI `--input`/`--db-path`/`--embed-mode` | `ingest.py` | config 위임 | |
| 9 | `evaluate.py --collection/--top-k` | `evaluate.py` | config 위임 | 평가 시 임계값을 **0.0으로 풀고**(`evaluate.py:51` `model_copy`) 순수 랭킹만 측정 — 임계값과 랭킹 품질을 분리해서 재는 올바른 설계 |

pydantic validator 4종(device/embed_mode/threshold 0~1/top_k≥1)이 환경변수 오타를 import 시점에 잡는다 — "환경변수 편의성 + 타입 안전"의 결합이 이 폴더의 특징.

### 2-3. 적재·검색·서빙 3부 구조

- **ingest.py**: STEP7과 달리 **컬렉션을 지우고 새로 만든다**(`ingest.py:135` `delete_collection`) — STEP7의 멱등 upsert와 다른 의미론(전량 리빌드). 원본 JSON 전체를 `raw_json`으로 payload에 보존(`ingest.py:159`)해 LLM 컨텍스트로 재사용 — 서시은의 핵심 아이디어. routing/confidence는 raw_json 안에 묻히면 Qdrant Filter로 못 거르므로 별도 최상위 키로도 저장(`ingest.py:156`).
- **search.py**: 검색 로직을 app/evaluate가 공유하는 함수로 분리. 임계값 미만은 여기서 거르고 전부 걸러지면 **빈 리스트 반환**(`search.py:63`) — LLM 호출 여부 판단을 호출부에 넘기는 계약.
- **app.py**: 모듈 최상단에서 `run_preflight()`(`app.py:38`) 후 모델 로드(uvicorn이 `app:app`을 찾는 구조상 여기밖에 없음). `/ask`가 이중 방어를 구현.

### 2-4. 시스템 프롬프트 원문과 이중 방어

**SYSTEM_PROMPT** (`app.py:43-49` 원문 그대로):

```text
너는 조립 현장의 신입 작업자를 돕는 음성 작업 보조 AI다.
아래 [검색된 암묵지]만 근거로 답하고, 근거가 부족하면 솔직히 부족하다고 말하라.
규칙:
- 첫 1~2문장에 핵심 답을 먼저 말한다 (음성으로 읽힐 것을 가정, 간결한 존댓말)
- 절차가 있으면 번호 단계로 나열
- 마지막 줄에 "출처: 영상ID (시작~끝 타임스탬프)" 표기
- 마크다운 강조 기호 사용 금지, 일반 텍스트로만
```

**이중 방어(환각 금지)**:
1. **방어선1(구조적)**: 검색 결과가 전부 임계값 미만이면 `/ask`가 **LLM을 아예 호출하지 않고**(`app.py:74`) "관련 암묵지를 찾지 못했어요" 즉시 반환. 검색 실패 상태를 LLM에 넘기면 그럴듯한 말을 지어낼 위험 자체를 차단.
2. **방어선2(프롬프트)**: 그래도 통과한 애매한 컨텍스트를 위해 시스템 프롬프트로 이중 강제.

**임계값 0.35→0.40 사건(방어선이 실제로 뚫렸던 기록)**: 서시은 원본 0.35에서 "오늘 점심 뭐 먹지" top1이 **0.372로 방어선1을 통과**했고, 방어선2(프롬프트)가 겨우 막았다 — "운 좋게 막은 것이지 구조적 보장이 아니다"(통합보고서 §3-1). STEP7이 실측해둔 0.40으로 올린 뒤 같은 질의가 **방어선1에서 0.47초 만에(LLM 호출 없이) 차단**됨을 재검증. 두 프로젝트가 독립적으로 같은 함정(느슨한 임계값)에 빠졌다가 같은 방법(무관/관련 질의군 유사도 분포 갭 실측)으로 고쳐진 사례.

### 2-5. 실측 결과 요약 (통합_작업메모.md 원자료 기준)

| 테스트 | 결과 |
|---|---|
| 적재 9건 | assert 통과. 가짜 불량 JSON 섞어도 건너뛰고 정상 적재(관용적 실패 검증됨) |
| Recall@3, embed_mode=**full** | **9/9 (1.00)** |
| Recall@3, embed_mode=insight | 8/9 — latch_confirm↔latch_removal 혼동 1건. **"situation을 붙여야 증상 언어와 붙는다"는 STEP7 실측 결론이 다른 스택에서 재현됨** |
| 정상 질의 → LLM | 근거만 사용, 형식 준수, **GPU 14초** (CPU였을 땐 2분+ 타임아웃) |
| 무관 질의 | 방어선1 즉시 차단(0.16~0.47초, LLM 미호출) |

환경 사건 2건: ① 이 계정에 sudo 없이 설치된 Ollama(`~/.local/ollama`)와 14b 모델이 이미 존재했음(서시은 선행 테스트 흔적) — 기본 모델을 그에 맞춤. ② Ollama를 로그인 노드(CPU)에서 띄우면 14b 추론 2분+로 비현실적 → **팀 공용 GPU 잡에 `srun --overlap`으로 편승** + `OLLAMA_HOST=0.0.0.0` 바인딩 필요.

### 2-6. 허점·리스크 전수

| # | 허점 | 위치 | 영향 |
|---|---|---|---|
| 1 | **임베딩 텍스트가 STEP7과 불일치**: STEP7은 diagnostic_steps를 "노이즈"라며 뺐는데(§1-3), STEP8 full 모드는 절차+키워드까지 **포함**(`ingest.py:67`)하고 Recall 9/9가 나왔다. 서로 반대 결정인데 양쪽 다 "실측으로 검증됨"으로 기록돼 있음 — 데이터 9건 규모에선 둘 다 만점이라 분별이 안 된 것. 실데이터 규모에서 재실험해 하나로 통일해야 함 | `ingest.py:67` vs `vectordb_utils.py:112` | 설계 불일치(잠복) |
| 2 | Ollama가 팀 공용 GPU 잡에 임시 편승 — 그 잡이 죽으면 서비스도 죽음. 전용 할당 필요 | 운영 | 가용성 |
| 3 | `/ask`에 인증·rate limit 없음, Ollama 호출 timeout 120s 고정(`app.py:92`) — 데모용으로는 충분하나 서비스화 시 보강 필요 | `app.py` | 보안/견고성 |
| 4 | 브라우저 STT가 `webkitSpeechRecognition` 의존(`index.html:107`) — Chrome 계열 전용, 사내 표준 브라우저 확인 필요 | UI | 호환성 |
| 5 | ingest가 delete→rebuild 방식이라 적재 중 서버가 검색하면 일시적으로 컬렉션이 빈다(STEP7의 upsert와 달리). 현재는 CLI 수동 운용이라 무해 | `ingest.py:135` | 운영 절차 |
| 6 | STEP7과 마찬가지로 **모든 품질 수치가 gold_records 9건(수작업 예시) 기준** | 데이터 | §1-5와 동일 |

---

## 3. STEP7 vs STEP8 종합 비교

### 3-1. 결정 사항 대조표

| 항목 | STEP7_DB | STEP8_RAG서비스 | 비고 |
|---|---|---|---|
| 형태 | CLI (적재+검색) | FastAPI 웹서비스 (음성 in/out) | STEP8이 통합 후속판 |
| 임베딩 모델 | bge-m3 (cpu) | 동일 | STEP6과도 동일 — 스택 통일 |
| 임베딩 텍스트 | 상황+노하우+이유 (**절차 제외**) | full: 작업+상황+노하우+이유+**절차+키워드** | ⚠️ 서로 반대 결정(§2-6 #1) |
| Qdrant 적재 | UUID5 멱등 upsert | delete→rebuild | 의미론 다름 |
| payload 구조 | langchain_qdrant 중첩(`metadata.*`) | 플랫 + `raw_json` 통짜 보존 | accept 필터 키도 달라짐 |
| 임계값 | 0.40 (실측 원조) | 0.40 (이식받음) | 팀 확정값 |
| 답변 생성 | 템플릿 조립(LLM 없음) | Ollama 14b 자연어 + 이중 방어 | 역할 분담 |
| 평가 | 스모크 assert | Recall@k 세트 9문항 | STEP8이 더 체계적 |
| DB 위치 | `STEP7_DB/qdrant_db` | `STEP8_RAG서비스/qdrant_db` | **중복 — 원인은 로컬 모드 .lock(§1-6 #4)** |

### 3-2. 구조적 평가

- **잘된 것**: 방어 원칙(preflight/assert/pydantic/이중 환각 방어)이 두 폴더에 일관 이식됐고, 결정마다 실측 근거가 주석·메모로 남아 있다. 특히 임계값 0.40은 "감이 아니라 무관/관련 질의군 분포 갭"으로 정해진 모범 사례.
- **정리 필요**: ① 임베딩 텍스트 정책 통일(실데이터 규모에서 A/B), ② DB 이원화 해소(장기적으로 Qdrant 서버 모드 + 컬렉션 하나), ③ STEP7의 질의-시-재적재 제거(#1-6 1).

---

## 4. 앞단(STEP4/5 수정·tacit2)과의 연결 — 실데이터 전환 체크리스트

2026-07-05 STEP4/5 구조 수정(YOLO 조용한 실패 제거, VLM 청크 관찰, 중복 압축)과 tacit2(1) 재작성으로 **실제 파이프라인 산출물이 나오기 시작했다.** gold_records를 실데이터로 교체하려면:

1. **STEP5(또는 tacit2 S5) tacit_json → STEP6 실행** → verification 블록이 붙은 최종 JSON 확보 (gold_records의 verification은 추정치였음 — 실행 결과로 대체)
2. `STEP7_DB/config.py:24`의 `input_dir`만 실데이터 폴더로 변경(코드 변경 불필요하게 설계돼 있음), STEP8은 `ingest.py --input`
3. **스키마 차이 주의**: tacit2 산출물은 diagnostic_steps에 `source_hint` 필드(evidence는 코드 계산)를 쓴다 — STEP7/8의 로더는 `id`/`knowledge`/`metadata.task`를 필수로 요구하므로(`ingest.py` load_entries), tacit2 최종 스키마와 v1.3 호환 여부 확인 필요. 특히 tacit2 쪽 metadata.task가 null이면 `ingest.py`가 그 파일을 건너뛴다(KeyError는 아니고 값 null은 통과되지만 payload.task가 null로 적재됨 — 검색 UI 표기 깨짐)
4. 실데이터로 **Recall 세트 재작성 + 임계값 0.40 재검증**(문서 수가 늘면 유사도 분포가 달라질 수 있음), embed_mode A/B 재실험(§2-6 #1)
5. `ACCEPT_ONLY=true` 켜기 — 실제 STEP6 routing이 생기므로 D5 훅이 드디어 의미를 가짐. 단 침묵 암묵지 2건이 hold로 몰리는 현상(§1-5)이 실데이터에서도 재현되면 STEP6 임계값/가중치 재논의 필요

---

## 5. 예상 질문 대비 포인트

- **"왜 벡터 DB가 두 개인가?"** — Qdrant 로컬 모드의 단일 프로세스 락 때문에 CLI(STEP7)와 서버(STEP8)가 한 DB를 공유할 수 없었다. 서비스화 시 서버 모드로 통합 예정.
- **"임계값 0.40의 근거는?"** — 무관 질의군 baseline(0.30~0.36)과 관련 질의군(0.44+)의 실측 분포 갭. 두 팀원이 독립적으로 같은 함정에 빠졌다가 같은 방법으로 수렴한 값.
- **"환각을 어떻게 막나?"** — 구조적 차단(임계값 미달 시 LLM 미호출)이 1차, 프롬프트("근거만, 부족하면 부족하다고")가 2차. 1차가 뚫렸던 실사례(0.372)와 그 수정 기록이 있음.
- **"Recall 9/9면 완성 아닌가?"** — 아니다. 데이터가 수작업 예시 9건이고 질문 세트도 그에 맞춰 작성됐다. 실데이터 규모에서 재측정 전까지는 "파이프라인이 동작한다"는 증명이지 "품질이 좋다"는 증명이 아니다.
- **"LLM 재작성을 STEP7에서 안 한 이유?"** — 역할 분리(검색·조립 vs 자연어화)와 검증 용이성. 템플릿 조립은 환각이 원천 불가능하므로, 환각 리스크는 STEP8 한 곳에 모아 이중 방어로 관리.
