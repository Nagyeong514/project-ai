# STEP8(RAG — 암묵지 검색·답변) 상세 기술 보고서

> 목적 (앞 보고서들과 동일)
> 1. **재디벨롭 참고 문서** — 검색·답변·음성·웹앱 계약을 바로 찾는다.
> 2. **트러블슈팅 아카이브** — 유사도 0.27·임계값·계약 사건 기록.
>
> **작성 규칙(엄수):** 코드/로그를 실제로 따라가 확인된 것만. 사실/추론("추정:") 구분, 확인 불가는
> "**(확인 불가)**" 후 §9 재수집. 모든 주장에 근거 위치. **코드 수정 없음.** 기준 2026-07-16, 커밋 `45be634`.

> **중심축(사용자 시나리오):** 신입이 증상 언어("화면이 안 떠요")로 물으면 → 명장의 암묵지를
> **핵심/이유/절차/출처**로 되돌려준다. 이 단계는 **파이프라인의 출구이자 유일하게 최종 사용자를
> 직접 만나는 단계**다(음성 웹앱).

> **적재(색인)와의 관계(중요):** STEP7 보고서(`docs/STEP7_상세보고서.md`)의 적재는 `STEP7_DB/`의
> `build_db.py`다. **STEP8은 그 DB를 재사용하지 않고 자체 `ingest.py`로 별도 DB를 만든다** —
> 소스 JSON은 같지만 **임베딩 텍스트·payload 스키마가 다르다**(§2). 그래서 색인 일반론은 STEP7을
> 참조하되, **검색 품질에 직결되는 임베딩 텍스트 설계는 여기(§2)에서 STEP8 코드 기준으로 다룬다.**

---

## §1. 전체 구조 개관

### 1-1. 순서도 (음성 입력~출력 전 구간)

```
[브라우저] static/메인화면.html (음성 UI: orb·자막·마이크)
   │
   ├─ (음성) 녹음 webm → POST /transcribe ─────────────────────────────┐
   │     app.transcribe(): faster-whisper large-v3-turbo(ko, beam_size=1, vad_filter)
   │     → 질의 텍스트 → ask() 재사용                                     │
   │                                                                      ▼
   └─ (텍스트) POST /ask {question} ──────────────────────► app.ask()  (app.py:109-162)
         ① 질문 임베딩: bge-m3(CPU, normalize)          se.search (search.py:52-96)
         ② Vector DB 검색: Chroma(top_k=3, cosine)       (routing 필터는 accept_only일 때만)
         ③ 유사도 임계값 필터: score ≥ 0.40 만 남김        distance→(1-distance) 유사도 복원
              │
              ├─ [방어선 1] 통과 결과 0건 → LLM 호출 안 함, 정직 실패 응답 즉시 반환(app.py:117-121)
              ▼
         ④ LLM 답변 생성: Ollama qwen3:14b (think:false, num_predict 256, keep_alive -1)
              [방어선 2] SYSTEM_PROMPT "검색된 근거만, 부족하면 부족하다"(app.py:47-53)
              → 답변(핵심→절차→"출처: 영상ID (시작~끝)")
              ▼
         반환 {answer, sources:[{id,similarity,routing,task,insight,video,clip}]}
   │
   └─ (음성 출력) POST /tts {text} → Supertonic-3(F1, steps=5) → wav → 브라우저 재생
        (발음사전 normalize_tts_text: RAM→램, BIOS→바이오스 …, app.py:57-75)
```

- STT/LLM/TTS 무거운 모델은 **preflight 통과 후** 로드(app.py:40-94, STEP3/6/7 동일 원칙).
- **협업 구조(사실)**: 서시은(voice-rag v2, 음성 계층·프론트) + 안나경(검색·DB·방어) 통합
  (app.py:3, 통합_작업메모.md 파트4). `voice-rag-upload/`는 흡수 후 아카이브.

### 1-2. config/상수 전수 (config.py, 환경변수 오버라이드)

| 키(env) | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `TOP_K`(top_k_default) | `3` | 검색 결과 수 | 올리면 컨텍스트↑·노이즈↑(§3-2) |
| `SIM_THRESHOLD` | `0.40` | cosine 유사도 컷오프 | 낮추면 무관 질의 통과(§7-2), 높이면 정답 놓침 |
| `EMBED_MODE` | `full` | `full`(합본) / `insight`(단독) — A/B 실험(§2) | insight면 증상 질의 유사도↓(0.27 사건) |
| `ACCEPT_ONLY`(accept_only_default) | `false` | routing=="accept"만 검색 | true면 hold/reject 제외(단 태그 부재 시 0건 위험, §3-3) |
| `COLLECTION`(qdrant_collection) | ""→`tacit_knowledge_full` | 컬렉션명(embed_mode로 자동) | insight 모드면 `_insight` |
| `LLM_MODEL` | `qwen3:14b` | Ollama 답변 모델 | 롤백 `qwen2.5:14b-instruct`(think 미전송) |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama 엔드포인트 | 노드 바뀌면 수정(운영 시 n5 등) |
| `VECTOR_BACKEND` | `chroma` | chroma / qdrant(롤백) | §3 |
| `STT_MODEL/DEVICE` | `large-v3-turbo`/`auto` | 질의 STT | §5-3 |
| `TTS_MODEL/VOICE/STEPS` | `supertonic-3`/`F1`/`5` | 음성 합성 | steps↓ 빠름·품질 미미차 |

### 1-3. 배포 형태 (사실)

- **Vector DB = Chroma `PersistentClient`(디스크 영속)**(search.py:33). 서버 프로세스 불필요. 롤백은
  `VECTOR_BACKEND=qdrant`(embedded, `QdrantClient(path=...)`, search.py:37). **서버 전환 시 바꿀 지점**:
  Qdrant 서버 모드로 가려면 `QdrantClient(path=)` → `QdrantClient(url="http://host:6333")`로 클라이언트
  인자 교체 필요(현재 코드엔 path=만, 서버 인자 경로 없음 → §9 확인 불가/미구현).
- **웹서버 = FastAPI + uvicorn**(app.py:102, 서버시작.sh 포트 8001 — 8000 타인 점유). 상주 서비스라
  배치 오케스트레이터(main.py)에서 제외(STEP6 §main 참고).
- **LLM = Ollama 별도 프로세스**(qwen3:14b, 공유 경로 `/home/ai_user/team_a2/.ollama/models`, config.py:56-65).

**근거 위치(§1):** `app.py:40-162,205-211`, `config.py:22-74,119-123`, `search.py:23-38`,
`통합_작업메모.md` 파트4, `서버시작.sh`.

---

## §2. 임베딩 텍스트 설계 (이 단계 최대의 설계 사연)

### 2-1. STEP8의 실제 임베딩 텍스트 (사실, ingest.py:67-78 `build_document`)

```python
if mode == "insight":
    return k["tacit_insight"]                       # 단독(A/B 비교 arm)
steps = " / ".join(s["action"] for s in k.get("diagnostic_steps", []))
return (f"작업: {m['task']}. 상황: {k['situation']}. "
        f"노하우: {k['tacit_insight']} 이유: {k.get('reasoning','')} "
        f"절차: {steps}. 키워드: {', '.join(m.get('keywords', []))}")   # full(기본)
```

- **`full` 모드(현재 기본)에는 `diagnostic_steps.action`(절차)과 `keywords`가 포함**된다.
- **⚠️ STEP7과 다르다(사실, 중요)**: STEP7 `vectordb_utils.py:112`의 임베딩 텍스트는
  `"[상황]{situation}\n[노하우]{tacit_insight}\n[이유]{reasoning}"` — **steps 제외**. STEP8 `full`은
  **steps·keywords·task 포함** + 형식도 다름(`작업:/상황:/노하우:/이유:/절차:/키워드:`).
  즉 사용자 프롬프트가 전제한 "steps 제외, situation+insight+reasoning 합본"은 **STEP7 설계**이고,
  **STEP8 서비스는 steps를 포함하는 별도 설계**다. 두 서비스가 임베딩 텍스트를 다르게 만든다.
  - **추정:** STEP8은 서시은 voice-rag의 "full=상황+노하우+이유+절차+키워드" 원형을 유지했고
    (config.py:32 주석과 일치), STEP7은 "steps=노이즈"로 판단해 뺐다 — 두 갈래가 통합되지 않고
    각자의 설계로 병존. steps 포함이 검색에 득인지 실인지는 코드로 확정 불가(§9).

### 2-2. payload (사실, ingest.py:130-143)

flat 필드 + 원본 통짜: `doc_id`, `document`(임베딩된 텍스트), `task`, `video_id`, `clip_start/end`,
`routing`, `confidence`, `raw_json`(entry 전체 — LLM 컨텍스트·답변 조립용). Chroma 스칼라 제약상
None 값 키는 제외(ingest.py:163). diagnostic_steps는 `raw_json` 안에 보존(그리고 full 모드에선
임베딩에도 포함).

### 2-3. 0.27 사연 (사실 — 단, 기록은 STEP7에 있음)

- **사연**: `tacit_insight` 단독 임베딩 시 신입의 증상 질의와 top1 유사도가 **0.27대**로 낮았음
  (STEP7 `vectordb_utils.py:9,109`, `STEP7_DB/README.md:23`). → **합본으로 교체 + 재적재**로 해결.
- **STEP8에서의 실체**: 그 "단독 vs 합본" 대립이 STEP8의 `embed_mode`(insight vs full) A/B 스위치로
  살아있고, `qdrant_db/`에 **두 컬렉션**(`tacit_knowledge_insight`, `tacit_knowledge_full`)이 실제로
  존재한다(직접 확인). 현재 기본은 `full`.
- **(확인 불가)** "교체 전후 유사도 변화"의 정확한 before/after 수치 쌍 — 0.27(before, insight)은 기록됨.
  after(full)의 대응 질의 유사도를 같은 표로 나란히 적은 기록은 못 찾음(§6에 full 모드 실측 유사도는 있음).

### 2-4. 설계 논리 명문화 (사실 = 주석, 추정 = 평가)

- **사실(STEP7 주석 논리)**: situation=신입 증상 언어와 붙는 접점 / tacit_insight+reasoning=의미 본체 /
  diagnostic_steps.action=절차 나열이라 노이즈 → STEP7은 steps 제외(vectordb_utils.py:108-111).
- **추정(평가)**: 이 "steps=노이즈" 논리는 타당하나 **STEP8 full 모드는 그 논리를 따르지 않는다**
  (steps 포함). 신입 질의가 절차 표현("어떻게 빼요")일 땐 steps 포함이 오히려 접점이 될 수 있어
  일률적 우열 판단은 어렵다 — A/B(full/insight)를 둔 것 자체가 미결 상태의 방증(추정).

**근거 위치(§2):** `ingest.py:67-78,130-143,163`, `STEP7_DB/vectordb_utils.py:9,108-112`,
`STEP7_DB/README.md:23`, `qdrant_db/collection/`(두 컬렉션), `config.py:32`.

---

## §3. 검색 파이프라인

### 3-1. 시그니처·흐름 (사실, search.py:52-96)

`search(embedder, client, config, query, top_k=None, accept_only=None) → list[dict]`.
흐름: query → `bge-m3.encode(normalize)` → 백엔드 분기(chroma: `collection.query(where=…)` / qdrant:
`query_points`) → **`score ≥ min_similarity_threshold`만** 남김 → `[{id, similarity(round3), routing,
confidence, entry=json.loads(raw_json)}]`(유사도순). 전부 미달이면 **빈 리스트**(환각 방지 — LLM
호출 여부는 app.ask가 결정, app.py:117).

### 3-2. top_k 기본값과 "@10 상향" 현황 (사실 + 정정)

- **현재 `top_k_default = 3`**(config.py:48, README.md:107, HANDOFF.md:37).
- **정정(사실)**: 사용자 언급 "Recall@5 0.73→@10 0.87이라 top_k=10 상향을 문서화"는 STEP8/STEP7
  문서에서 **명시적 상향 권고·결정으로 확인되지 않는다**. 존재하는 것은 STEP1 실험의 Recall@k
  데이터(STEP7 보고서 §2-1: @5 0.733/@10 0.867)뿐이고, **top_k=10으로 올리자는 결정 문구는 못 찾음**
  (grep 결과 top_k 관련은 전부 값 "3"). → **top_k=10 미반영, 상향 검토 문서도 미확인(§9).**
  - **추정:** 현 서비스 DB가 6~9건 규모라 top_k=3이면 사실상 전건에 가깝게 훑어 상향 유인이 약했다.

### 3-3. routing 필터 훅 — 기본 OFF, 현재 리스크 (사실 + 평가)

- **사실**: `accept_only` 인자 존재, **기본 OFF**(`ACCEPT_ONLY=false`, config.py:54). ON이면 chroma
  `where={"routing":"accept"}`(search.py:70)로 accept만.
- **의도된 임시 상태인가**: config.py:53 주석 "STEP7_DB --accept_only 훅과 동일. **기본 꺼짐(진짜
  STEP3 데이터 나오기 전까지)**" → 명시적으로 임시 OFF.
- **현재 리스크(사실)**: OFF라 **hold/reject가 검색 결과에 섞여 나온다.** 실측(통합_작업메모.md:124):
  "래치…" 질의 → `latch_confirm_007(0.608, accept), latch_removal_005(0.606, accept),
  **ch_preread_001(0.445, hold)**` — hold 후보가 top3에 실제로 포함됨.
- **지금 켤 조건이 충족됐나(추정)**: **아니오/부분.** 배포 DB(`input_motion_obs_run6_accept/` 6건)는
  폴더명만 accept일 뿐 **`verification` 블록이 없어 routing 태그가 없다**(STEP7 §5-5#2 확인). 그래서
  `_payload`의 routing=None → chroma metadata에서 None 키 제외(ingest.py:163) → **accept_only를 켜면
  `where={"routing":"accept"}`가 아무것도 못 맞춰 0건 반환**할 위험. → 먼저 step7_adapter를 거쳐
  routing 태그가 붙은 입력으로 재적재해야 필터 ON이 안전(추정).

### 3-4. 유사도/거리 혼동 방어 (사실)

- Chroma는 cosine `distance = 1 - cos_sim` 반환 → `1.0 - distance`로 유사도 복원(search.py:74).
  Qdrant는 score가 이미 유사도(search.py:81). 두 경로를 (payload, 유사도)로 정규화해 공통부 태움
  (search.py:64-65 "반환 형태 바이트 수준 동일"). 임계값 0.40은 양쪽 cosine 기준 유효(§7-3).

### 3-5. UUID5 idempotent (사실)

- ids = `uuid.uuid5(NAMESPACE_URL, e["id"])`(ingest.py:160,184). **단 STEP8은 재적재 시 컬렉션을
  delete 후 create**(ingest.py:154-158,173-174) — "재적재 = 전체 교체". 즉 중복 방지는 upsert-merge가
  아니라 **전체 교체 + UUID5 고정**의 조합이다(STEP7의 upsert 방식과는 다름). 같은 소스 재적재 시
  동일 UUID5라 결과 동일(idempotent).

**근거 위치(§3):** `search.py:41-96`, `config.py:48,53-54`, `ingest.py:154-190`,
`통합_작업메모.md:124`, `STEP7_상세보고서.md §5-5`.

---

## §4. 답변 조립과 근거 검증

### 4-1. 템플릿 vs LLM 재작성 — 현황 확정 (사실)

- **STEP8 서비스(app.ask)는 LLM 재작성 단계까지 구현돼 있다.** Ollama `qwen3:14b`에 검색 결과 JSON을
  컨텍스트로 주고 답변 생성(app.py:123-146). **코드 고정 템플릿(format_answer)이 아니다.**
- **혼동 주의**: 사용자가 말한 "핵심/이유/절차/출처 + ⚠초보자 함정" **고정 템플릿**은 **STEP7의
  `search_tacit.py:73-91`(format_answer, "LLM 재작성 없음")** 경로다. **STEP8 서비스는 그 함수를 쓰지
  않는다** — 대신 SYSTEM_PROMPT로 LLM에게 그 형식(핵심→절차→"출처: 영상ID (시작~끝)")을 지시한다
  (app.py:47-53). 즉 "핵심/이유/절차/출처" 형식은 **코드 템플릿이 아니라 LLM 프롬프트 지시**로 달성.
- **⚠초보자 함정 한 줄**: STEP7 search_tacit.py에는 있으나(`extract_beginner_trap`), **STEP8 app.py의
  SYSTEM_PROMPT/답변 경로엔 없다**(grep상 app에 beginner_trap 미참조). → STEP8 답변엔 초보자 함정
  전용 줄이 구조적으로 보장되지 않음(사실).

### 4-2. LLM 프롬프트·환각 방어 해부 (사실, app.py:47-146)

- **SYSTEM_PROMPT 전문**(app.py:47-53):
  > "너는 조립 현장 신입 작업자를 돕는 음성 작업 보조 AI다. 아래 [검색된 암묵지]만 근거로 답하고,
  > 근거가 부족하면 솔직히 부족하다고 말하라. 규칙: 첫 1~2문장 핵심 먼저(간결한 존댓말) / 절차는
  > 번호 단계 / 마지막 줄 '출처: 영상ID (시작~끝 타임스탬프)' / 마크다운 강조 금지, 일반 텍스트."
- **user 메시지**: `[검색된 암묵지]\n{retrieved JSON}\n\n[신입의 질문]\n{question}`(app.py:132).
- **이중 방어(환각)**: (1) 검색 0건이면 LLM 호출 안 함(app.py:117-121, 정직 실패). (2) SYSTEM_PROMPT
  "검색된 근거만"(app.py:14-15,48). qwen3면 `think:false`(app.py:138-139, 사고문 유출 방지).
- **들어가는 것/안 들어가는 것**: 들어감 = 검색된 후보 raw_json 전체(situation/insight/reasoning/
  steps/source). 안 들어감 = GT, 다른 후보, 매뉴얼. num_predict 256(답변 상한), keep_alive -1(VRAM 상주).

### 4-3. 근거 검증 (clip 범위) — 현황 (사실 + 정정)

- **정정(사실)**: `clip_start ≤ step.timestamp ≤ clip_end` 검증은 **/ask 응답 경로에 없다.** 이 검증은
  **`ingest.py:43-64 validate_timestamps`(적재 시점, 경고만·탈락 안 함)**와 STEP7 `search_tacit.py`
  에만 있다. STEP8 서비스가 답변 반환 직전 timestamp를 재검증하는 코드는 없음(app.py /ask에 미호출).
  → 적재 때 1회 경고, 서비스 응답 때는 미검증(사실).
- **top1 유사도 표기**: 답변 **본문(answer 텍스트)에는 유사도가 안 들어간다**(LLM이 생성). 대신
  **구조화 `sources[]`에 `similarity`가 각 후보별로 포함**(app.py:153) → 프론트가 표시 가능.

**근거 위치(§4):** `app.py:47-53,117-162`, `ingest.py:43-64`, `STEP7_DB/search_tacit.py:73-91`,
grep(beginner_trap app 미참조).

---

## §5. 웹앱 접점 계약 감사 (백엔드 ↔ 프론트)

> **구조(사실)**: 프론트(`static/메인화면.html`, `static/index.html`)와 백엔드(`app.py`)가 **같은
> 리포에 함께** 있다(서시은 v2 역머지). 따라서 크로스-리포 계약이 아니라 **한 리포 내 양측을 직접
> 대조** 가능. 과거 계약 산출물 = `HANDOFF.md`(서시은→안나경 인계), `통합_작업메모.md`(안나경 통합).

### 5-1. API 접점·요청/응답 스키마 (양측 대조)

| 엔드포인트 | 백엔드 제공(app.py) | 프론트 호출 | 일치? |
|---|---|---|---|
| `POST /ask` | req `{question}` → `{answer, sources[]}`(app.py:105-162) | `fetch('/ask', {question})` → `data.answer`, `data.sources`(index.html:72-96, 메인화면.html:484-493) | ✅ |
| `POST /tts` | req `{text}` → audio/wav(app.py:169-182) | `fetch('/tts', {text})`(메인화면.html:505) | ✅ |
| `POST /transcribe` | multipart `file` → `{answer, sources, question}`(app.py:185-202) | `fetch('/transcribe', FormData file)`(메인화면.html:786) | ✅ |
| `GET /` | `static/메인화면.html`(app.py:205-208) | 진입 화면 | ✅ |

- **sources 필드 계약**: 백엔드 제공 = `{id, similarity, routing, task, insight, video, clip}`(app.py:150-159).
  프론트 소비 = `s.insight, s.video, s.clip`(메인화면.html:791), `s.insight` 등(index.html:94). 프론트가
  응답 `{answer, sources}`를 내부적으로 `{answer, refs}`로 매핑(메인화면.html:481,492-493) — 필드 이름만
  내부 변환, **계약 불일치 아님**.

### 5-2. 임베딩 가정 일치 (사실)

- 임베딩은 **백엔드 단독**이 수행(bge-m3 1024, cosine, normalize) — 프론트는 텍스트/음성만 보내고
  임베딩에 관여 안 함. 따라서 "임베딩 모델·차원·거리척도 가정 불일치" 위험은 **구조적으로 없음**
  (프론트가 벡터를 만들지 않음). ingest와 search가 같은 모델·정규화·cosine을 쓰는지가 관건이고
  이는 §3-4에서 일치 확인.

### 5-3. 음성 STT — STEP3 STT와 비교 (사실)

| | STEP8 /transcribe(질의) | STEP3(파이프라인 STT) |
|---|---|---|
| 모델 | faster-whisper `large-v3-turbo` | faster-whisper `large-v3-turbo`(mobiuslabs 리포) |
| language | ko | ko |
| beam_size | **1**(app.py:196, 짧은 질의 속도) | 미지정(기본 5) |
| vad_filter | True | True(vad_parameters threshold 0.5) |
| condition_on_previous_text | 미지정(기본) | **False**(환각 억제) |

- **같은 모델, 다른 설정(사실)**: STEP8은 짧은 질의라 `beam_size=1`로 속도 우선(app.py:194 주석),
  STEP3는 긴 명장 영상이라 정확도·환각억제 설정(condition_on_previous_text=False). **추정:** 용도
  차이에 따른 의도적 분리 — 질의 STT는 빠르고 짧게, 콘텐츠 STT는 정확하게.

**근거 위치(§5):** `app.py:105-208`, `static/메인화면.html:481-506,786-791`, `static/index.html:63-101`,
`HANDOFF.md`, `통합_작업메모.md` 파트4, `STEP3_상세보고서.md §3-2`.

---

## §6. 검색 품질 현황 (실측 기반)

> 실측 출처 = `통합_작업메모.md`(파트3·4 검증 로그) + `logs/smoke_qwen3_14b_2629.log`(E2E 스모크).
> **주의(사실)**: 두 실측은 **다른 DB**다 — 메모 로그는 `gold_records`(9건, routing 태그 有), 후자
> 스모크는 배포 DB `run6_accept`(6건, tk_CLIP*). 섞어 비교 금지.

### 6-1. gold_records 기준 대표 질의 (통합_작업메모.md:124,148,184)

| 질의 | top1 유사도 | 결과 | 방어 |
|---|---|---|---|
| "래치 어떻게 눌러야 해요?" | 0.608 | latch_confirm_007(0.608,accept)/latch_removal_005(0.606,accept)/**ch_preread_001(0.445,hold)** | 정상 답변, hold 혼입(§3-3) |
| "새 램인데도 인식이 안 돼요" | 0.582 | 정상 답변 | 통과 |
| "오늘 날씨 어때요" | <0.40 | sources=[] (0.157s) | 방어선1 차단(LLM 미호출) |
| "오늘 점심 뭐 먹지" | 0.372 | 0.40 상향 후 sources=[] | 방어선1 차단(과거 0.35땐 통과→방어선2가 막음) |

### 6-2. 배포 DB(run6_accept) E2E 스모크 (logs/smoke_qwen3_14b_2629.log)

- "램을 뺄 때 주의할 점?" → 답변(래치 동시 누름, 슬롯 파손 경고, 절차 2단계, "출처: tk_CLIP2_004
  (00:01:20~00:01:35)") / sources=[tk_CLIP2_004, tk_CLIP3_…_002, tk_CLIP1_…_004]. **SMOKE_OK.**
- "메모리 채널 정상 확인?" → 답변 + 출처 tk_CLIP4_…_007. LLM 지연 ~6~16s, 42~44 tok/s(ollama_smoke.log).
- **사실**: 배포 DB E2E가 실제로 "증상 질의 → 근거 기반 답변(핵심/절차/출처)"을 산출함을 확인.
  단 이 로그엔 **유사도 수치가 안 찍혀 있다**(sources id만) → run6 DB의 질의별 top1 유사도는 로그에
  없음(§9).

### 6-3. 음성 왕복 (통합_작업메모.md:182-190)

- TTS 합성 질문 wav → /transcribe STT "레츠"(래치 오인식) → **그래도 래치 문서 정상 회수(top1 0.556)**
  → 답변 → TTS wav. 왕복 16.9s. **추정:** 짧은 STT 오인식에 임베딩 검색이 관대(부분 오타 견딤).
  단 실마이크(브라우저 webm) 검증은 미완(통합_작업메모.md:189, §9).

### 6-4. Recall 미스 2건 이관 기록 (확인 불가/부분)

- STEP1 실험 Recall@10=0.867 → 15질문 중 **2건 미스**(STEP7 §2-1). **(확인 불가)**: 그 2건이
  구체적으로 어느 질의였고 "청킹·임베딩 개선 과제로 이관"했다는 명시 기록을 STEP8 폴더에서 못 찾음.
  results.csv에서 세 DB 공통 hit@10=0인 질의를 역추적하면 후보를 특정할 수는 있으나(예: hit@10 0인
  "Qwen3-14B enable_thinking", "아멘 repeat_hallucination" 질의 — results.csv), 이는 **문서 도메인 QA**
  질의라 암묵지 후보 검색과 성격이 다르다 → 개선 과제 이관 문구 자체는 미확인.

**근거 위치(§6):** `통합_작업메모.md:124,148,182-190`, `logs/smoke_qwen3_14b_2629.log`,
`logs/ollama_smoke.log`, `STEP7_상세보고서.md §2-1`.

---

## §7. 트러블슈팅 아카이브

### 7-1. 유사도 0.27 → 임베딩 합본 재설계

- **증상**: tacit_insight 단독 임베딩 시 증상 질의 top1 유사도 0.27대(STEP7 vectordb_utils.py:9).
- **원인**: insight 한 문장이 신입의 "증상 언어"와 어휘 접점이 적음.
- **해결**: situation 포함 합본. STEP8은 `full` 모드(+steps+keywords)로, STEP7은 situation+insight+
  reasoning으로(설계 분기, §2-1). insight 모드는 A/B arm으로 보존(두 컬렉션).
- **남은 장치**: `EMBED_MODE` 스위치 + `evaluate.py`로 Recall 재측정 가능.
- **재발 시**: insight로 되돌리면 증상 질의 유사도 하락 — evaluate.py로 확인.

### 7-2. SIM_THRESHOLD 0.35 → 0.40 (방어선1 구멍)

- **증상**: "오늘 점심 뭐 먹지" top1=0.372 > 0.35라 방어선1 통과, LLM까지 감(통합_작업메모.md:148).
- **원인**: bge-m3 무관 질의 baseline이 0.30~0.36대. 서시은 원본 0.35가 너무 낮음.
- **실패한 시도(운)**: 방어선2(프롬프트)가 그 질의를 운좋게 막음 — "구조적 보장 아님"(메모:159).
- **해결**: STEP7 실측값 **0.40으로 상향**(config.py:49-52). 재검증: 0.372<0.40 차단, "새 램" 0.582 회귀 없음.
- **남은 장치**: 이중 방어(임계값 + 프롬프트). 재발 시 무관 질의 top1이 0.40 근처인지 확인.

### 7-3. 유사도/거리 스코어 혼동 방어 (백엔드 전환)

- **증상/원인**: Qdrant→Chroma 전환 시 Chroma는 distance(=1-cos), Qdrant는 similarity를 반환 —
  그대로 임계값 비교하면 의미 반전.
- **해결**: chroma 경로에서 `1.0 - distance`로 유사도 복원(search.py:74), 임계값 0.40 그대로 유효.
  두 경로를 공통 형태로 정규화(search.py:64-65).
- **재발 시**: 백엔드 교체 시 유사도 방향(높을수록 유사)이 유지되는지, 임계값 의미 불변인지 확인.

### 7-4. 백-프론트 계약 / 음성 계층 역머지

- **증상/원인**: 서시은 v2 requirements에 `python-multipart` 누락 → UploadFile(/transcribe) 실패
  (통합_작업메모.md:166 부근).
- **해결**: requirements에 python-multipart 명시 추가(통합_작업메모.md 파트4). SIM_THRESHOLD는 v2의
  0.35로 되돌리지 않고 0.40 유지, search.py 분리·accept_only 훅·gold_records DB 유지.
- **남은 장치**: preflight가 backend별 필요 패키지 검사(preflight.py:21 `BACKEND_MODULES`).
- **재발 시**: 음성 경로 의존성(supertonic/faster-whisper/multipart) preflight 통과 확인.

### 7-5. Ollama 운영 함정

- **증상**: Ollama 기본 바인딩 127.0.0.1이라 로그인 노드에서 접근 불가 + 남의 GPU잡에 편승한 임시방편.
- **해결**: `OLLAMA_HOST=0.0.0.0`, `OLLAMA_URL=http://n5:11434`, `서버시작.sh`로 전용 GPU잡
  (salloc --no-shell)+ollama+앱 일괄 기동(통합_작업메모.md:110-115, 파트4). keep_alive:-1로 모델 상주.
- **재발 시**: 운영 배포 시 전용 GPU 할당 필요(임시 편승 금지, 메모:113-115).

**근거 위치(§7):** `통합_작업메모.md` 파트3·4, `config.py:49-52,64-65`, `search.py:74`,
`preflight.py:21`, `서버시작.sh`, `STEP7_상세보고서.md §7`.

---

## §8. 재디벨롭 가이드

### 8-1. 커맨드

```bash
cd project-ai/STEP8_RAG서비스
# 1) DB 구축(자체 ingest — STEP7 DB와 별개)
.venv/bin/python3 ingest.py                          # config input_dir(run6_accept) → full 모드
EMBED_MODE=insight .venv/bin/python3 ingest.py       # insight A/B arm
.venv/bin/python3 ingest.py --input <다른 폴더>
# 2) 검색 스모크(읽기 전용)
.venv/bin/python3 search.py "램 뺄 때 주의점" --top_k 3
.venv/bin/python3 evaluate.py --top-k 3              # gold_records EVAL_SET Recall@k
# 3) 웹앱 구동
OLLAMA_MODELS=/home/ai_user/team_a2/.ollama/models ~/.local/ollama/bin/ollama serve &
bash 서버시작.sh                                     # 전용 GPU잡+ollama+uvicorn(포트 8001)
# 또는 .venv/bin/uvicorn app:app --port 8000
```

### 8-2. 개선 후보 우선순위 (근거 병기)

| 순위 | 항목 | 근거 |
|---|---|---|
| ① | **routing 필터 ON 전환** — 단, 먼저 step7_adapter로 routing 태그 붙은 입력 재적재(현 run6_accept는 태그 부재). 그래야 hold/reject 혼입(§3-3) 차단 | §3-3, ingest.py:140 |
| ② | **top_k 재검토** — 현 3. Recall@10 데이터는 있으나 상향 결정 문서 부재. DB 규모 커지면 5~10 검토(evaluate.py로 재측정) | §3-2 |
| ③ | **임베딩 텍스트 STEP7/STEP8 통일** — full(steps 포함) vs situation+insight+reasoning(steps 제외)이 병존. 하나로 확정 후 A/B 근거 문서화 | §2-1 |
| ④ | **답변 근거 검증 이식** — /ask 응답 전 clip 범위·초보자 함정 줄을 STEP7 search_tacit 수준으로 추가(현재 미검증·미포함) | §4-1,4-3 |
| ⑤ | **서버 모드 클라이언트 경로** — Qdrant/Chroma 서버 배포 시 client(url=) 인자 경로 추가(현재 path=만) | §1-3 |
| ⑥ | **실마이크(브라우저 webm) 검증** — 합성음 왕복만 검증됨 | §6-3 |

### 8-3. 건드리면 위험한 불변식

1. **임베딩 텍스트 구성 변경 = 전량 재적재 필수.** build_document를 바꾸면 저장 벡터와 질의 임베딩
   공간 정합이 깨진다. STEP8은 재적재가 delete+create(전체 교체, ingest.py:154-158)라 부분 변경 시
   신구 벡터 혼재는 없으나, **모델 자체를 바꾸면(bge-m3→X) 반드시 재적재**(§STEP7 불변식과 동일).
2. **UUID5 idempotent.** id 파생 규칙(uuid5(NAMESPACE_URL, tacit_id))을 바꾸면 재적재 중복 위험.
3. **답변은 payload 사실만으로.** SYSTEM_PROMPT "검색된 근거만" + 검색 0건 시 LLM 미호출 —
   출처 없는 문장 생성 금지 장치를 약화하지 말 것(app.py:14-15,117-121).
4. **백-프론트 JSON 계약.** `/ask`={question}→{answer,sources[]}, `/tts`={text}, `/transcribe`=multipart.
   프론트가 이 형식에 의존(§5-1) — 스키마 변경 시 양측 동시 수정.
5. **유사도/거리 방향.** 백엔드 교체 시 distance→similarity 복원(search.py:74) 유지 — 임계값 0.40 의미 불변.

**근거 위치(§8):** `ingest.py`, `search.py`, `evaluate.py`, `app.py:14-15,117-121`, `서버시작.sh`,
`config.py`, `STEP7_상세보고서.md`.

---

## §9. "확인 불가" / 미구현 / 계약 불일치

### 9-1. 확인 불가 항목

1. **top_k=10 상향 검토 문서** — 명시 권고·결정 문구 미확인(Recall@10 데이터만 존재). 현행 top_k=3.
2. **0.27 before/after 유사도 쌍** — before(insight 0.27) 기록됨, after(full) 대응 질의 유사도를 나란히
   적은 표는 미확인(§6에 full 실측 유사도는 있으나 동일 질의 대조쌍은 아님).
3. **배포 DB(run6_accept) 질의별 top1 유사도** — 스모크 로그에 sources id만, 유사도 수치 미기록.
   (재실측하려면 bge-m3 로드 필요 — 무거워 본 보고서에서 미실행. gold_records 유사도는 §6-1에 실측 有.)
4. **Recall 미스 2건의 구체 질의·이관 기록** — 개선 과제 이관 문구 미확인(§6-4).
5. **서버 모드(Qdrant/Chroma 원격) 실배포** — client(url=) 경로 코드 없음, 미검증.
6. **실마이크 음성 검증** — 합성음 왕복만 검증(§6-3).

### 9-2. 미구현 목록 (재디벨롭 착수 지점)

- routing 필터 ON 미전환(기본 OFF, run6_accept에 routing 태그 부재로 켜면 0건 위험).
- top_k=10 미반영(현 3).
- **/ask 응답 시 clip 범위 근거 재검증 미구현**(적재 시 경고만; STEP7 search_tacit엔 있음).
- **⚠초보자 함정 한 줄 미포함**(STEP8 답변 경로; STEP7엔 extract_beginner_trap 존재).
- 임베딩 텍스트 STEP7/STEP8 미통일(full vs steps-제외 병존).
- 서버 모드 클라이언트 경로 미구현. 실마이크 미검증.
- **(주의)** "LLM 답변 재작성"은 **미구현 아님** — Ollama qwen3:14b로 구현됨(§4-1). 사용자 프롬프트의
  "미구현 시 기록" 전제와 달리 실제로는 가동 중.

### 9-3. §5 계약 불일치 표 ("터지는 지점")

| 접점 | 백엔드 | 프론트 | 상태 |
|---|---|---|---|
| `/ask` req/resp | {question}→{answer,sources[]} | 동일 소비(sources→refs 내부매핑) | **일치** |
| `/tts`, `/transcribe` | {text}/multipart file | 동일 호출 | **일치** |
| sources 필드 | id/similarity/routing/task/insight/video/clip | insight/video/clip 소비 | **일치**(프론트가 일부만 사용, 잉여는 무해) |
| 임베딩 가정 | 백엔드 단독 수행 | 관여 안 함 | **불일치 위험 없음**(구조적) |
| STT 설정 | /transcribe beam_size=1 | — | STEP3와 다르나 의도적(§5-3) |

→ **현재 코드 기준 백-프론트 계약 불일치(터지는 지점) 없음.** 과거 사건은 requirements의
`python-multipart` 누락(해결됨, §7-4)과 SIM_THRESHOLD 0.35↔0.40(0.40으로 확정)뿐.

---

*(끝) 본 보고서는 조사·설명·평가 전용이며 코드를 수정하지 않았다.*
