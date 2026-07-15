# STEP7_DB — 확정된 암묵지 JSON → Embedding & Vector DB → 검색(RAG)

STEP6을 통과(혹은 통과했다고 가정)한 암묵지 최종 JSON(schema_version 1.3)을 읽어
Vector DB(Qdrant, 로컬 디스크)로 적재하고, 신입 작업자의 증상 질의(자연어)를 받아
[핵심/이유/절차/출처] 형식으로 답변을 조립하는 검색 파이프라인.

## 스택

- 임베딩: `BAAI/bge-m3` (STEP6_품질검증과 동일 모델, `langchain_huggingface`)
- Vector DB: Qdrant, `path=` 로컬 디스크 영구 저장(서버 프로세스 불필요)
- 문서 9건(짧은 텍스트)만 다루므로 **GPU 불필요** — `config.device` 기본값이 `cpu`라
  로그인 노드에서 `srun` 없이 바로 실행된다.

## 임베딩 대상 (2026-07-04 변경)

`situation + tacit_insight + reasoning` 합본을 임베딩한다:
```
[상황] {situation}
[노하우] {tacit_insight}
[이유] {reasoning}
```
`tacit_insight` 단독 임베딩이었을 때는 신입의 증상 질의("화면이 안 떠요")와 유사도가
낮았다(실측 0.27대). `situation`이 증상 질의와 붙는 접점 역할을 하도록 합본으로 바꿔
top1 유사도가 0.44~0.61대로 올라갔다. `diagnostic_steps.action`은 절차 나열이라
노이즈가 커서 임베딩엔 안 넣고 payload로만 보관한다.

## 실행

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python3 -m pip install -r requirements.txt

.venv/bin/python3 preflight.py     # 단독 프리플라이트만
.venv/bin/python3 build_db.py      # 적재 + 검색 스모크 테스트

# 검색(RAG) — 신입 작업자 질의 테스트
.venv/bin/python3 query_tacit.py "화면이 안 떠요"
.venv/bin/python3 query_tacit.py "래치 어떻게 눌러요" --top_k 2
.venv/bin/python3 query_tacit.py "커넥터를 손으로 눌러서 확인" --accept_only   # hold/reject 제외
```

재실행해도 안전하다 — 각 암묵지 `id`를 UUID5로 변환해 Qdrant point ID로 쓰기 때문에
같은 `id`는 새로 쌓이지 않고 덮어써진다(idempotent upsert). `build_db.py`를 몇 번을
돌려도 컬렉션은 항상 입력 JSON 개수만큼만 유지된다.

## 답변 조립 규칙 (템플릿 조립까지만, LLM 재작성 없음)

```
[유사도 0.444]
핵심: {tacit_insight}
이유: {reasoning}
절차:
  1. {diagnostic_steps[0].action}
  ...
출처: {video_id} ({clip_start}–{clip_end})
⚠ 초보자 함정: {"초보자"가 들어간 문장}   ← 있을 때만
```
- `source_utterance`는 절차에 안 씀(action_only 항목은 null이라 action만 기술).
- **환각 금지**: `min_similarity_threshold`(현재 0.40) 미만이면 답변을 조립하지 않고
  "관련 노하우를 찾지 못했습니다"로 응답한다. 이 값은 실측으로 정함 — 처음 0.30으로
  뒀더니 완전 무관한 질의("오늘 점심 뭐 먹지" 등)도 bge-m3 baseline 유사도가 0.30~0.36대로
  나와서 안 걸러졌다. 무관한 질의군(0.30~0.36)과 실제 관련 질의군(0.44+) 사이에 뚜렷한
  갭이 있어 0.40으로 올림(`config.py` 참고).
- `diagnostic_steps[].timestamp`가 `source.clip_start~clip_end` 범위 밖이면 경고 로그를
  남긴다(`search_tacit.validate_timestamps`, STEP6 timestamp_validity와 같은 규칙) —
  검색 결과 자체를 폐기하진 않고 경고만.
- `routing=="accept"`만 검색하고 싶으면 `--accept_only`(또는 `search_tacit(..., accept_only=True)`).
  기본값은 꺼짐 — 지금은 STEP3 VLM 버그로 진짜 routing 데이터가 없어 훅만 만들어둔 상태.

## 지금 상태 (2026-07-04)

- 입력: `gold_records/`(STEP3 VLM이 아직 0건 버그라 실제 STEP3 산출물 대신 손으로 만든 예시 9건)
- **routing 필터 기본 OFF** — hold/reject 후보도 검색 대상에 포함, `verification.routing`/
  `confidence`는 payload에 저장돼 결과에 그대로 딸려나온다. `--accept_only`로 켤 수 있음
  (D5 훅). STEP3 VLM이 고쳐져서 진짜 데이터가 나오면 기본값을 켜는 걸 검토할 것.
- 실측 확인 완료: 9건 적재, 예시 질의 3개("화면이 안 떠요"/"램이 안 잡혀요"/"래치 어떻게
  눌러요") 전부 관련 레코드가 top1으로 정확히 나옴. 무관한 질의는 "못 찾음"으로 정직하게
  응답. `--accept_only` 필터도 hold 후보(001/002)를 정확히 제외함. 재적재해도 9건 유지.
- 미룬 것(명시적 범위 밖): LLM 답변 재작성(명장 말투 → 초보 눈높이), 하이브리드/리랭커 검색,
  routing=accept 강제 필터(기본 ON 전환).

## 파일 구조

| 파일 | 역할 |
|---|---|
| `config.py` | pydantic 설정(input_dir, 임베딩 모델, Qdrant 경로/컬렉션, 검색 임계값) |
| `preflight.py` | 무거운 임베딩 모델 로드 전 패키지/config/경로 검증 |
| `vectordb_utils.py` | 임베딩 로더, JSON→Document 변환(합본 임베딩), Qdrant 적재/검색 |
| `build_db.py` | 진입점 — 적재 + 검색 스모크 테스트 |
| `search_tacit.py` | 검색(RAG) 파이프라인 — 질의→검색→[핵심/이유/절차/출처] 답변 조립, timestamp 검증, accept_only 필터 |
| `query_tacit.py` | 진입점 — 신입 작업자 질의 테스트용 CLI |
| `gold_records/` | 예시 입력 JSON 9건 + 생성 스크립트(`make_gold_records.py`) |
| `qdrant_db/` | 실제 Vector DB 파일(로컬 디스크, git에는 안 올림) |
