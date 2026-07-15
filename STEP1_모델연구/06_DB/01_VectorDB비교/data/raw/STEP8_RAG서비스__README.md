# STEP8_RAG서비스 — 현장 암묵지 음성 RAG (통합본)

STT로 질문 → 벡터검색(RAG) → LLM 답변 생성 → TTS 응답하는 웹앱의 **백엔드 3파트**
(적재/검색/LLM 답변 생성)만 구현한 것. 두 프로젝트를 통합했다:

- **서시은 `voice-rag-upload`** — 뼈대(FastAPI 서버, 브라우저 UI, Ollama LLM 생성,
  환경변수 하이퍼파라미터 스위치, Recall@k 평가)
- **안나경 `STEP7_DB`** — 방어 로직(프리플라이트, pydantic config 검증, 스키마 관용적
  실패, timestamp 검증, assert 스모크 테스트, accept/hold/reject 필터)

통합 배경·결정 근거·실측 로그는 `통합_작업메모.md`에 자세히 있고, 팀 공유용 요약본은
`docs/STEP8_RAG서비스_통합보고서.md`(project-ai 최상위 docs)에 있다.

## 데이터

입력은 이 폴더에 복제하지 않고 **`../STEP7_DB/gold_records`를 그대로 참조**한다
(`config.py`의 `input_dir` 기본값). STEP3 VLM이 아직 실제 데이터를 못 뽑아서 STEP7_DB가
만든 예시 9건(Dell Precision 7920 조립/부팅 시나리오, `verification.routing`/`confidence`
포함)을 쓰는 중이다. 실제 STEP3 산출물이 나오면 `input_dir`만 바꾸면 된다.

## 아키텍처

```
[브라우저] 마이크(Web Speech STT) ──▶ POST /ask ──▶ [FastAPI]
                                                       │
                                          ① 질문 임베딩 (bge-m3)
                                          ② Qdrant 검색 (cosine, top-k)
                                          ③ 유사도 임계값 필터 ─┐
                                                       │        │ 미만이면 LLM 호출 안 함
                                          ④ Ollama LLM 답변 생성 ┘ (방어선1)
                                             (시스템 프롬프트: 근거 부족하면 솔직히 — 방어선2)
                                                       │
[브라우저] 스피커(TTS) ◀──────────── JSON {answer, sources}
```

## 파일 구조

| 파일 | 역할 | 베이스 |
|---|---|---|
| `preflight.py` | 무거운 임베딩 모델 로드 전 5초 내 검증(import/config/경로) | 안나경 |
| `config.py` | pydantic 설정. 서시은의 환경변수를 pydantic 필드로 감싸 검증 | 병합 |
| `ingest.py` | JSON→임베딩→Qdrant 적재. raw_json 통째 보존 + embed-mode 스위치 | 서시은+안나경 |
| `search.py` | 질문 임베딩→검색→임계값 필터(+accept_only 필터). app.py/evaluate.py 공용 | 안나경 이식 |
| `evaluate.py` | Recall@k 평가(EVAL_SET은 STEP7 gold_records 9건 기준) | 서시은 |
| `app.py` | FastAPI `/ask` — search.py 결과로 Ollama 호출, 이중 방어 | 서시은+안나경 |
| `static/index.html` | 브라우저 UI(마이크 버튼, STT/TTS) | 서시은 |
| `통합_작업메모.md` | 통합 과정 원자료 로그(파트별 검증 기록) | 신규 |

## 실행

### 0. 최초 1회 준비

```bash
cd STEP8_RAG서비스
python3 -m venv --system-site-packages .venv   # torch/transformers는 시스템 것 재사용
.venv/bin/pip install -r requirements.txt
```

### 1. Ollama 서버

이 팀 계정(`team_a2`)에는 sudo 없이 설치된 Ollama가 `~/.local/ollama`에 이미 있고
`qwen2.5:14b-instruct`가 pull되어 있다(추가 다운로드 불필요).

**GPU 노드에서 띄우는 것을 강력히 권장** — 로그인 노드(CPU)로 띄우면 14b 모델 추론에
2분 넘게 걸려 타임아웃 난다. 이미 떠 있는 팀 공용 GPU 잡이 있으면 거기 편승:

```bash
squeue                      # team_a2 소유 RUNNING 잡의 JOBID 확인 (예: 2034, n5, RTX6000 x2)
srun --jobid=<JOBID> --overlap bash -c \
  'OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=~/.local/ollama/data ~/.local/ollama/bin/ollama serve'
```

- `OLLAMA_HOST=0.0.0.0`을 꼭 줘야 한다 — 기본값(127.0.0.1)이면 이 서버가 뜬 노드(n5) 밖에서
  (로그인 노드의 app.py에서) 접근이 안 된다.
- 떠 있는 공용 GPU 잡이 없으면 `srun -p RTX6000 -w n5 --gres=gpu:1 --pty bash -l`로 직접
  새로 잡고 그 안에서 위 명령을 돌린다(docs/전처리_파이프라인_계획서.md, CLAUDE.md 클러스터
  섹션 참고).
- 기동 확인: `curl http://<노드>:11434/api/tags` — `qwen2.5:14b-instruct`가 보이면 OK.

### 2. 벡터DB 적재

```bash
.venv/bin/python3 ingest.py                 # config.py의 input_dir(STEP7_DB/gold_records) 사용
EMBED_MODE=insight .venv/bin/python3 ingest.py   # tacit_insight만 임베딩(비교용, 별도 컬렉션)
```

### 3. 서버 실행

```bash
OLLAMA_URL="http://<Ollama가 뜬 노드>:11434/api/chat" \
  .venv/bin/uvicorn app:app --port 8000
# → http://localhost:8000 접속, 마이크 버튼(Chrome 권장)
```

### 4. 검색/평가만 따로 테스트

```bash
.venv/bin/python3 search.py "래치 어떻게 눌러요"       # 검색만(LLM 없이)
.venv/bin/python3 search.py "래치 어떻게 눌러요" --accept_only
.venv/bin/python3 evaluate.py --top-k 3               # Recall@k
```

## 하이퍼파라미터 (환경변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `TOP_K` | 3 | 검색 결과 수 |
| `SIM_THRESHOLD` | **0.40** | 코사인 유사도 컷오프. 2026-07-04 실측으로 0.35(서시은 원본)→0.40(STEP7_DB 실측값)로 상향 — 아래 참고 |
| `COLLECTION` | (embed_mode로 자동 결정) | 강제 지정 시 사용 |
| `EMBED_MODE` | full | `full`(상황+노하우+이유+절차+키워드) / `insight`(단독) |
| `ACCEPT_ONLY` | false | true면 `verification.routing=="accept"`만 검색 |
| `LLM_MODEL` | qwen2.5:14b-instruct | 이 환경에 이미 pull된 모델 |
| `OLLAMA_URL` | http://localhost:11434/api/chat | Ollama가 다른 노드에 떠 있으면 그 주소로 변경 |

## 환각 방지 이중 방어

1. **방어선1(구조적)** — `search.py`가 `SIM_THRESHOLD` 미만 결과를 걸러내고, 전부 걸러지면
   빈 리스트를 반환한다. `app.py`는 빈 리스트를 받으면 **LLM을 아예 호출하지 않고** 즉시
   "찾지 못했다"고 응답한다.
2. **방어선2(프롬프트)** — 방어선1을 통과했지만 애매한 경우를 위해 시스템 프롬프트에
   "검색된 암묵지만 근거로, 부족하면 부족하다고" 명시.

실측: `SIM_THRESHOLD=0.35`였을 때 "오늘 점심 뭐 먹지"(top1=0.372)가 방어선1을 뚫었지만
방어선2가 막아냄. `0.40`으로 올린 뒤 재검증하니 방어선1에서 바로 차단됨(자세한 로그는
`통합_작업메모.md` 파트3 참고).

## 알려진 제약 / 다음 할 일

- **입력 데이터가 예시(gold_records)** — 실제 STEP3 산출물이 아니다. STEP3 VLM 버그가
  고쳐져서 실제 데이터가 나오면 `config.py`의 `input_dir`을 그 경로로 바꾸고 재적재할 것.
- **Ollama가 팀 공용 GPU 잡에 편승 중** — 정식 배포 시 이 프로젝트 전용 GPU 할당이 필요.
- **timestamp 검증은 적재 시점 1회성 경고** — 실패해도 적재를 막지 않음(의도된 설계).
- `voice-rag-upload/voice-rag/inspect_db.py`, `knowledge_db/`(ChromaDB 잔재)는 안 가져옴 —
  지금 스택(Qdrant)과 안 맞는 이전 실험 흔적이라 죽은 코드로 판단.
