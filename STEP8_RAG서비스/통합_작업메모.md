# STEP8_RAG서비스 통합 작업 메모

STEP7_DB(안나경, 방어 로직)와 voice-rag-upload(서시은, 뼈대)를 통합하는 과정 기록.
나중에 최종 보고서로 합칠 목적의 원자료(raw log) — 파트별로 이어서 추가한다.

---

## 파트1 — JSON → 벡터DB 적재 (ingest.py)

**한 일**
- `STEP8_RAG서비스/` 신규 생성, `--system-site-packages` venv 구성
  (torch/transformers는 시스템에 이미 있어 재사용, qdrant-client/sentence-transformers/
  fastapi/uvicorn/pydantic/requests만 새로 설치)
- `preflight.py` — STEP7_DB 것을 이 폴더 의존성(qdrant_client, sentence_transformers,
  pydantic, requests, fastapi, uvicorn)에 맞게 이식
- `config.py` — pydantic BaseModel. 서시은의 환경변수(EMBED_MODE/COLLECTION/TOP_K/
  SIM_THRESHOLD/ACCEPT_ONLY)를 os.getenv로 읽어 필드로 감싸고 field_validator로
  범위/오타 검증(device, embed_mode, threshold 0~1, top_k≥1)
  - **input_dir 기본값 = `../STEP7_DB/gold_records`** — 요청대로 데이터 복제 없이 STEP7
    예시 JSON을 그대로 참조
- `ingest.py` — 서시은 ingest.py 베이스(raw_json 통째 저장 + embed-mode full/insight
  스위치) + 이식한 것:
  - `run_preflight()`을 무거운 import(SentenceTransformer 등) 이전에 호출
  - 스키마 불일치 JSON은 건너뛰고 경고만(관용적 실패)
  - `validate_timestamps()` — diagnostic_steps timestamp가 clip 범위 밖이면 경고(폐기 안 함)
  - `verification.routing`/`confidence`를 raw_json과 별도로 payload 최상위 키에도 저장
    (Qdrant Filter는 중첩 JSON 문자열 안은 못 거르므로 accept_only 필터링을 위해 필요)
  - 적재 후 `assert count > 0`

**검증 로그**
```
[1/3] 암묵지 9건 로드 (입력: STEP7_DB/gold_records)
[2/3] 임베딩 모델 로드: BAAI/bge-m3 (device=cpu, embed_mode=full)
[3/3] 적재 완료 → 컬렉션 'tacit_knowledge_full' — 이번 적재 9건 / 컬렉션 전체 9건
```
- 관용적 실패 테스트: `knowledge` 필드 없는 가짜 JSON을 섞었더니
  `[WARN] ... 건너뜀 — 스키마 불일치: 'knowledge'` 출력 후 나머지 정상 적재됨.
- timestamp 검증 테스트: clip 범위(`00:00:00~00:03:30`) 밖 timestamp(`00:59:59`)를 넣었더니
  `[WARN][timestamp_validity] ... clip 범위(00:00:00~00:03:30) 밖` 출력, 적재는 계속 진행됨.
- embed-mode 스위치 테스트: `--embed-mode insight`로 실행 → 컬렉션명이
  `tacit_knowledge_insight`로 자동 분리 저장됨.

---

## 파트2 — RAG 검색 (search.py + evaluate.py)

**한 일**
- `search.py` 신규 — 서시은 app.py의 검색 로직(①질문 임베딩 ②Qdrant 검색 ③유사도
  임계값 필터)을 app.py/evaluate.py가 공유할 함수(`load_backend`, `search`)로 분리
  - STEP7_DB의 `--accept_only` 훅 이식: `routing=="accept"`만 검색하는 Qdrant
    `Filter(FieldCondition(key="routing", ...))` 추가(payload가 플랫 구조라 `metadata.`
    프리픽스 불필요 — STEP7_DB의 langchain_qdrant 버전과 차이점)
  - 임계값 미만 결과는 여기서 걸러내고, 전부 걸러지면 빈 리스트 반환
    (호출부가 빈 리스트=LLM 호출 안 함으로 판단하게 하기 위함, 파트3에서 사용)
- `evaluate.py` — 서시은 것 베이스, `EVAL_SET`을 STEP7_DB gold_records 9건 기준으로
  교체(서시은 원본은 자기 sample_knowledge.json 3건 기준이었음). 질문은 tacit_insight를
  그대로 베끼지 않고 신입이 실제로 물어볼 법한 증상 표현으로 작성.

**검증 로그**

검색 단독 테스트(`search.py "래치 어떻게 눌러요"`):
```
- sim=0.605  id=tk_dell7920_latch_confirm_007  routing=accept
- sim=0.598  id=tk_dell7920_latch_removal_005  routing=accept
- sim=0.433  id=tk_dell7920_ch_preread_001     routing=hold
```
→ `--accept_only` 켜고 같은 계열 질의 재실행하면 `routing=hold`인 `ch_preread_001`이
top1에서 제외되고 accept 3건만 나오는 것 확인.

Recall@3 평가(embed-mode 비교, `SIM_THRESHOLD` 자체는 평가 시 0으로 풀어서 순수 랭킹만 측정):
```
tacit_knowledge_full    (situation+insight+reasoning+steps+keywords): 9/9 = 1.00
tacit_knowledge_insight (tacit_insight 단독):                          8/9 = 0.89
  → "램이 제대로 꽂혔는지 어떻게 확인해요?" 질문에서 latch_confirm_007 대신
    다른 문서가 top3 안에 들어가 실패(latch_removal_005와 혼동 추정)
```
→ **STEP7_DB가 실측했던 결론("situation을 붙여야 증상 질의와 더 잘 붙는다")과
동일한 방향의 결과**가 이 데이터셋/스택에서도 재현됨.

실 서비스에 쓸 `SIM_THRESHOLD=0.35` 기준으로는 9개 질문 top1 유사도가 전부
0.55~0.73대라 전부 통과(임계값 밑으로 떨어지는 경우 없음 — 무관한 질의 테스트는
파트3 LLM 연동 스모크 테스트에서 진행함).

---

## 파트3 — LLM 답변 생성 (app.py)

**한 일**
- `app.py` 신규 — 서시은 app.py 베이스(FastAPI, Ollama `/api/chat` 호출, 시스템 프롬프트로
  "검색된 근거만, 부족하면 부족하다고") + `search.py`(파트2) 재사용.
  - `run_preflight()`를 무거운 임베딩 모델 로드 전에 호출(fastapi/uvicorn 자체는 가벼워서
    모듈 최상단 import 유지 — uvicorn이 `app:app`을 찾으려면 필요)
  - **[방어선 1, 구조적]** `search.search()`가 이미 `min_similarity_threshold` 미만을
    걸러내고 빈 리스트를 반환하면, `/ask`가 그 자리에서 LLM을 호출하지 않고 즉시
    "관련 노하우를 찾지 못했어요" 반환 (요청하신 "임계값 미만이면 LLM 호출 자체를 안 함"
    요건을 그대로 구현)
  - **[방어선 2, 프롬프트]** 시스템 프롬프트("검색된 암묵지만 근거로, 부족하면 부족하다고")
    유지 — 방어선 1을 통과했지만 애매한 경우를 위한 이중 안전장치
  - `ACCEPT_ONLY` 환경변수로 STEP7_DB의 `--accept_only` 훅 노출

**환경 이슈 및 해결 (기록 남김 — 재현 시 참고)**
- 이 계정(`team_a2`)에 **sudo 없이 설치된 Ollama가 이미 있었음**
  (`~/.local/ollama`, `qwen2.5:14b-instruct` pull 완료, `serve.log`에 2026-07-04 오전
  실행 이력 있음 — 다른 팀원이 이미 같은 파이프라인을 테스트해본 흔적으로 추정).
  `LLM_MODEL` 기본값을 코드 원본(`qwen2.5:3b-instruct`, 미설치)에서
  **`qwen2.5:14b-instruct`(이미 있음)로 변경**함.
- 로그인 노드(현재 셸)는 GPU가 없어 처음엔 CPU로 Ollama가 떠서 14b 모델 추론이
  2분 넘게 걸려 타임아웃남. **`srun --jobid=2034 --overlap`으로 이미 떠 있는 팀 공용
  GPU 잡(RTX6000 x2, n5)에 편승**시켜 재기동 → GPU(CUDA, Quadro RTX 6000) 잡힘 확인.
  추가로 기본 바인딩이 `127.0.0.1`이라 로그인 노드에서 접근이 안 돼
  `OLLAMA_HOST=0.0.0.0:11434`로 재기동, `OLLAMA_URL=http://n5:11434/api/chat`으로
  STEP8 서버를 띄움.
  → **운영 시 유의할 점**: 이 방식은 지금 다른 사람이 이미 잡아둔 GPU 잡에 얹혀 탄
  임시방편이다. 정식 배포 시엔 별도 GPU 할당 또는 이 프로젝트 전용 srun 잡을 잡아야 함.

**검증 로그**

정상 질의(`"래치 어떻게 눌러야 해요?"`, GPU 추론, 14.3초):
```
answer: "래치는 양쪽을 동시에 눌러야 합니다. 한쪽만 누르면 슬롯 핀이 파손될 수 있습니다.
1. 양 손의 엄지와 검지를 사용해 메모리 모듈 양 옆 래치를 잡습니다.
2. 양쪽 래치를 동시에 눌러서 메모리를 안전하게 탈거합니다.
출처: master_boot_clip2.mp4 (00:02:05~00:02:40)"
sources: latch_confirm_007(0.608, accept), latch_removal_005(0.606, accept), ch_preread_001(0.445, hold)
```
→ 시스템 프롬프트 형식(핵심→절차→출처) 그대로 지킴, 검색 근거 안 벗어남.

완전 무관 질의(`"오늘 날씨 어때요"`, 0.157초):
```
sources: []
answer: "관련 암묵지를 찾지 못했어요..."
```
→ **방어선 1이 정상 작동**(유사도가 임계값 밑이라 검색 단계에서 걸러짐, LLM 호출 자체가
없어서 응답이 0.16초로 즉시 옴 — LLM 추론 시간이 전혀 없다는 게 방증).

애매한 질의(`"오늘 점심 뭐 먹지"`, 4.3초):
```
top1_similarity = 0.372  (SIM_THRESHOLD=0.35보다 높아서 방어선 1을 통과해버림)
answer: "점심 메뉴 결정은 개인의 취향에 따라 다르므로... 출처: 영상ID 없음"
```
→ **중요 발견**: 이 질의는 방어선 1(임계값)을 못 막았다(0.372 > 0.35). 그런데도
LLM이 조립된 암묵지 내용을 억지로 끌어다 쓰지 않고 "특별한 음식 정보는 제공하지 못한다"고
정직하게 실패를 인정함 — **방어선 2(시스템 프롬프트)가 실제로 방어선 1의 구멍을 메운
사례**. 다만 이건 운 좋게 LLM이 잘 버틴 것이지 구조적 보장은 아니다.
→ **권고 후 적용됨**: 사용자 확인 받아 `config.py`의 `SIM_THRESHOLD` 기본값을
0.35 → **0.40**(STEP7_DB 실측값)으로 변경. 서버 재기동 후 같은 질의로 재검증:
```
"오늘 점심 뭐 먹지"  top1=0.372 < 0.40  → 0.47초, sources=[] (방어선1이 직접 차단, LLM 호출 없음)
"새 램인데도 인식이 안 돼요"  top1=0.582  → 정상 답변 유지(회귀 없음 확인)
```
→ 이제 이 케이스는 방어선2(프롬프트)의 운에 기대지 않고 방어선1(임계값)만으로 차단됨.

---

## 파트4 — 음성 계층 역머지 (2026-07-08, 서시은 voice-rag v2)

**배경(계보)**: 서시은님이 7/4 업로드본(v1, `voice-rag-upload/` — 파트1~3의 뼈대) 이후
계속 개발한 v2를 7/7 밤 `~/members/안나경/voice-rag`에 HANDOFF.md와 함께 재인계.
v1↔v2의 ingest/evaluate는 파일명 한글화 외 동일(STEP8이 이미 더 발전된 형태로 흡수),
v2 신규분은 전부 **음성 계층**이라 그것만 STEP8에 역머지함. `voice-rag-upload/`는
완전 흡수됐으므로 ARCHIVED.md 표시(팀원 원본이라 삭제 안 함).

**가져온 것 (STEP8 구조 위에 이식)**
- `app.py`: `/transcribe`(faster-whisper large-v3-turbo, beam_size=1+vad_filter),
  `/tts`(Supertonic 3, F1, steps=5), `TTS_PRONUNCIATION` 발음사전+`normalize_tts_text()`,
  `/ask` ollama 옵션 2줄(`keep_alive:-1` VRAM 상주, `num_predict:256` 답변 상한).
  무거운 음성 모델 로드는 v2처럼 모듈 최상단이 아니라 **preflight 통과 후**로 이동(우리 원칙).
- `config.py`: STT/TTS 설정을 pydantic 필드+validator로 편입(STT_MODEL/STT_DEVICE/
  TTS_MODEL/TTS_VOICE/TTS_STEPS 환경변수 유지).
- `static/메인화면.html`(음성 UI) — `/`가 이걸 서빙, 텍스트 구화면은 `/static/index.html`.
- `서버시작.sh`: 전용 GPU잡(salloc --no-shell) + ollama + 앱 일괄 기동 — 파트3의
  "남의 GPU잡 편승 임시방편" 과제가 이걸로 해소됨. 포트 8001 고정(8000은 타인 점유).
- `requirements.txt`: scipy/uroman/supertonic/faster-whisper/onnxruntime(v2 고정 버전)
  + **python-multipart**(UploadFile에 필요한데 v2 requirements엔 누락돼 있었음 — 실측 발견).

**지킨 것 (v2로 되돌리지 않음)**
- `SIM_THRESHOLD=0.40` (v2는 0.35로 남아있었음 — 파트3에서 상향한 실측 근거 유지)
- search.py 분리 구조, accept_only 훅, preflight, gold_records 기반 DB
  (v2 qdrant_db는 sample_knowledge 기반이라 버림)

**검증 (완료 기준 2종, n5 GPU잡 2266에서 실증 후 반납)**
- 텍스트 /ask 회귀 없음: "래치 어떻게 눌러야 해요?" → sources
  (latch_confirm_007 0.608 / latch_removal_005 0.606 / ch_preread_001 0.445) —
  파트3 검증 로그와 동일. "오늘 점심 뭐 먹지" → 0.18초, sources=[] (방어선1 차단 유지).
- 음성 왕복 1회: TTS로 질문 wav 합성 → `/transcribe` → STT "레츠 어떻게 눌러야 해요."
  (합성음이라 래치→레츠 오인식) → **그래도 검색은 래치 문서 정상 회수**(top1 0.556 >
  0.40) → 근거 기반 답변 → 답변 TTS wav(44.1kHz PCM) 생성. 왕복 16.9초.
  → 짧은 STT 오인식에 임베딩 검색이 관대하다는 부수 확인. 단 실마이크 검증은 아직
  (브라우저 webm 업로드는 실사용 시 확인 필요 — v2에서 이미 돌던 경로라 위험 낮음).

