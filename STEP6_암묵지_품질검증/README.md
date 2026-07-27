# 6단계 품질 검증(Quality Verification) 파이프라인

파이프라인 계획 문서의 **6. 품질 검증** 단계를 구현한 코드
암묵지 후보 JSON(step 5의 output)을 입력받아 **accept / hold / reject** 중 하나로 자동 라우팅한다.
(후보 JSON의 내용을 생성/수정하지 않고, 판정만 한다)


## 사용 기술 (전부 오픈소스, API 미사용)
LLM (judge) : **Qwen2.5-14B-Instruct** (HuggingFace transformers, 로컬)
임베딩 : **BAAI/bge-m3** (langchain-huggingface) 
벡터DB : **ChromaDB** (in-memory, 매 실행 재색인 — 2026-07-14 Qdrant에서 전환, STEP7/8 표준 통일) 
RAG : **LangChain** (text splitter, vector store)
에이전트 : **LangGraph** (StateGraph, conditional edge)

## 폴더 구조

```
quality_verification/
├── config.py            # 경로/모델/가중치/threshold 등 전체 설정
├── schemas.py            # 암묵지 후보 JSON pydantic 스키마 (문서 5단계 스키마)
├── rules.py               # 결정론적 탈락조건(timestamp_validity) + 마커 규칙(utterance_signal 규칙부분)
├── embedding_utils.py     # BGE-M3 임베딩 + ChromaDB 매뉴얼 RAG (Gate A)
├── llm_utils.py           # Qwen2.5-14B-Instruct 로더 + 게이트별 judge 프롬프트/파서
├── graph.py               # LangGraph StateGraph (0~4단계 노드 + 조건분기)
├── main.py                # 실행 진입점 (CLI)
├── requirements.txt
├── manuals/               # Gate A(Manual RAG)용 매뉴얼 원문 (.txt/.md) — 반드시 1개 이상 필요
│   └── dell_precision_7920_manual.txt   (예시용 더미 매뉴얼, 실제 매뉴얼로 교체 필요)
├── input/                 # 검증할 암묵지 후보 JSON을 여기에 넣음
│   ├── tk_dell7920_mem_boot_001.json    (예시 candidate)
│   └── transcripts/
│       └── transcript_example.json      (예시 transcript, candidate가 transcript_ref로 참조)
└── output/                # 실행 결과가 저장되는 폴더 (자동 생성)
```


## 실행 방법

### 1) 패키지 설치
```bash
pip install -r requirements.txt
```
Qwen2.5-14B-Instruct, BAAI/bge-m3 모델은 최초 실행 시 HuggingFace Hub에서 자동 다운로드됨.


### 2) 입력 준비
- `input/` 폴더에 검증하려는 암묵지 후보 JSON(5단계 스키마)을 넣고!
- 각 JSON의 `metadata.source.transcript_ref` 가 가리키는 transcript 파일도 실제 위치에 준비하고!
  (candidate JSON 기준 상대경로 / `input/` 기준 상대경로 / 절대경로 모두 지원)
- `manuals/` 폴더의 더미 매뉴얼을 **실제 시나리오 매뉴얼**로 교체하고!


### 3) 실행

경로만 넣으면 바로 실행됨 (기본값 = `input/` 폴더 전체 처리):
```bash
python main.py
```

특정 파일/폴더 지정:
```bash
python main.py --input /path/to/candidate.json
python main.py --input /path/to/dir --output_dir /path/to/out
```

모델 다운로드 없이 파이프라인 로직/그래프 흐름만 오프라인으로 확인하고 싶을 때:
```bash
python main.py --mock
```


### 4) 결과 확인
후보 1건당 `output/` 에 2개 파일이 생성됨
- `<id>_result.txt` : 0~4단계 **중간 결과 전부 + 최종 결과**를 상세히 기록한 로그
  (timestamp 체크 항목별 결과, Gate A 검색쿼리/유사도/재검색 여부, same/delta/novel 판정과 근거,
  Gate B/C 각 점수와 근거, confidence 계산식, 최종 accept/hold/reject)
- `<id>_verified.json` : 원본 후보 JSON + `verification` 블록 추가본 (문서 6-8 대응)



## 파이프라인 흐름

```
[0단계] timestamp_validity (Rule, 탈락조건)
   └─ FAIL → reject (STOP)
[1단계] Gate A: Manual RAG 검색 (ChromaDB + BGE-M3)
   └─ top1 유사도 < threshold → 키워드 재구성(tacit_insight로) 후 1회 재검색
[2단계] Gate A: LLM Judge → same / delta / novel (탈락조건)
   └─ same → reject (STOP)
[3단계] Gate B, C 점수 항목 (LLM Judge, 서로 독립)
   - action_reason_consistency
   - reasoning_grounding
   - step_grounding_ratio (스텝별 grounded 여부 LLM 판정 → 비율)
   - utterance_signal (인과/이탈/주의/부정 마커 규칙기반 탐지 + LLM 보정)
[4단계] confidence = Σ(weight × score) → T_high/T_low 비교 → accept / hold / reject
```

- action_only(발화 없는) 스텝은 발화를 전제하는 모든 검사에서 면제됨 (설계원칙 4, `rules.py` 참고).
- 가중치(`weight_a`)와 threshold(`t_high_a`, `t_low_a`)는 문서 6-5/6-6의 **트랙 A(라벨 0~20건)** 기본값임.
  라벨이 50건 이상 쌓이면 `config.py`의 `track = "B"`로 바꾸고, 새로 학습한 로지스틱 회귀 계수로
  `weight_b`, `t_high_b`, `t_low_b`를 갱신됨 (트랙B)



## 커스터마이징 가능 요소
- `config.py`: 모델명, device, threshold, 가중치, 마커 리스트 전부 여기서 조정
- `llm_utils.py`: 프롬프트 문구 및 할루시네이션 방지 지시문(`HALLUCINATION_GUARD`) 수정
- `embedding_utils.py`: 기본 ChromaDB(in-memory). `config.vector_backend="qdrant"`로 롤백 가능(코드 보존)
- 여러 장비/시나리오를 다룬다면 `manuals/`에 장비별 매뉴얼 파일을 여러 개 넣으면 됨
  (metadata에 저장된 파일명이 `source` 메타데이터로 남아 어떤 매뉴얼에서 검색됐는지 추적 가능)

## 매뉴얼 코퍼스(Gate A) 변경 이력
- **2026-07-14 (ESC 오탐 진단 — 코퍼스 무변경 결론)** — run8에서 CLIP4_004 "ESC 키로 BIOS 메뉴 이동"이
  accept 1.0으로 새는 오탐 발생. 진단 결과 **코퍼스 문제 아님**: 매뉴얼 §2-5에 "Esc 키를 누르면 현재
  화면을 닫고 이전 메뉴로 돌아간다"가 이미 명시돼 있고, run6·run8 모두 동일 청크를 동일 점수
  (top1=0.523)로 검색함. 차이는 순수 **판정자**뿐 — Qwen2.5(run6)=`same`으로 정확히 reject,
  Qwen3(run8)=`delta`로 오탐(자명한 결과를 새 정보로 과대해석). → **코퍼스 추가 안 함**(§2-5에 이미
  있어 중복 padding이 되며 원인(판정자)도 못 고침). 검증: run9에서 판정자만 Qwen2.5로 되돌려 재평가
  → `output/motion_obs_run9_qwen25judge/`. `STEP6_LLM_MODEL` env로 판정자 오버라이드 추가(config.py).
- **2026-07-14** — `manuals/dell_precision_7920_manual.txt`에 **§2-8 "메모리 인식 상태(용량·채널)
  확인 — 표준 마무리 검증 절차"** 항목 추가.
  - 배경: 기존 매뉴얼은 메모리 용량 인식 확인을 §2-6·§5-5)·§8에 흩어 서술만 해두어, Gate A RAG가
    "BIOS에서 용량·채널 인식값을 확인" 계열 후보(motion_obs run6의 CLIP4_006 용량 확인, CLIP4_007
    채널 확인)를 매뉴얼에 이미 규정된 표준 절차로 인식하지 못하고 delta/novel로 통과시켰다
    (run7 대조: 006=hold/delta, 007=accept/delta). 채널 인식 확인 및 "채널 저하 인식 시 대역폭 절반
    성능 저하" 연관성은 아예 명문화돼 있지 않았다.
  - 조치: 용량·채널 인식 확인을 조립·교체 후 **표준 마무리 검증 절차**로 §2-8에 명시(채널 저하 시
    대역폭 절반 성능 저하 근거 포함). 이는 임의 조작이 아니라 매뉴얼에 이미 부분 규정돼 있던 표준
    절차의 명문화·통합이다.
  - 검증: 코퍼스 갱신 후 후보 24건 전체를 판정모델 Qwen3-14B로 재평가하여 CLIP4_006·007이
    Gate A `same`으로 라우팅되는지 확인 → 출력 `output/motion_obs_run8_manualaug/`.
    run6(6/4/14, Qwen2.5 판정)·run7(Qwen3+옛 매뉴얼)은 기준 보존.
  - ⚠️ 격리 주의: run8은 아래 벡터DB 전환과 **동시에** 반영됐다. run8 vs run7은 (매뉴얼 + 벡터DB)
    두 변수가 함께 달라진 것이므로, 순수 매뉴얼 효과만 격리하려면 별도 대조(Chroma+옛 매뉴얼)가
    필요하다(STEP7 실측상 Chroma≈Qdrant cosine 점수라 매뉴얼 효과가 지배적일 것으로 추정).

- **2026-07-14** — 벡터DB **Qdrant → ChromaDB 전환**(`config.vector_backend="chroma"`, in-memory
  EphemeralClient). STEP7_DB·STEP8_RAG서비스가 이미 Chroma로 전환돼 있어 Gate A 매뉴얼 RAG도
  팀 표준에 통일. Chroma는 `distance=1-cos_sim`을 반환하므로 `ChromaManualAdapter`가 `(1-distance)`로
  유사도 역변환(STEP7과 동일 패턴) → `manual_similarity_threshold=0.45` 그대로 유효. 롤백은
  `vector_backend="qdrant"`(코드 보존). chromadb는 `.venv`에 직접 설치(1.5.9, n5 시스템 미보유).
