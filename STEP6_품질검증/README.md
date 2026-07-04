# 6단계 품질 검증(Quality Verification) 파이프라인

파이프라인 계획 문서의 **6. 품질 검증** 단계를 구현한 코드
암묵지 후보 JSON(step 5의 output)을 입력받아 **accept / hold / reject** 중 하나로 자동 라우팅한다.
(후보 JSON의 내용을 생성/수정하지 않고, 판정만 한다)


## 사용 기술 (전부 오픈소스, API 미사용)
LLM (judge) : **Qwen2.5-14B-Instruct** (HuggingFace transformers, 로컬)
임베딩 : **BAAI/bge-m3** (langchain-huggingface) 
벡터DB : **Qdrant** (로컬이지만 서버로 전환 가능) 
RAG : **LangChain** (text splitter, vector store)
에이전트 : **LangGraph** (StateGraph, conditional edge)

## 폴더 구조

```
quality_verification/
├── config.py            # 경로/모델/가중치/threshold 등 전체 설정
├── schemas.py            # 암묵지 후보 JSON pydantic 스키마 (문서 5단계 스키마)
├── rules.py               # 결정론적 탈락조건(timestamp_validity) + 마커 규칙(utterance_signal 규칙부분)
├── embedding_utils.py     # BGE-M3 임베딩 + Qdrant 매뉴얼 RAG (Gate A)
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
[1단계] Gate A: Manual RAG 검색 (Qdrant + BGE-M3)
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
- `embedding_utils.py`: `qdrant_location`을 로컬 디스크 경로나 서버 URL로 바꾸면 영구 저장 가능
- 여러 장비/시나리오를 다룬다면 `manuals/`에 장비별 매뉴얼 파일을 여러 개 넣으면 됨
  (metadata에 저장된 파일명이 `source` 메타데이터로 남아 어떤 매뉴얼에서 검색됐는지 추적 가능)
