# STEP7(암묵지 DB) 상세 기술 보고서

> 목적 (앞 보고서들과 동일)
> 1. **재디벨롭 참고 문서** — 실험 설계·결과·적재 스키마·손잡이를 바로 찾는다.
> 2. **트러블슈팅 아카이브** — 측정 방식 교정·버전 분화·백엔드 전환의 사건 기록.
>
> **작성 규칙(엄수):** 코드/results 파일을 실제로 따라가 확인된 것만. **숫자는 전부 results 파일
> 출처 명기, 기억 재구성 금지.** 사실/추론 구분("추정:"), 확인 불가는 "**(확인 불가)**" 후 §9 재수집.
> **코드 수정 없음.** 작성 기준 2026-07-16, 커밋 `45be634`.

> **STEP7은 두 덩어리다.** [1부] Vector DB 선정 실험(`STEP1_모델연구/06_DB/01_VectorDB비교/`),
> [2부] 실제 적재 파이프라인(`STEP7_DB/`). 명확히 나눠 서술하며, 2부의 미구현/미결정은
> "현황"으로 정직하게 기록한다.

---

## §0. 산출물 위치 맵 (먼저 확정)

| 덩어리 | 위치 | 핵심 파일 |
|---|---|---|
| [1부] 실험 | `STEP1_모델연구/06_DB/01_VectorDB비교/` | `benchmark.py`(v1), `benchmark_v3.py`, `benchmark_v4.py`; `results.csv`(v1)/`results_v3.csv`/`results_v4.csv`; `slide_table*.md`; `chunks.json`, `embeddings.npy`, `questions.csv`, `docker-compose.yml`, `*.log`, `*.sbatch` |
| [2부] 적재 | `STEP7_DB/` | `build_db.py`, `vectordb_utils.py`, `config.py`, `search_tacit.py`, `query_tacit.py`, `preflight.py`; `chroma_db/`(실 DB), `qdrant_db/`(롤백 보존), `gold_records/`(예시 9건), `input_motion_obs_run6_accept/`(실 6건) |
| 문서 | `docs/` | `_STEP7_8_심층분석_보고서.md`(git 1177c7a), 계획서·raw 사본은 `06_DB/.../data/raw/` |

---

# [1부] Vector DB 선정 실험

## §1. 실험 설계 요약

### 1-1. 비교 대상·모드 (사실, benchmark.py/benchmark_v4.py)

- **3종**: Chroma / FAISS / Qdrant.
- **실행 모드(실측 코드로 확정)**:
  - Chroma = `PersistentClient`(디스크 영속), HNSW cosine(benchmark_v4.py:build, vectordb_utils.py:160-165).
  - FAISS = Flat(IndexFlatIP류, brute-force) — 규모 커질수록 O(n) 선형(§2 역전 근거).
  - Qdrant = **embedded(로컬 파일) 모드**. **Docker 서버 아님** — "이 클러스터는 docker/sudo가
    없어 localhost:6333 서버를 못 띄운다. 6333 접속 실패 시 embedded로 폴백"(benchmark.py:9-15,
    docker-compose.yml:1-2). 실배포(STEP7/8)와 동일 모드로 측정(results_v4.csv `Qdrant(embedded)`).
- **(확인 불가)** 서버 모드(Docker) 실측치 — 환경상 실행 불가라 측정 안 됨. "서버 모드는 별도 재측정
  필요"로 명시(slide_table_v4.md 하단 주석).

### 1-2. 임베딩·재현성 원칙 (사실)

- **임베딩 모델 = BAAI/bge-m3, 1024차원, CPU, HF 오프라인 캐시**(benchmark.py:6-8). STEP7/8 실적재와
  동일 모델.
- **"임베딩 1회 계산 → 세 DB 동일 주입" 구현 확인**: 실데이터 실험(v1)은 임베딩을 1회 계산해
  `embeddings.npy`+`chunks.json`에 캐시하고 3 DB에 동일 주입(benchmark.py:22 "임베딩은 1회 계산 후
  … 캐시, L2 정규화, 3 DB 동일 주입"). → DB 간 품질 차이는 인덱싱/검색만의 함수임이 설계로 보장.
- **cosine 통일**: 전 DB cosine(L2 정규화 벡터, benchmark_v4.py:35-37, vectordb_utils.py:165).
- **시드 42**: `SEED=42`(benchmark.py:44, benchmark_v4.py:26), random/numpy 고정.

### 1-3. 데이터셋 실체 (사실, 직접 확인)

- **품질 실험(v1)**: 실문서 **46건**(`data/raw/*.md` 파일 수 = 46, 직접 카운트) → **951 청크**
  (`chunks.json` 길이 = 951, 직접 카운트). **500자 청크, overlap 100**(benchmark.py:19). 청크 키 =
  `{id, text, category, year}`; 메타데이터 category(파일명 규칙), year(mtime)(benchmark.py:19-21).
  청크 ID = `<상대경로>#<3자리 순번>`(questions.csv 정답청크ID가 이걸 가리킴, benchmark.py:22).
  **질문 15건**(questions.csv 16행 = 헤더+15).
- **규모/성능 실험(v3·v4)**: **합성 데이터** — `np.random.default_rng(SEED)`로 1024차원 정규분포
  벡터를 정규화해 생성(benchmark_v4.py:31-38). 규모 100/1,000/10,000(+100,000 별도).
- **⚠️ 대표성 구분(사실)**: **품질(Recall)은 실데이터**(bge-m3 951청크), **성능(latency/insert/qps)은
  합성 벡터**(차원만 1024로 동일, 의미분포는 무작위). latency·throughput은 의미와 무관해 합성으로도
  대표성 있으나, Recall 수치는 실데이터 15질문 기준임을 혼동하지 말 것.

### 1-4. 측정 프로토콜 (사실, benchmark_v4.py:1-20)

- **워밍업 10회**(`N_WARMUP=10`) 후 **측정 쿼리 100회**, 단일 스레드.
- p50 / p95 보고(v4). **p99는 미측정**(results_v4.csv 컬럼에 없음 — 사용자 언급 p99는 확인 불가).
- **프로세스 격리 측정**: build 프로세스(삽입·persist)와 measure 프로세스(cold start·latency·qps)를
  **분리**, DB×규모 조합마다 새 프로세스(benchmark_v4.py:12-16). Cold Start = 새 프로세스 디스크
  로드→첫 검색 성공까지.

**근거 위치(§1):** `benchmark.py:1-44`, `benchmark_v4.py:1-38`, `docker-compose.yml:1-14`,
`chunks.json`(951), `data/raw/*.md`(46), `questions.csv`(15Q), `slide_table_v4.md` 주석.

---

## §2. 실측 결과 전수 정리 (results 파일에서 직접 추출)

> **버전 정리(사실)**: 결과가 세 파일에 나뉜다 — **품질(Recall)=`results.csv`(v1)**,
> **메모리=`results_v3.csv`(v3)**, **규모별 성능=`results_v4.csv`(v4, 최신)**. 단일 파일에 전부
> 있지 않다. 아래 각 표에 출처를 명기한다. `results_v4_3scales_backup.csv`=100k 전 3규모 백업,
> `results_run1_backup.csv`=v1 백업.

### 2-1. 품질 — Recall@1/5/10 (출처: results.csv, v1, 실데이터 951청크·15질문)

직접 재계산(results.csv exp A hit@k 합/15):

| DB | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| Chroma | 8/15 = **0.533** | 11/15 = **0.733** | 13/15 = **0.867** |
| FAISS | 8/15 = **0.533** | 11/15 = **0.733** | 13/15 = **0.867** |
| Qdrant(embedded) | 8/15 = **0.533** | 11/15 = **0.733** | 13/15 = **0.867** |

- **세 DB 완전 동일**(slide_table.md:12-14와 일치). **"K 선택과 무관하게 성립" 논리(사실)**: @1/@5/@10
  모두 세 DB가 같으므로, 어느 K를 봐도 품질 순위가 갈리지 않는다 — §1-2의 "동일 임베딩 동일 주입"
  덕에 검색 결과 랭킹이 DB 간 동일(정규화 cosine + 같은 벡터 → 같은 이웃). 품질은 선정 변별력이 없음.

### 2-2. 성능 — 규모별 (출처: results_v4.csv, 합성)

| 지표(규모) | Chroma | FAISS | Qdrant(embedded) |
|---|---|---|---|
| Insert(s) 100 / 1k / 10k / 100k | 2.49 / 3.15 / 17.84 / 215.7 | 0.07 / 0.08 / 0.17 / 0.96 | 3.13 / 10.43 / 83.94 / 830.72 |
| p50(ms) 100 / 1k / 10k / 100k | 2.43 / 2.97 / 3.97 / 4.43 | 0.03 / 0.23 / 6.22 / 60.85 | 5.26 / 6.8 / 57.6 / 534.66 |
| p95(ms) 100 / 1k / 10k / 100k | 2.57 / 2.99 / 4.1 / 4.79 | 0.04 / 0.24 / 6.33 / 62.11 | 5.32 / 6.91 / 58.02 / 539.35 |
| Cold Start(s) 100 / 1k / 10k / 100k | 2.087 / 2.078 / 2.116 / 2.714 | 0.061 / 0.066 / 0.1 / 0.433 | 2.254 / 2.646 / 6.793 / 49.281 |
| QPS 100 / 1k / 10k / 100k | 409.4 / 340.7 / 252.4 / 224.7 | 30055.7 / 4293.9 / 160.8 / 16.4 | 189.7 / 146.7 / 17.3 / 1.9 |

- **인덱스 효과 분기점(사실, 확인됨)**: latency p95에서 **10,000건에서 Chroma가 FAISS를 역전**한다 —
  Chroma 4.1ms < FAISS 6.33ms(10k), 100k에서 격차 확대(Chroma 4.79ms << FAISS 62.11ms). 100·1,000건에선
  FAISS가 압도적(brute-force가 소규모에선 빠름). **추정:** FAISS Flat은 O(n) 선형탐색, Chroma HNSW는
  O(log n) → 규모가 커질수록 HNSW 우위. QPS도 같은 교차(10k에서 Chroma 252 > FAISS 160).
- **Qdrant(embedded)는 전 규모에서 최하위 성능**: 10k p95 57.6ms(Chroma의 14배), 100k 534ms. insert도
  최느림(100k 830s). → embedded Qdrant(순수 파이썬)의 구조적 한계(config.py:34 "embedded Qdrant보다
  검색 p95 10배+ 우위" 실측 근거).

### 2-3. 메모리 (출처: results_v3.csv, 1,000건, 프로세스 RSS 증가분)

| | Chroma | FAISS | Qdrant(embedded) |
|---|---|---|---|
| RAM(MB) | 73.4 | **18.9** | 116.6 |
| Cold Start(s) | 1.701 | 0.066 | 2.407 |
| QPS | 295.9 | 4235.9 | 120.8 |
| p95(ms) | 3.58 | 0.24 | 11.48 |

- FAISS가 메모리 최소(18.9MB), Qdrant 최대(116.6MB). (v3는 1,000건 단일 규모만.)

### 2-4. 필터 검색 latency (출처: results.csv exp B)

| 필터 p95(ms) | Chroma | FAISS | Qdrant(embedded) |
|---|---|---|---|
| 1,000 | 5.42 | 0.55 | 26.93 |
| 10,000 | 31.07 | 22.32 | 302.58 |
| 100,000 | 263.16 | 1076.79 | 3578.93 |

- **FAISS 필터 직접 구현 비용(사실)**: `A,FAISS,951,filter_impl_lines,,,11`(results.csv) — FAISS는
  네이티브 메타데이터 필터가 없어 **직접 구현 11줄**이 필요했고, 100k에서 필터 p95가 1076ms로
  post-filter 비용이 폭증(브루트포스 후 필터). Chroma는 263ms.
- **QPS/동시성**: 단일 스레드 QPS만 측정(§2-2). 다중 동시성(concurrent) 실험은 results 파일에 없음
  (확인 불가).

**근거 위치(§2):** `results.csv`(Recall·filter·exp B), `results_v3.csv`(메모리), `results_v4.csv`(규모별),
`slide_table.md:12-15`, `slide_table_v4.md`.

---

## §3. 선정 논리와 최종 결정

### 3-1. ⚠️ 문서 결론과 구현이 불일치한다 (크게 표시)

| 소스 | 선정 DB | 근거 |
|---|---|---|
| **v1 실험 슬라이드**(slide_table.md:15) | **Qdrant** | "검색 품질 동등, 속도 차이 ms 단위 → 필터 표현력(FAISS 직접 구현)과 프로덕션 기능·무변경 서버 전환 경로 기준 **Qdrant 선정**" |
| **실제 구현**(STEP7_DB/config.py:38) | **Chroma** | `vector_backend="chroma"`(기본), qdrant는 롤백 |
| **STEP7 README**(README.md:3-10,86-93) | **Qdrant**로 서술 | 갱신 안 됨(stale) |
| **vectordb_utils.py docstring**(:3-4) | **Qdrant**로 서술 | 갱신 안 됨(stale) |

- **사실**: **최종 구현 = Chroma**. 전환 근거는 config.py:32-34 —
  > "2026-07-12 전환(지시): Qdrant(embedded) → ChromaDB(PersistentClient). 근거: 실측 — 동일 임베딩에서
  > Recall 완전 동일, **1만 건 이상에서 Chroma(HNSW)가 embedded Qdrant보다 검색 p95 10배+ 우위**."
  이는 §2-2 v4 실측(10k p95 Chroma 4.1 vs Qdrant 57.6)과 일치.
- **불일치의 실체(사실)**: v1 슬라이드의 "Qdrant 선정"은 **embedded Qdrant의 성능 열위가 드러나기
  전(v4 이전)** 결론이고, 서버 모드 전환 경로를 전제했다. 그러나 서버 모드는 이 클러스터에서 실행
  불가(§1-1)라 embedded로 갈 수밖에 없었고, embedded Qdrant는 §2-2에서 최하위 → **실측이 결론을
  뒤집어 Chroma로 전환**됐다. **README·docstring·slide_table.md는 이 전환을 반영하지 못한 stale 문서다.**
- **추정:** 발표/문서 산출물(slide_table.md, README)이 코드 전환(2026-07-12)보다 먼저 작성돼 갱신
  누락. 재디벨롭 시 "문서는 Qdrant, 코드는 Chroma"의 혼동 위험 — 문서 갱신이 필요(§8).

### 3-2. FAISS 제외 논리 (최종 교정본, 사실)

- slide_table.md 결론 + config 근거를 종합하면 FAISS 제외 사유는 **"Persistence 안 됨"이 아니라
  "라이브러리 vs DB"**다: FAISS는 인덱스 라이브러리라 **메타데이터·CRUD·영속성을 직접 구현**해야 한다.
  실측으로 그 비용이 드러남 — **필터 직접 구현 11줄**(results.csv `filter_impl_lines=11`)과 **100k
  필터 p95 1076ms**(§2-4, post-filter 폭증). 즉 성능(소규모 최速)은 좋으나 **DB로서의 기능(필터·CRUD·
  영속·메타데이터)을 스스로 떠안는 비용** 때문에 제외.

### 3-3. 예상 반론 3종 문서화 여부 (사실)

- **"Chroma가 낫지 않냐"**: **현재는 Chroma가 실제 선정됨**(config). v1 슬라이드는 Qdrant였으나
  실측 역전으로 Chroma 채택 → 이 반론은 사실상 **채택됨**(§3-1).
- **"왜 서버 모드로 안 쟀냐"**: 문서화됨 — docker/sudo 없어 서버 실행 불가, embedded 폴백,
  "서버 모드 별도 재측정 필요"(benchmark.py:9-15, docker-compose.yml:1-2, slide_table_v4.md 주석).
- **"왜 @5냐"**: v1은 실험 A를 "Recall@5"로 명명(benchmark.py:5)하되 실제로는 @1/@5/@10 전부
  측정(results.csv). **(확인 불가)** "@5를 대표로 쓴 이유"를 명시한 반론 답변 문구는 slide/README에서
  못 찾음 — @1/@5/@10 동시 제시로 갈음.

**근거 위치(§3):** `slide_table.md:15`, `config.py:32-38`, `README.md:3-10`, `vectordb_utils.py:3-4`,
`results.csv`(filter_impl_lines), `benchmark.py:5,9-15`.

---

## §4. 온프레미스·보안 제약의 구현 확인

- **외부 API 완전 금지 / 로컬 임베딩(사실)**: 임베딩은 `HuggingFaceEmbeddings(model_name="BAAI/bge-m3")`
  로컬 로드(vectordb_utils.py:51-56, benchmark.py:6-8 "HF 캐시 오프라인 로드"). LLM 없음(STEP7은 검색만,
  답변 조립은 템플릿 — search_tacit.py:73-74 "LLM 재작성 없음"). API 호출 코드 없음.
- **텔레메트리 차단(사실, 한 줄씩 검증)**:
  - Chroma: `Settings(anonymized_telemetry=False)` — 실적재(vectordb_utils.py:162), 벤치(benchmark_v4.py),
    STEP6/STEP8 동일.
  - Qdrant embedded: 로컬 파일 모드라 텔레메트리 원천 없음(benchmark.py:11). 서버 모드용
    `docker-compose.yml:12` `QDRANT__TELEMETRY_DISABLED=true` 명시(직접 확인).
- **폐쇄망 반입 절차(사실, 부분)**: `docker-compose.yml:6` 주석 "폐쇄망이면 미리 이미지 반입
  (docker save/load)". **(확인 불가)** pip wheel 반입 절차의 별도 문서화는 이 폴더에서 못 찾음
  (requirements.txt는 있으나 오프라인 반입 스크립트/절차 문서 미확인).

**근거 위치(§4):** `vectordb_utils.py:51-56,162`, `benchmark.py:6-11`, `docker-compose.yml:6,12`,
`search_tacit.py:73-74`.

---

# [2부] 적재 파이프라인 (현황 그대로)

## §5. 적재 구현 현황

### 5-1. STEP6 출력 → DB 적재 전 과정 (사실)

- **진입점**: `build_db.py` → preflight → `vectordb_utils.build_or_load_vectorstore`. 원샷은
  `파이프라인_통합실행/main.py`가 step7_adapter → build_db 호출(STEP6 §2-5).
- **적재 대상 판정**: **현재 routing으로 안 거른다.** config.py:20-23 —
  > "지금은 STEP6 routing으로 거르지 않고 input_dir 안 JSON을 전부 적재한다(VLM이 막혀 있어
  > gold_records 예시로 파이프라인 자체를 먼저 만드는 단계). routing 값은 payload에 저장만."

  `accept_only_default=False`(config.py:56), 기본 `input_dir=gold_records`(9건 예시). accept 필터
  훅(`ChromaVectorStoreAdapter.accept_filter`={routing:"accept"})은 **존재하나 기본 미가동**.
- **임베딩 대상 텍스트(사실, D1)**: `"[상황] {situation}\n[노하우] {tacit_insight}\n[이유] {reasoning}"`
  **합본**(vectordb_utils.py:112). tacit_insight 단독이 아니다 — 단독은 증상 질의 유사도가 낮았음
  (실측 0.27, vectordb_utils.py:108-109). **`diagnostic_steps.action`은 임베딩 제외**(절차 나열 노이즈),
  payload로만 보관(vectordb_utils.py:110-111).

### 5-2. 컬렉션 스키마 (payload, 사실 — vectordb_utils.py:87-107)

| payload 필드 | 출처 | 임베딩? |
|---|---|---|
| `id` | data["id"] | (point ID = uuid5) |
| `schema_version` | data | 아니오 |
| `source_file` | 파일명 | 아니오 |
| `equipment`, `task`, `keywords`, `scenario_id`, `scenario_title` | metadata | 아니오 |
| `video_id`, `clip_start`, `clip_end` | metadata.source | 아니오 |
| `situation`, `tacit_insight`, `reasoning` | knowledge | **예(합본)** |
| `reasoning_origin` | knowledge | 아니오 |
| `diagnostic_steps` | knowledge | 아니오(payload만) |
| `routing`, `confidence` | verification | 아니오(있으면 저장, 필터엔 미사용) |

- **Chroma 스칼라 제약 대응(사실)**: Chroma metadata는 스칼라만 허용 → 전체 payload를 `payload_json`
  한 필드에 JSON 문자열로 저장 + 필터용 `routing`만 평탄 필드 병행(vectordb_utils.py:171-173). 검색 시
  `payload_json` 역직렬화로 dict 복원(하류는 Qdrant 때와 동일 dict, vectordb_utils.py:151-152).
- **거리→유사도**: Chroma cosine `distance` → `1 - distance`(vectordb_utils.py:152). 임계값 0.40 그대로.

### 5-3. 임베딩 모델·차원 — 1부와 동일한가 (사실)

- **동일**: 실적재 = BAAI/bge-m3, 1024차원(vectordb_utils.py:52-53). v1 품질 실험도 동일 모델·차원
  (§1-2). → 품질 실험의 대표성 유지. (성능 실험만 합성 벡터, §1-3.)

### 5-4. ID 체계·중복 방지·영속·백업 (사실)

- **ID**: `uuid5(고정 네임스페이스, tacit_id)`(vectordb_utils.py:59-60). 재적재 시 같은 tacit_id → 같은
  point ID → **upsert로 덮어씀**(중복 방지, vectordb_utils.py:22-23,167).
- **영속 위치**: `STEP7_DB/chroma_db/`(PersistentClient, config.py:41). qdrant_db/도 존재(롤백 보존).
- **백업(사실/현황)**: 명시적 백업 로직 없음. chroma_db/ 디스크 폴더가 곧 영속체(git 제외, README.md:93).
  **(확인 불가)** 정기 백업/스냅샷 절차 문서 미확인.

### 5-5. 미구현/미결정 목록 (재디벨롭 착수 지점)

1. **routing 기반 적재 필터 미가동** — accept/hold/reject 구분 없이 전량 적재(config.py:20-23,56).
   accept_only 훅은 구현돼 있으나 기본 off. "VLM 버그 수정 후 켤 자리(D5)".
2. **기본 input_dir = gold_records(예시 9건)** — 실 STEP6 산출이 아닌 예시. 실데이터는
   `input_motion_obs_run6_accept/`(6건)에 있으나 **이 파일들엔 `verification` 블록이 없다**
   (직접 확인: verification=None) → 적재 시 routing="none"으로 저장됨(폴더명은 accept인데 태그 부재).
3. **schema_version 드리프트** — config/README/docstring은 "1.3"이라 하나 실제 후보는 **1.4**(STEP5).
4. **hold 후보 편입 경로 미정** — 전량 적재라 지금은 hold도 들어가나 accept와 payload routing으로만
   구분. 전문가 검증 후 승격 파이프라인 없음.
5. **README·vectordb_utils docstring이 Qdrant로 stale** — 실 구현 Chroma와 불일치(§3-1).
6. **백업/스냅샷 절차 미문서화**(§5-4).

**근거 위치(§5):** `build_db.py`, `vectordb_utils.py:59-116,119-175,204-218`, `config.py:20-56`,
`input_motion_obs_run6_accept/*.json`(verification 부재), `README.md:93`.

---

## §6. 설계 판단 질문 (CLI 분석 — 사실/추정 구분)

### 6-1. 적재 단위 = 후보 1건 = 벡터 1개, 청킹 없음

- **사실**: `build_or_load_vectorstore`가 "documents를 **chunk 없이** upsert"(vectordb_utils.py:204).
  후보 1건 → 합본 텍스트 1개 → 벡터 1개.
- **추정(평가)**: 적절하다. 암묵지 후보는 situation+insight+reasoning 합쳐도 짧은 구조화 문서라
  청킹하면 오히려 한 지식이 쪼개져 검색 단위가 흐트러진다. 현 규모(수~수십 건)에선 청킹 불필요.
  단 diagnostic_steps를 임베딩에서 뺀 것(§5-1)은 절차 검색이 필요해지면 재고 대상(추정).

### 6-2. hold 후보 취급

- **사실**: 현재 전량 적재(routing 필터 off, §5-5#1). payload에 routing 보존은 되나 accept/hold가
  검색에서 구분되지 않는다(기본). STEP8이 `accept_only`로 필터하면 그때 구분(search.py:7).
- **추정**: 전문가 검증 후 hold→accept 편입 경로는 **미구현**. accept_only 훅이 준비만 돼 있어,
  라우팅을 켜고 재적재(upsert)하면 태그가 갱신되는 구조는 가능하나 그 워크플로우가 코드로 없음.

### 6-3. STEP8(RAG)의 검색 요구를 감당하는 스키마인가 (사실 — 교차 대조)

- STEP8 존재함(`STEP8_RAG서비스/`: app.py, search.py, ingest.py 등). STEP8이 읽는 필드(app.py:154-156,
  search.py): `routing`, `entry["knowledge"]["tacit_insight"]`, 그리고 STEP7 payload의
  reasoning/diagnostic_steps/clip_*/video_id(search_tacit.py:73-91과 동형 템플릿).
- **감당 여부**: STEP8이 요구하는 필드가 전부 STEP7 payload에 있다(§5-2) → **스키마상 감당 가능**.
  top_k=3(config.py:49), routing 필터(accept_only), SIM_THRESHOLD 0.40 공유(config.py:54, STEP8도 동일
  0.40 — 메모리 [[project_step7_step8_rag]]). STEP8도 chroma 기본/qdrant 롤백으로 STEP7과 백엔드 정렬
  (search.py:26,30).
- **추정**: 필터 패턴이 routing(accept) 단일 축이라 단순하다. 향후 equipment/task별 필터가 필요하면
  Chroma where에 평탄 필드 추가가 필요(현재 payload_json 안에 묻혀 있어 필터 불가 — routing만 평탄).

**근거 위치(§6):** `vectordb_utils.py:112,204`, `config.py:49,54,56`, `search.py:7,26-32`,
`app.py:154-156`, `search_tacit.py:73-91`.

---

## §7. 트러블슈팅 아카이브

### 7-1. 측정 방식 교정 — 서버 vs embedded, 프로세스 격리

- **증상/원인**: Qdrant를 서버 모드로 재려 했으나 클러스터에 docker/sudo 없음 → 서버 기동 불가.
  또 in-process 측정은 캐시·JIT 워밍 상태가 섞여 cold start를 못 잰다.
- **해결**: (a)6333 접속 시도 후 실패 시 **embedded 자동 폴백**(실배포와 동일 모드, benchmark.py:9-15).
  (b)**build/measure 프로세스 분리**로 cold start를 정직하게 측정(benchmark_v4.py:12-16).
- **남은 장치**: results에 `mode`(in-process/embedded) 컬럼 기록(results.csv). slide_table_v4.md에
  "서버 모드 별도 재측정 필요" 주석.
- **재발 시**: Qdrant 서버 재측정은 docker 가능한 환경에서만. mode 컬럼으로 어느 모드 측정인지 확인.

### 7-2. 결과 파일 버전이 v4까지 간 이유 (사실)

- v1(`results.csv`): 품질 Recall + 필터(exp A/B) — 실데이터. 슬라이드 결론 Qdrant.
- v3(`results_v3.csv`): **메모리(RSS) 추가** — 1,000건 단일, 프로세스 격리 RSS 증가분.
- v4(`results_v4.csv`): **규모별 성능(100/1k/10k, +100k 별도)** 재측정 — 프로세스 격리·cold start
  정식화. 2026-07-12 지시로 "10만 취소→100 추가"(benchmark_v4.py:24)했으나 100k는 별도 sbatch
  (`vdbbench_v4_100k.sbatch`, 로그 `..._100k_2627.log`)로 측정해 CSV에 병합. `results_v4_3scales_backup.csv`
  = 100k 전 백업.
- **추정**: 지표를 한 번에 못 정하고 (품질→메모리→규모별 성능+cold start) 점진 확장하며 버전이 늘었다.
  **최신·대표 성능표 = v4**, 품질은 v1, 메모리는 v3이 각각 authoritative.

### 7-3. FAISS 필터 직접 구현 비용 실측 (사실)

- **증상**: FAISS엔 네이티브 메타데이터 필터 없음.
- **원인**: 라이브러리라 필터를 앱단에서 구현 → post-filter(브루트포스 후 걸러냄).
- **실측**: 직접 구현 11줄(results.csv `filter_impl_lines=11`), 100k 필터 p95 1076ms(§2-4).
- **결론**: FAISS 제외의 핵심 근거(§3-2).

### 7-4. Qdrant → Chroma 전환 (사실)

- **증상/원인**: embedded Qdrant가 1만 건+에서 검색 p95 최하위(§2-2, 57.6ms vs Chroma 4.1ms).
- **해결**: 2026-07-12 config `vector_backend="chroma"`로 전환, qdrant_db/·코드 롤백 보존. 거리→유사도
  역변환 어댑터로 임계값 0.40 그대로 유지(vectordb_utils.py:119-153).
- **남은 장치**: `vector_backend="qdrant"` 한 줄로 롤백 가능(config.py:38).
- **재발 시**: 백엔드 전환 시 SIM_THRESHOLD 의미(cosine)가 두 백엔드 동일한지 확인(실측 대조됨).

### 7-5. SIM_THRESHOLD 0.40 상향 (사실)

- **증상**: 처음 0.30이었는데 무관한 질의("오늘 점심 뭐 먹지")도 bge-m3 baseline 0.30~0.36으로 안 걸러짐.
- **해결**: 무관(0.30~0.36) vs 관련(0.44+) 갭 실측 → **0.40으로 상향**(config.py:50-54). 미만이면 "못
  찾았다"로 정직 실패.

**근거 위치(§7):** `benchmark.py:9-15`, `benchmark_v4.py:12-24`, `results.csv`, `results_v3.csv`,
`results_v4.csv`, `vdbbench_v4_100k.sbatch`, `config.py:32-54`, `vectordb_utils.py:119-153`.

---

## §8. 재디벨롭 가이드

### 8-1. 실험 재현 커맨드

```bash
cd project-ai/STEP1_모델연구/06_DB/01_VectorDB비교
# 전체 벤치(품질+필터): STEP7 venv 사용
../../../STEP7_DB/.venv/bin/python3 benchmark.py            # 본 실험(46문서·951청크)
../../../STEP7_DB/.venv/bin/python3 benchmark.py --dry-run  # 문서 10개·1k만(빠른 점검)
# 규모별 성능(v4): 프로세스 격리
python3 benchmark_v4.py --n-query 100                       # 100/1k/10k
sbatch vdbbench_v4_100k.sbatch                              # 100k 별도
```

### 8-2. 적재 재실행 커맨드

```bash
cd project-ai/STEP7_DB
.venv/bin/python3 build_db.py                               # gold_records(예시 9건)
.venv/bin/python3 build_db.py --input_dir <STEP6 accept 폴더>   # 실데이터
.venv/bin/python3 build_db.py --query "래치 두 개 동시에"     # 적재 후 검색 스모크
# 원샷(STEP3~7): 파이프라인_통합실행/main.py (step7_adapter → build_db)
```

### 8-3. §5-5 미구현 기반 착수 순서 (제안)

1. **문서 갱신**(즉시): README·vectordb_utils docstring·slide_table.md의 Qdrant→Chroma, 1.3→1.4 정정
   (혼동 위험 제거, 비용 낮음).
2. **실데이터 적재 경로 확정**: `input_motion_obs_run6_accept/`에 verification 부재 → step7_adapter를
   거쳐 routing 태그가 붙은 입력을 쓰도록 통일(원샷 main.py 경로).
3. **routing 필터 가동**: accept_only 훅 켜기(config.accept_only_default 또는 STEP8 필터). accept/hold 구분.
4. **hold 승격 워크플로우**: 전문가 검증 결과를 upsert로 반영하는 경로 신설.
5. **백업 절차**: chroma_db/ 스냅샷 정책 문서화.

### 8-4. 재검토 트리거 (문서에 남긴 것)

- **서버 모드 재측정**: "docker 가능 환경에서 Qdrant 서버 모드 별도 재측정 필요"(slide_table_v4.md 주석).
- **규모 확대 시 DB 재검토**: v4가 100k까지 커버 — 그 이상/실시간 QPS 요구 시 재측정(현 DB 수십 건이라
  당면 이슈 아님, benchmark_v4.py:24).
- **RAG top_k 상향**: 현재 top_k=3(config.py:49). 상향 검토 근거·기준은 이 폴더에서 명시 문구 미확인
  (확인 불가) — STEP8 evaluate.py/문서 참조 필요.

### 8-5. 건드리면 위험한 불변식

1. **임베딩 모델-인덱스 결합.** 임베딩 모델(bge-m3 1024)을 바꾸면 저장된 벡터와 질의 벡터가 다른
   공간이 돼 검색이 무의미 → **모델 변경 시 전체 재적재 필수**. (STEP7·STEP8·벤치가 전부 bge-m3여야.)
2. **온프레미스 제약.** 외부 API(임베딩/LLM) 유입 금지 — 로컬 모델만. 텔레메트리 off 유지(§4).
3. **거리↔유사도 변환 + 임계값.** Chroma distance→(1-distance), 임계값 0.40은 cosine 기준. 백엔드
   교체 시 이 변환이 유지돼야 임계값 의미 불변(vectordb_utils.py:152).
4. **point ID = uuid5(tacit_id).** ID 규칙을 바꾸면 upsert 중복 방지가 깨져 재적재 시 중복 벡터 발생.

**근거 위치(§8):** `benchmark.py`, `benchmark_v4.py:24`, `build_db.py`, `config.py:38,49`,
`vectordb_utils.py:59-60,152`, `slide_table_v4.md` 주석.

---

## §9. "확인 불가" / 미구현 / 사실·추론 구분

### 9-1. 확인 불가 항목

1. **Qdrant 서버 모드 성능** — 환경상 실행 불가라 미측정(embedded만).
2. **p99 latency, 다중 동시성(concurrent) QPS** — results 파일에 없음(p50/p95, 단일 스레드만).
3. **"@5를 대표로 쓴 이유"의 반론 답변 문구** — slide/README에서 명시 문구 못 찾음(@1/@5/@10 동시 제시로 갈음).
4. **pip wheel 폐쇄망 반입 절차 문서** — Docker 이미지 반입은 언급, wheel 절차는 미확인.
5. **정기 백업/스냅샷 절차** — 미문서화.
6. **RAG top_k 상향 검토 근거 위치** — STEP7 폴더엔 없음(STEP8 쪽 확인 필요).
7. **`input_motion_obs_run6_accept/` 6건의 생성 경위** — verification 부재 상태로 어떻게 이 폴더에
   놓였는지(수동 큐레이션 추정) 코드로 확정 불가.

### 9-2. §5 미구현 목록 (재수집)

routing 적재 필터 미가동 / 기본 input=예시(gold_records) / 실데이터 폴더에 verification 부재 /
schema_version 1.3↔1.4 드리프트 / hold 승격 경로 없음 / README·docstring stale(Qdrant) / 백업 절차 미문서화.

### 9-3. §6 사실 / 추론 구분

**사실:** 적재 단위=후보1건=벡터1개, 청킹 없음(vectordb_utils.py:204) / hold 현재 전량 적재, routing
필터 off / STEP8이 읽는 필드가 STEP7 payload에 전부 존재(감당 가능) / top_k=3·SIM 0.40 공유.
**추론/추정:** 청킹 불필요가 "적절"하다는 평가 / diagnostic_steps 임베딩 제외의 재고 필요성 / hold
승격 워크플로우 부재 / equipment·task 필터 확장 시 평탄 필드 추가 필요 / 버전이 v4까지 간 것은
지표 점진 확장 때문 / v1 슬라이드가 코드 전환보다 먼저 작성돼 stale.

---

*(끝) 본 보고서는 조사·설명·평가 전용이며 코드를 수정하지 않았다.*
