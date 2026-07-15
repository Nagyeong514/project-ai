# 파이프라인 구조 & 실행 가이드

영상(정비공 작업 클립) → **암묵지 후보 생성 → 품질검증 → DB 적재 → RAG 서비스**로 이어지는
전체 파이프라인의 **실행 단위 지도**와 실행법. "뭘 돌리면 전체가 도느냐"의 정답 문서.

---

## 1. 한눈에 — 뭐가 같이 돌고 어디서 갈라지나

```
┌──────────────────────────────────────────────────────────────┐
│  통합러너 (이 폴더: 파이프라인_통합실행/)                       │
│  ── 한 프로세스 · venv 1개 공유 · 모델 한 번 로드해 재사용 ──   │
│  venv: 파이프라인_통합실행/.venv                               │
│                                                                │
│    STEP3 전처리    →   STEP4 YOLO+VLM   →   STEP5 LLM 융합      │
│    (STT/프레임/        (객체탐지 +          (암묵지 후보 생성)  │
│     모션샘플링)         VLM 관찰)                              │
└──────────────────────────────────────────────────────────────┘
         │  파일 전달: STEP5.../output/tacit_json/*.tacit.json
         ▼   (+ step6_adapter 로 STEP6 입력 형식 변환)
┌──────────────────────────────────────────────────────────────┐
│  STEP6 품질검증       ── 독립 · 자기 venv ──                   │
│  venv: STEP6_암묵지_품질검증/.venv                             │
│  LangGraph + 판정LLM + bge-m3 + ChromaDB(Gate A 매뉴얼 RAG)    │
│  → accept / hold / reject                                      │
└──────────────────────────────────────────────────────────────┘
         │  step7_adapter(decision→routing, confidence→{score}) — main.py [4/4] 자동
         ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP7 DB 적재        ── 독립 · 자기 venv ──                   │
│  venv: STEP7_DB/.venv · bge-m3 + ChromaDB(tacit_knowledge)     │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP8 RAG 서비스     ── 독립 · 자기 venv · 웹 서비스(상주) ──  │
│  venv: STEP8_RAG서비스/.venv · app.py                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 실행 단위 · venv 경계

| 구간 | 실행 단위 | venv | 성격 |
|---|---|---|---|
| **STEP3 + 4 + 5** | **함께** (통합러너 1프로세스) | `파이프라인_통합실행/.venv` **공유** | 배치 GPU 파이프라인 |
| STEP6 | 따로 | `STEP6_암묵지_품질검증/.venv` | 배치 판정 |
| STEP7 | 따로 | `STEP7_DB/.venv` | 배치 적재 |
| STEP8 | 따로 (상주 서비스) | `STEP8_RAG서비스/.venv` | 웹 서비스 |

### 왜 이렇게 나뉘나
- **STEP3·4·5 = 한 덩어리**: 원래 `STEP3_전처리` 하나였다가 2026-07-04에 3개로 **분할**됐지만,
  GPU 한 프로세스가 STT→YOLO→VLM→LLM을 **순차 로딩**하도록 설계된 하나의 파이프라인.
  그래서 **venv를 하나만** 쓰고 모델을 한 번 로드해 재사용한다. (각자 단독 실행도 가능 — §4)
- **STEP6·7·8 = 독립 배포 서비스**: langchain/transformers 버전이 서로 충돌하고 모델을 동시에
  다 못 올려서 **venv를 각각 분리**(CLAUDE.md 규칙). 프로세스를 합치지 않는다.
- **단계 사이 데이터는 항상 파일로 주고받는다** (모델 재로딩 최소화 + 느슨한 결합).

---

## 3. 실행법

### 원샷 (STEP3→4→5→adapter→STEP6→step7_adapter→STEP7 DB적재) — 권장 진입점
```bash
cd 파이프라인_통합실행
srun -p RTX6000 -w n5 --gres=gpu:2 .venv/bin/python3 main.py
#  옵션:  --run-tag <출력태그>   --judge-model Qwen/Qwen2.5-14B-Instruct(판정자 격리)
#         --skip-db(STEP6까지만)  --accept-only-db(DB에 accept만 적재)
#  SLURM 배치:  sbatch run_e2e_step3to6.sbatch   (RUN_TAG=myrun 지정 가능)
```
`main.py` 가 각 단계를 그 단계의 venv 파이썬으로 subprocess 호출하고, 단계마다 산출물 개수가
0건이면 즉시 중단한다(빈 결과로 조용히 진행하는 사고 방지). **STEP7(DB 적재)까지 자동**이며,
STEP8(웹 서비스)만 별도 기동.

### 통합러너만 (STEP3→4→5, 후보 생성까지)
```bash
srun ... .venv/bin/python3 run_all_clips.py                 # 4클립 일괄(VIDEOS 고정)
srun ... .venv/bin/python3 run_one_clip.py --video <경로>   # 클립 1개
```

### 이후 단계는 각 폴더에서 따로
```bash
# STEP6 품질검증
STEP6_암묵지_품질검증/.venv/bin/python3 STEP6_암묵지_품질검증/main.py --input <adapter_out> --output_dir <out>
# STEP7 DB 적재
STEP7_DB/.venv/bin/python3 STEP7_DB/build_db.py --input_dir <폴더>
```

### STEP8 RAG 서비스 — 단독 기동 (배치 아님 · 상주 서버)
STEP8은 파이프라인의 일부가 아니라 **질의응답 웹 서비스**다. 아래 한 파일이 GPU 할당 →
ollama(qwen2.5:14b) 기동 → uvicorn 앱(포트 8001)까지 한 번에 띄운다:
```bash
bash STEP8_RAG서비스/서버시작.sh [GPU노드]      # 기본 n5
#  → VS Code 포트탭에서 8001 포워딩 후 http://localhost:8001
#  종료:  fuser -k 8001/tcp && scancel <JOBID>
```
⚠️ 포트 **8001**(8000은 타 사용자 점유). main.py 파이프라인이 STEP7까지 DB를 채운 뒤 기동한다.
각 STEP은 자기 폴더 안에서 단독 실행도 되지만, STEP3/4/5의 단독 러너(`run_step3/4/5.py`)는
통합러너와 **같은 `파이프라인_통합실행/.venv`** 를 쓴다.

---

## 4. 데이터 흐름 (파일 핸드오프)

`config.yaml` 의 `paths:` 에 정의된 중간 산출물 경로(프로젝트 루트 기준 상대경로):

| 산출물 | 경로 |
|---|---|
| STEP3 프레임 | `STEP3_전처리/output/_frames` |
| STEP3 전사(STT) | `STEP3_전처리/transcripts` |
| STEP4 검출 | `STEP4_YOLO_VLM관찰/output/detections` |
| STEP4 VLM 관찰 | `STEP4_YOLO_VLM관찰/output/vlm_observations` |
| STEP5 정렬 윈도우 | `STEP5_LLM_암묵지_후보생성/output/aligned_windows` |
| **STEP5 암묵지 후보** | `STEP5_LLM_암묵지_후보생성/output/tacit_json` |
| STEP5→6 변환 입력 | `STEP5_LLM_암묵지_후보생성/output/step6_adapter_input/<tag>` |
| STEP6 판정 결과 | `STEP6_암묵지_품질검증/output/<tag>` |

---

## 5. 범위 & 알려진 공백

- **`main.py` 포함 범위 = STEP3~7.** (후보 생성 → 판정 → 벡터DB 적재까지 원샷)
- **STEP6→STEP7 스키마 어댑터 = `STEP6_암묵지_품질검증/step7_adapter.py`** (2026-07-14 신설):
  STEP6 출력(`verification.decision` / `confidence: float`)을 STEP7 로더가 읽는 형식
  (`verification.routing` / `confidence.score`)으로 변환한다. main.py [4/4] 에서 자동 호출.
  `--accept-only-db` 로 accept만 적재하거나, 기본은 전건 적재 + routing 태그 보존(STEP7이 검색 시 필터).
- **STEP8은 상주 웹 서비스**라 배치 체인 대상이 아님 — 파이프라인 종료 후 별도 기동:
  **`bash STEP8_RAG서비스/서버시작.sh`** (GPU 할당→ollama→uvicorn 8001 한 번에). 상세는 §3.

---

## 6. 주의사항 (이 클러스터 / 함정)

- **GPU는 srun 으로만**: 로그인 노드엔 GPU 없음. `srun -p RTX6000 -w n5 --gres=gpu:2 ...` 로 실행.
- **무거운 모델 로드 전 preflight**: `preflight.py` 가 5초 안에 죽을 검증(패키지/경로/GPU)을 먼저
  수행한다. 위 진입점들은 시작 시 이를 호출한다.
- **폴더명 드리프트 주의**: 통합러너·어댑터는 STEP 폴더명을 경로로 참조한다. 폴더명이 바뀌면
  `config.yaml`(step5 경로 2줄)과 `step6_adapter.py` 를 같이 고쳐야 한다. (실제로 STEP5가
  `LLM으로_암묵지…`→`LLM_암묵지…`로 바뀌었을 때 이 두 곳이 드리프트해 빈 결과가 났던 이력 있음
  — 2026-07-14 수정.)
- **로그**: 실행 기록은 `run_logs/` 에 남는다(`.sbatch` + `_<jobid>.log`).
