# project-ai — 작업 영상에서 암묵지를 뽑아 검색 가능한 지식으로 만드는 파이프라인

숙련자(명장)가 스마트글래스로 찍은 PC 수리 영상 1개를 넣으면,
**"말로는 잘 안 적히는 노하우"를 구조화된 JSON 후보로 만들고 → 자동 품질검증으로 걸러 → 벡터DB에 적재해 → 신입이 음성으로 물어보면 답하는** 전 과정을 8단계로 구현한 프로젝트입니다.

```
[입력] 스마트글래스 영상 1개 (화면 + 음성)
   │
   ├─ 음성 ──→ STT(Whisper) ──→ 전사 정제(용어 정규화·반복 환각 태깅) ──┐
   │                                                                   │
   └─ 화면 ──→ 프레임 추출 ──→ YOLO(부품 위치) ──→ VLM(행동 관찰) ──────┤
                                                                       │
                                        ±4초 윈도우로 타임스탬프 정렬 ←┘
                                                   ↓
                            LLM 융합(Qwen2.5-14B) → 암묵지 후보 JSON (v1.3)
                                                   ↓
                         품질검증(LangGraph 4게이트) → accept / hold / reject
                                                   ↓
                        임베딩(bge-m3) → Qdrant 적재 → 음성 RAG 질의응답
```

**설계의 뼈대 세 줄**
1. **VLM은 "눈"이다** — 보이는 것만 관찰하고, 해석·진단·묶기는 전부 LLM이 한다.
2. **타임스탬프가 접착제다** — 음성과 영상은 끝까지 따로 가다가 시각으로만 만난다.
3. **생성과 판별을 분리한다** — STEP3~5는 후보를 만들기만 하고, 진짜인지는 STEP6이 판정한다.

### 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 기간 | [확인 필요] |
| 팀 구성 / 담당 범위 | [확인 필요] — STEP8은 `voice-rag-upload`(뼈대)와 `STEP7_DB`(방어 로직) 두 사람의 작업을 통합한 것으로 코드 주석에 기록돼 있음 |
| 실행 환경 | SLURM 클러스터, Quadro RTX 6000 (Turing sm75) ×2, 폐쇄망(전 모델 로컬 구동, 외부 API 미사용) |
| 데이터 | Dell Precision 7920 조립·부팅 트러블슈팅 촬영분 CLIP1~4 (영상은 저장소에 미포함) |
| 라이선스 | [확인 필요] |

---

## 실행 결과 샘플

### 입력
`CLIP1_정상조립과정.mp4` (198초, Dell Precision 7920 메모리 조립/부팅 트러블슈팅, 발화 비율 8.2%)

### 단계별 산출물 (실제 실행 로그 기준)

| 단계 | 산출물 | 실측 예시 |
|---|---|---|
| STEP3 | `transcripts/<video_id>.json` | 발화 세그먼트 + 정제 플래그 |
| STEP4 | `output/detections/*.json`, `output/vlm_observations/*.observations.json` | YOLO **검출 162건** → VLM **관찰 61건** |
| STEP5 | `output/tacit_json/<video_id>.tacit.json` | **암묵지 후보 1~2건** |
| STEP6 | `<id>_verified.json`, `<id>_result.txt` | `confidence=0.5050` → **HOLD** |
| STEP7 | Qdrant 컬렉션 `tacit_knowledge` | 9건 적재, top1 유사도 0.44~0.61 |
| STEP8 | `POST /ask` JSON 응답 | 아래 참고 |

> 관찰/후보 건수는 클립·설정에 따라 편차가 큽니다(같은 CLIP1에서 관찰 34~61건). 회차별 값은 `파이프라인_통합실행/run_logs/`에 원본 로그가 있습니다.

### 최종 암묵지 JSON (schema v1.3) — 실제 레코드

```json
{
  "id": "tk_dell7920_led_read_003",
  "schema_version": "1.3",
  "metadata": {
    "equipment": "Dell Precision 7920",
    "task": "메모리 부팅 트러블슈팅",
    "keywords": ["LED", "점멸 패턴", "POST", "RAM", "진단"],
    "source": { "video_id": "master_boot_clip2.mp4", "clip_start": "00:00:00", "clip_end": "00:02:40" }
  },
  "knowledge": {
    "situation": "부팅 실패 후 전원 LED로 원인 계열을 진단하는 상황",
    "tacit_insight": "전원 LED 점멸 패턴을 끝까지 읽어 원인을 RAM/CPU 계통으로 즉시 좁힌다. 초보자는 패턴을 봐도 의미를 해석하지 못한다.",
    "reasoning": "LED 점멸은 POST 진단 코드라 패턴만으로 원인 서브시스템을 특정할 수 있다.",
    "reasoning_origin": "utterance",
    "diagnostic_steps": [
      {
        "order": 1,
        "action": "전원 LED 점멸 패턴 판독(황색1·백색3)으로 원인 계열 좁히기",
        "evidence": "utterance",
        "source_utterance": "황색 한 번, 백색 세 번. 이 패턴이면 RAM 쪽이야.",
        "timestamp": "00:00:20"
      }
    ]
  },
  "verification": {
    "gate_a": { "relation": "delta", "top1_similarity": 0.66 },
    "confidence": { "score": 0.926 },
    "routing": "accept"
  }
}
```

핵심 필드 두 개가 이 스키마의 존재 이유입니다.
- `evidence: utterance | action_only` — **말은 안 했지만 몸으로만 한 행동**을 따로 표시합니다. 암묵지는 주로 여기 있습니다.
- `reasoning_origin: utterance | model_inferred` — LLM이 자기 일반지식으로 채운 설명을 **발화 근거인 척 위장하지 못하게** 강제 태깅합니다.

### STEP6 품질검증 판정 로그 (실제 출력 발췌)

```
[0단계] timestamp_validity … => PASS
    - diagnostic_step[4] @ 00:06:10: OK (action_only -> 발화 실재 검사 면제)
[1단계] Gate A - Manual RAG 검색   1차 top1 유사도: 0.681 (임계값 0.45)
[2단계] Gate A - LLM Judge         relation = delta
[3단계] action_reason_consistency = 0.400 / reasoning_grounding = 1.000
        step_grounding_ratio = 0.250 (1/4 grounded) / utterance_signal = 0.000
[4단계] confidence = 0.35*1.000 + 0.3*0.250 + 0.2*0.400 + 0.15*0.000 = 0.5050
        threshold: T_high=0.7, T_low=0.4  => 최종 결정: HOLD
```

judge가 왜 그렇게 매겼는지도 항목마다 근거 문장이 함께 남습니다. 이 후보는 `step_grounding_ratio`가 낮아(LLM이 발화에 없는 절차를 덧붙임) hold로 빠졌습니다.

### STEP8 음성 RAG 응답 (실측)

정상 질의 — `"래치 어떻게 눌러야 해요?"` (GPU 추론 14.3초)

```
answer: "래치는 양쪽을 동시에 눌러야 합니다. 한쪽만 누르면 슬롯 핀이 파손될 수 있습니다.
        1. 양 손의 엄지와 검지를 사용해 메모리 모듈 양 옆 래치를 잡습니다.
        2. 양쪽 래치를 동시에 눌러서 메모리를 안전하게 탈거합니다.
        출처: master_boot_clip2.mp4 (00:02:05~00:02:40)"
sources: latch_confirm_007(0.608, accept), latch_removal_005(0.606, accept), ch_preread_001(0.445, hold)
```

무관 질의 — `"오늘 날씨 어때요"` (0.157초)

```
sources: []
answer: "관련 암묵지를 찾지 못했어요. 부품명이나 증상을 포함해 다시 질문해주시겠어요?"
```

0.157초라는 응답 시간 자체가 **LLM을 아예 호출하지 않았다는 증거**입니다. 유사도 임계값(0.40) 미만이면 검색 단계에서 끊고 LLM에 넘기지 않습니다.

---

## 왜 이 모델을 골랐나 (대안과 실측 근거)

모델은 전부 사전 비교 실험을 거쳐 골랐고, 원자료는 `00_사전연구/`, `STEP2_모델선정/`, `_archive_pre_server/참고_방법론/`에 있습니다.

### STT: Whisper large-v3-turbo

| 후보 | 결과 |
|---|---|
| **Whisper large-v3-turbo** | **CLIP1~4 전부 승리** (선정룰: RTF≤0.10 통과 후 CER 최소) |
| Seamless-M4T-v2-Large | CER은 개선 가능하나 `num_beams=5` 탓에 RTF 0.21~0.30 → 속도 게이트 탈락 |
| Wav2Vec2-XLSR-Korean | RTF는 최고(0.006)지만 CER이 turbo보다 나쁨(CLIP1 0.437 vs 0.215) |

**VAD 적용 여부가 모델 선택보다 영향이 컸습니다** — Whisper 기준 CER 평균 0.5241 → 0.2362(**-54.9%**). 다만 threshold를 0.5→0.2로 낮추는 건 -2.1%로 사실상 무의미해서 기본값을 유지했습니다.

여기서 결론이 한 번 뒤집힌 기록도 남겨뒀습니다. 초기 VAD는 발화 구간을 **잘라 이어붙이는** 방식이었는데, 이러면 오디오가 짧아져 Whisper가 주는 타임스탬프가 원본 영상 시각과 어긋납니다. 이 파이프라인은 타임스탬프로 영상·음성을 정렬하므로 이 오차가 곧 정렬 오류가 됩니다. faster-whisper 내장 `vad_filter=True`(오디오를 자르지 않고 무음만 스킵, 시각은 원본 기준 유지)로 바꾸니 CER도 더 좋았습니다(평균 0.1813). **정확도만 보고 고르면 파이프라인이 깨지는 사례**여서 문서에 남겼습니다.

### VLM: Qwen3-VL-8B-Instruct

동일 입력(같은 영상·구간·전사·프레임·프롬프트)으로 비교했습니다.

| 지표 | **Qwen3-VL-8B** | InternVL2.5-8B |
|---|---|---|
| JSON 파싱 성공 | **10/10** | 9/10 |
| 총 추출 후보 수 | **35개** | 12개 |
| 개념 recall (프레임+말) | **0.667** | 0.167 |

MiniCPM-V-2.6(int4)도 후보였으나 환경 마찰로 보류했습니다. 정량·정성 모두 Qwen 우세라 하드웨어 타협이 아닌 선택이었습니다.

recall 채점에서 "영상만" 조건이 양쪽 다 낮게 나왔는데(0.167 / 0.000), 이건 모델 문제가 아니라 **정답지를 STT 전사에서 도출해 영상에만 있는 비언어 노하우가 정답지에 없어서**입니다. 모델 간 비교에만 쓰고 영상 기여도 해석에는 쓰지 않았습니다.

> ⚠️ 출처 구분: 위 수치는 `_archive_pre_server/참고_방법론/05_VLM암묵지추출_MAIN/VLM_모델비교_결과보고서.md`(초기 비교)의 값입니다. 이후 `STEP2_모델선정/`에서 더 최신 버전(InternVL3.5 / MiniCPM4.5)으로 재비교를 돌린 **원자료 JSON은 있으나 집계 보고서가 저장소에 없습니다**(`compare_results.py`를 실행하면 생성됨). 즉 현재 채택 근거로 제시할 수 있는 건 초기 비교 쪽입니다.

### 융합 LLM: Qwen2.5-14B-Instruct

골든셋 30개에 대해 동일 조건(같은 프롬프트·glossary·temperature·seed·양자화)으로 생성 → 6항목 루브릭 채점 → 대응표본 t-검정.

| 모델 | Final 평균 | 충실도(환각 억제) | 한국어 |
|---|---|---|---|
| **Qwen2.5-14B** | **4.743** | **4.93** | **4.87** |
| Llama-3.1-8B | 3.833 | 3.93 | 3.40 |

평균차 +0.910, **p<0.0001**, 샘플별 승패 28승 2패. Llama의 주 실패 패턴은 ① 용어집에 있는 단어를 입력에 없는데 근거로 끌어옴 ② 인과 역전("볼트를 *풀면* 좌표 재설정" — 원문은 조임) ③ 전사 복붙이었습니다. 후속 검증을 통과해 지식 DB를 오염시키는 게 가장 큰 리스크라 **충실도에 최대 가중(0.25)** 을 뒀습니다.

단, Qwen이 진 2건도 기록했습니다 — 장문·다조건 입력에서 중국어 혼입·코드펜스로 출력 형식이 무너진 사례가 있어, 출력 안정성은 프롬프트·파서 쪽에서 따로 방어했습니다.

### 임베딩: BAAI/bge-m3 — 임베딩 대상을 바꾼 게 모델보다 중요했음

`tacit_insight` 단독으로 임베딩했을 때 신입의 증상 질의("화면이 안 떠요")와 유사도가 0.27대로 낮았습니다. `situation + tacit_insight + reasoning` 합본으로 바꾸자 top1 유사도가 0.44~0.61대로 올라갔습니다. Recall@3으로도 재현됐습니다.

| 임베딩 대상 | Recall@3 |
|---|---|
| 합본(situation+insight+reasoning+steps+keywords) | **9/9 = 1.00** |
| tacit_insight 단독 | 8/9 = 0.89 |

`diagnostic_steps.action`은 절차 나열이라 노이즈가 커서 임베딩엔 넣지 않고 payload로만 보관합니다.

---

## 인프라 제약이 설계를 바꾼 지점

GPU가 **Quadro RTX 6000 (Turing, sm75) ×2** 인 SLURM 클러스터(sudo 없음)입니다. 이 제약이 코드에 그대로 반영돼 있습니다.

| 제약 | 대응 |
|---|---|
| Turing은 bf16·FP8 미지원 | 전 모델 **fp16 / 4bit(NF4)** 고정 |
| **vLLM 사용 불가** (python3.12-devel 없음 + sudo 없음 → Turing용 TRITON_ATTN JIT 컴파일 실패) | LLM 서빙을 `transformers` + `bitsandbytes(NF4)` 직접 로딩으로 전환. `requirements.txt`의 `vllm` 줄은 주석 처리 상태 유지 |
| 단일 프로세스에 VLM+LLM 동시 적재 불가 | VLM 추론 완료 → `unload()` → LLM 로드 순차 실행 |
| ultralytics가 `select_device()`에서 `CUDA_VISIBLE_DEVICES`를 덮어써 VLM이 GPU 1장에 갇힘 | YOLO를 **별도 서브프로세스**(`yolo_subprocess_entry.py`)로 격리 → VLM이 `device_map="auto"`로 2장(47GB)을 실제로 확보 |
| VLM 전체 영상 1회 생성 시 자기복제 루프·타임스탬프 날조·관찰 0건 동결·KV캐시 OOM | 영상을 **40초 청크로 분할**해 청크마다 `generate()` 별도 실행 |
| 로그인 노드와 GPU 노드의 pip 패키지가 다름(실제로 `pyparsing` 누락으로 YOLO가 조용히 검출 0건) | venv에 직접 고정 설치 + preflight에서 실제 GPU 노드 기준 검사 |

### 실행 전 방어 체크리스트 (이 저장소의 공통 규칙)

> **무거운 것(모델 로드) 위로는 전부 5초 안에 죽을 수 있는 검증만, 무거운 것 아래로 내려간 에러만 진짜 GPU/로직 문제로 취급한다.**

`langgraph` 미설치 상태로 GPU 잡을 띄워 할당 시간을 날린 사고(최근 3일 SLURM 잡 151건 중 FAILED 56건 분석) 이후 만든 규칙입니다. 문서로만 두면 안 지켜지므로 **코드로 강제**했습니다.

- 각 STEP에 `preflight.py` — ① import 체크 ② `sys.executable` 로깅 ③ pydantic config 검증 ④ 경로/GPU 확인 순서로 실행
- 진입점 첫 줄에서 `run_preflight()` 강제 호출
- **무거운 라이브러리(torch/transformers/ultralytics/langgraph 등)를 모듈 최상단에서 import하지 않음** — preflight가 통과한 뒤로 미룸
- 프리플라이트는 고정 리스트가 아니라 **config에서 실제로 선택된 impl/backend에 필요한 패키지만** 검사 (안 쓰는 backend의 미설치를 오탐하지 않기 위해)
- 스모크 테스트는 "에러 없이 끝남"으로 통과 처리하지 않고 결과 개수·값까지 `assert` (exit 0으로 조용히 빈 결과를 내는 게 가장 위험한 실패)

상세: [`docs/실행전_방어_체크리스트.md`](docs/실행전_방어_체크리스트.md)

---

## 환각을 막는 장치들

품질보다 "지어내지 않는 것"을 우선한 설계라, 단계마다 방어가 들어가 있습니다.

| 위치 | 장치 |
|---|---|
| VLM 프롬프트 | 해석·진단·묶기 금지 / 추상동사("확인","점검") 금지 → 구체 동작만 / 측정값 추정 금지 / LED는 색+횟수만, 의미 해석 금지(진단 코드표는 LLM만 봄) |
| VLM 파서 | JSON 파싱 실패 시 brace 매칭으로 완성된 관찰만 부분 복구 → 그래도 실패하면 원문을 관찰 1건으로 폴백. **완전 실패하지 않음** |
| 타임스탬프 | 프레임 시각은 모델이 아니라 `i/fps` 그리드에서 **코드가** 부여. VLM이 뱉은 시각은 `snap_to_grid()`로 강제 보정 |
| 메타데이터 | `id`/`scenario_id`/`source.*`는 LLM에 맡기지 않고 시스템이 덮어씀 (LLM이 `"..."` placeholder를 그대로 뱉은 사고 이후) |
| LLM 융합 후처리 | `enforce_evidence_grounding()` — LLM이 매긴 timestamp/evidence를 VLM·STT 원본과 대조해 **코드가 강제로 정정**. 4초 이상 어긋나면 자동으로 `action_only`로 강등 |
| 충돌 처리 | 본 것(VLM)과 들은 것(STT)이 다르면 판단하지 않고 `conflict=true`로 **둘 다 보존** |
| STEP6 | 규칙 기반 탈락조건(timestamp_validity) + 매뉴얼 RAG로 "이미 매뉴얼에 있는 내용(same)"이면 reject |
| STEP8 | 이중 방어 — ①유사도 임계값 미만이면 LLM 호출 자체를 안 함 ②시스템 프롬프트로 "검색된 근거만" |

STEP8 임계값 0.40은 실측으로 정했습니다. 0.35였을 때 `"오늘 점심 뭐 먹지"`(top1=0.372)가 방어선1을 통과했고, 그때는 방어선2(프롬프트)가 막아냈습니다. 하지만 그건 운이지 구조적 보장이 아니라서 임계값을 올렸고, 재검증에서 방어선1이 직접 차단하는 것을 확인했습니다.

---

## 실행 방법

### 사전 준비

```bash
# GPU 노드 확보 (로그인 노드엔 GPU 없음)
srun -p RTX6000 -w n5 --exclusive --gres=gpu:2 --pty bash -l

# STEP3/4/5는 venv 하나를 공유 (단일 프로세스가 STT+YOLO+VLM+LLM을 순차 로딩하므로)
cd 파이프라인_통합실행
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

cp config.example.yaml config.yaml
# config.yaml에서 채울 것:
#   detector.params.weights_path : 학습된 YOLO best.pt 경로
#   video_path                   : 영상 경로 (--video로 전달해도 됨)
```

### STEP3~5 — 암묵지 후보 생성

```bash
# 클립 1개
python run_one_clip.py --config config.yaml --video /path/to/CLIP1.mp4

# 여러 클립 (모델 재로딩 없이 순차 루프)
python run_all_clips.py

# STEP별 단독 실행 (디버깅용 — 이전 STEP 산출물이 파일로 있어야 함)
cd ../STEP3_전처리                  && python run_step3.py --config ../파이프라인_통합실행/config.yaml --video <영상>
cd ../STEP4_YOLO_VLM관찰            && python run_step4.py --config ../파이프라인_통합실행/config.yaml --video-id <video_id>
cd ../STEP5_LLM으로_암묵지_후보생성  && python run_step5.py --config ../파이프라인_통합실행/config.yaml --video-id <video_id>
```

### STEP6 — 품질검증

```bash
cd STEP6_품질검증
pip install -r requirements.txt
python main.py                          # input/ 폴더 전체 처리
python main.py --input /path/to/candidate.json
python main.py --mock                   # 모델 다운로드 없이 그래프 흐름만 확인
```

`manuals/`의 더미 매뉴얼을 실제 장비 매뉴얼로 교체해야 Gate A가 의미를 갖습니다.

### STEP7 — 벡터DB 적재 및 검색

```bash
cd STEP7_DB
python3 -m venv --system-site-packages .venv
.venv/bin/python3 -m pip install -r requirements.txt

.venv/bin/python3 build_db.py                      # 적재 + 검색 스모크 테스트
.venv/bin/python3 query_tacit.py "화면이 안 떠요"
.venv/bin/python3 query_tacit.py "래치 어떻게 눌러요" --top_k 2
.venv/bin/python3 query_tacit.py "커넥터 확인" --accept_only   # hold/reject 제외
```

문서 9건 규모라 GPU 없이 돌아갑니다(`config.device` 기본값 `cpu`). 각 암묵지 `id`를 UUID5로 변환해 point ID로 쓰므로 재적재해도 중복이 쌓이지 않습니다.

### STEP8 — 음성 RAG 웹서비스

```bash
cd STEP8_RAG서비스
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# 1) Ollama 서버 (GPU 노드 권장 — CPU면 14b 추론이 2분 넘어 타임아웃)
srun --jobid=<GPU잡ID> --overlap bash -c \
  'OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=~/.local/ollama/data ~/.local/ollama/bin/ollama serve'

# 2) 적재 → 3) 서버
.venv/bin/python3 ingest.py
OLLAMA_URL="http://<노드>:11434/api/chat" .venv/bin/uvicorn app:app --port 8000
# → http://localhost:8000 (마이크 버튼, Chrome 권장)

# 검색/평가만 따로
.venv/bin/python3 search.py "래치 어떻게 눌러요"
.venv/bin/python3 evaluate.py --top-k 3
```

`OLLAMA_HOST=0.0.0.0`을 반드시 줘야 합니다 — 기본값(127.0.0.1)이면 서버가 뜬 노드 밖에서 접근이 안 됩니다.

---

## 저장소 구조

```
project-ai/
├── 00_사전연구/              STT 선정, VAD 재검증 (실험 코드 + 결과 원자료)
├── STEP2_모델선정/           VLM 3종 재비교 실행 코드 + 결과 JSON (집계 보고서는 미생성)
├── tacit_common/             STEP3/4/5 공용 — 인터페이스, 중간 산출물 스키마, config 로더
├── STEP3_전처리/             STT + 전사 정제 + 프레임 추출
├── STEP4_YOLO_VLM관찰/       YOLO 부품 검출 + VLM 행동 관찰 (영상 갈래 전용)
├── STEP5_LLM으로_암묵지_후보생성/  윈도우 정렬 + LLM 융합 + 최종 스키마(v1.3)
├── 파이프라인_통합실행/       STEP3~5를 한 프로세스에서 순차 실행 (모델 재로딩 최소화) + run_logs/
├── STEP6_품질검증/           LangGraph 4게이트 → accept/hold/reject
├── STEP7_DB/                 임베딩 + Qdrant 적재 + 검색(RAG)
├── STEP8_RAG서비스/          FastAPI 음성 RAG 웹서비스 (STT→검색→LLM→TTS)
├── docs/                     계획서, 체크리스트, 심층분석 보고서
└── _archive_pre_server/      서버 이관 전 사전연구 원자료(LLM/VLM 비교 보고서 포함)
```

**폴더 규칙**: `STEP{N}_*/`에는 실행 코드와 산출물만, 문서는 전부 `docs/`에 둡니다. 2개 이상 STEP이 함께 쓰는 코드만 `tacit_common/`에 두고, 특정 STEP 전용 타입에 의존하는 코드는 그 STEP 소유로 둡니다(공용 패키지를 import하는 다른 STEP까지 같이 깨지는 것을 실제로 겪은 뒤 정한 규칙).

각 STEP은 **자기 폴더 안에서 단독 실행이 가능**하고, 통합 러너는 컴포넌트를 한 번만 빌드해 재사용하되 **단계 사이 데이터는 항상 파일로 주고받습니다**. 그래서 STEP4만 다시 돌리거나 STEP5만 재실행하는 게 가능합니다.

### 모델 교체 방식

STT/VLM/LLM/YOLO/정렬기 전부 `tacit_common/interfaces/`의 Protocol 뒤에 있고, `config.yaml`의 `impl` 문자열 → `step{N}_registry.py`의 딕셔너리 → 어댑터 클래스로 연결됩니다.

새 모델 추가 = `step{N}_components/`에 클래스 1개 + registry 한 줄 + config 한 줄. **기존 코드는 수정하지 않습니다.**

---

## 현재 상태와 알려진 한계

정직하게 적습니다. 면접에서 물어보면 그대로 답할 내용입니다.

**동작 확인된 것**
- STEP3~5 end-to-end 실행 완료 (CLIP1 기준: YOLO 162건 검출 → VLM 관찰 → 암묵지 후보 JSON 생성)
- STEP6 4게이트 판정 동작 (예시 후보에서 HOLD 판정 + 항목별 근거 로그)
- STEP7 적재·검색 동작 (9건, 관련 질의 top1 정확, 무관 질의는 "못 찾음" 응답)
- STEP8 음성 RAG 동작 (Recall@3 = 9/9, 이중 방어 실측 검증)

**아직 해결되지 않은 것**
- **CLIP4(115프레임) 미해결** — `max_memory` 21/17, 21/15 양쪽 다 OOM, `fps=0.4`로 낮추면 관찰 0건으로 얼어붙음. fps 문턱 재탐색 또는 `max_pixels` 하향으로 별도 처리 예정
- **STEP7/8의 입력이 실제 STEP5 산출물이 아님** — STEP4 VLM이 안정적으로 결과를 못 낼 때 만든 예시 레코드 9건(`STEP7_DB/gold_records/`)을 쓰는 중입니다. 실제 산출물이 나오면 `config.py`의 `input_dir`만 바꾸면 됩니다
- **`verification.routing=="accept"` 필터가 기본 OFF** — 진짜 routing 데이터가 없어 훅만 만들어둔 상태(`--accept_only`로 켤 수 있음)
- **YOLO가 프레임 배치 추론이 아니라 프레임별 순차 `predict()` 호출** — 미최적화 지점
- **`vlm.params.input_mode`는 코드에 분기가 없는 죽은 설정값** — 항상 `observe_frames()`를 직접 호출합니다
- **`FrameMeta.fps`가 실제 영상 fps가 아니라 30.0 하드코딩 폴백** — 현재 사용 경로엔 영향이 없지만 TODO
- **STEP6 가중치·임계값이 트랙 A(고정 가중치) 기본값** — 라벨이 50건 이상 쌓이면 로지스틱 회귀로 재산정(트랙 B)하도록 `config.py`에 자리는 만들어 뒀습니다
- **STEP8 Ollama가 팀 공용 GPU 잡에 편승 중** — 정식 배포 시 전용 할당 필요
- **STEP2_모델선정의 집계 보고서가 없음** — 최신 VLM 재비교의 결과 JSON은 있지만 `compare_results.py`를 돌려 만든 비교표가 저장소에 없어, 현재 모델 채택 근거로 제시 가능한 건 초기 비교(archive) 쪽입니다
- **`docs/` 일부 문서가 구버전 구조(RTX 2080, vLLM, `tacit_pipeline/orchestrator.py`) 기준** — STEP3/4/5 분할 전에 쓴 문서들이라 경로·하드웨어 서술이 현재와 다릅니다. 현재 확정값의 기준은 `파이프라인_통합실행/config.yaml`입니다

LLM의 시간·근거 판단을 원천적으로 신뢰하지 않고 코드가 사후에 강제 정정하는 구조 자체가, 로컬 sLLM의 시간 정합성 추론이 아직 불안정하다는 방증이기도 합니다.

---

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/전처리_파이프라인_계획서.md`](docs/전처리_파이프라인_계획서.md) | 확정 설계 문서. STEP3/4/5 관련해 애매하면 이 문서 기준 |
| [`docs/실행전_방어_체크리스트.md`](docs/실행전_방어_체크리스트.md) | GPU 잡 실행 전 필수 순서 |
| [`docs/면접대비_파이프라인_심층분석.md`](docs/면접대비_파이프라인_심층분석.md) | 코드 인용 포함 전체 분석 |
| [`docs/면접대비_파이프라인_요약.md`](docs/면접대비_파이프라인_요약.md) | 위 문서 요약본 |
| [`docs/_프롬프트_전수_보고서.md`](docs/_프롬프트_전수_보고서.md) | STEP3~5 실제 전송 프롬프트 전문 |
| [`docs/_STEP7_8_심층분석_보고서.md`](docs/_STEP7_8_심층분석_보고서.md) | STEP7/8 전수 분석 |
| [`docs/STEP8_RAG서비스_통합보고서.md`](docs/STEP8_RAG서비스_통합보고서.md) | STEP8 통합 배경·결정 근거 |
| [`00_사전연구/STT_VAD_연구결과.md`](00_사전연구/STT_VAD_연구결과.md) | STT 4-arm 비교 + VAD 재검증 실측 |
