# STEP3 전처리 — 암묵지 후보 생성 파이프라인

스마트글래스로 찍은 숙련자(명장)의 작업 영상에서 **영상(행동)** 과 **음성(설명)** 을 각각
따로 추출·정제한 뒤, 마지막에 LLM으로 융합해 **"암묵지 후보"를 구조화 JSON으로 생성**하는
전처리 파이프라인.

> 이 파이프라인은 후보 **생성**까지만 담당한다. 진짜 암묵지인지 **판별/검증은 다음 단계**(품질검증 팀).

---

## 데이터 흐름

```
영상 ─┬─ [영상 갈래] (성긴검출→) motion-guided 샘플링 → YOLO 검출(위치) → VLM 관찰추출
      └─ [음성 갈래] STT(Whisper) → transcript 정제(영어정규화 + 근거발화 태깅)
                          ↓  타임스탬프로 정렬(±N초 윈도우)
                   LLM 융합(Qwen2.5-14B) → 암묵지 후보 JSON (스키마 검증)
```

YOLO=위치, VLM=관찰(눈), STT=말 — 셋이 서로 다른 정보를 들고 와 **타임스탬프로** LLM에서 합쳐진다.

**역할 분리(중요):** VLM은 '눈'으로 **관찰만**(observations: actor/action/objects_visible). 묶기·해석·
진단(예: LED 깜빡임 의미)은 **전부 LLM이** 한다. LED는 VLM이 "황색 1회+백색 3회"처럼 횟수만 적고,
진단 코드표(Dell 7920)는 LLM 프롬프트에만 둔다.

**확정된 실행 방식:**
- **VLM 입력 = 네이티브 비디오 모드(본선)** — 영상+fps를 Qwen3-VL에 직접. (프레임-리스트는 `input_mode: frame_list` 옵션)
- **단일 GPU 순차 실행** — VLM 추론 완료 → `unload()` → LLM 로드. (오케스트레이터가 자동 처리)
- **VLM·LLM 둘 다 4bit NF4.** 서버 증설 시 config로 양자화 해제.

---

## 설계 원칙 (코드에 반영됨)

1. **모델 교체 용이성** — STT/VLM/LLM/YOLO/샘플러/정렬 전부 추상 인터페이스(`interfaces/`) 뒤에 두고,
   `config.yaml` 의 `impl` 한 줄로 구현을 갈아끼운다.
   새 모델 = `components/` 클래스 1개 + `registry.py` 한 줄 + config 한 줄. **기존 코드 무수정.**
2. **스키마 = 단일 진실 공급원** — 최종 JSON 구조는 `schema/tacit_schema.py` 한 곳에서만 정의.
   키 문자열 하드코딩 금지. `schema_version` 으로 버전 관리.
3. **타임스탬프가 뼈대** — 모든 중간 산출물은 공통 **초(float)** 단위. 프레임↔초 변환은 `FrameMeta(fps)`.
4. **할루시네이션 규율** — LLM은 입력에 없는 사실 생성 금지(묶기·분류·라벨링은 허용).
   추론은 `reasoning_origin="model_inferred"` 로 정직 태깅. **발화 근거인 척 위장 금지.**
   (`TacitKnowledgeCandidate.cross_check()` 가 위반 의심 지점을 경고로 출력)

---

## 폴더 구조

```
STEP3_전처리/
├── run.py                      # 엔트리포인트
├── config.example.yaml         # 설정 템플릿(복사 → config.yaml)
├── requirements.txt
└── tacit_pipeline/
    ├── config.py               # config 로더
    ├── constants.py            # YOLO 클래스 상수(실제 best.pt 기준 7개)
    ├── registry.py             # impl 이름 → 클래스 매핑(팩토리)
    ├── orchestrator.py         # 단계 배선(파이프라인 본체)
    ├── schema/                 # tacit_schema(최종) + intermediate(중간 계약)
    ├── interfaces/             # 추상 인터페이스(Protocol)
    ├── components/             # 구체 어댑터(whisper/yolo/qwen-vl/qwen-llm/정제/정렬/샘플)
    ├── prompts/                # VLM/LLM 프롬프트(코드와 분리)
    └── resources/              # en_normalization.json(기술용어 정규화 사전)
```

---

## 내일(서버 수령 후) 실행 순서

```bash
# 1) 의존성 설치 (Turing sm75: fp16/4bit만)
pip install -r requirements.txt

# 2) 설정 만들기
cp config.example.yaml config.yaml
#    config.yaml 에서 아래 TODO(fill) 자리를 채운다:
#    - detector.params.weights_path : best.pt 경로
#    - video_path (또는 --video 로 전달)

# 3) 실행
python run.py --config config.yaml --video /path/to/master_S1_take3.mp4
#    → output/<video_id>.tacit.json (암묵지 후보)
#    → transcripts/<video_id>.json (STT 결과, transcript_ref가 가리킴)
```

---

## 내일 채울 자리 (TODO)

| 위치 | 무엇 |
|---|---|
| `config.yaml` `detector.weights_path` | 튜닝된 YOLO `best.pt` 경로 |
| `config.yaml` `video_path` | 명장 영상 경로 |
| `components/vlm_qwen.py` `_infer_video()` | **본선** Qwen3-VL 네이티브 비디오 추론(video+fps, NF4 generate) |
| `components/vlm_qwen.py` `_infer()` | (옵션) 프레임-리스트 모드 이미지 추론 |
| `components/vlm_qwen.py` `unload()` | 백엔드별 GPU 해제 마무리(순차 실행) |
| `components/llm_fusion.py` `_infer()` | Qwen2.5-14B 텍스트 추론(backend별) |
| `resources/en_normalization.json` | 실제 STT 오인식 패턴 추가 |
| `resources/videos_parts.json` | (옵션) 영상→부품명 주입 매핑(example 복사) |

코드 곳곳의 `# TODO(decision):` 주석 = 내가 정해야 할 설계 선택지.

---

## ⚠️ YOLO 클래스 — 실제 모델 확인 결과 (2026-06-30)

`best.pt` 내부 `names` 에서 직접 추출. **실제 7개** (스펙 표는 8개였음):

```
0:GPU  1:RAM  2:RAM_slot  3:eraser  4:hand  5:monitor  6:power_button_LED
```

- 스펙에 있던 **`motherboard` 는 불필요로 판단해 의도적으로 학습 제외**(팀 확인). 누락 아님.
  "좌표 기준점"이 필요하면 `RAM_slot` 등 다른 부품으로 대체.
- 클래스 정의(`data.yaml`)는 학습 당시 원본 PC 경로에 있어 export 폴더엔 미포함이라
  `args.yaml` 엔 클래스가 안 보였던 것(args.yaml은 하이퍼파라미터 전용).
- `constants.validate_model_names()` 가 로딩 시 실제 모델 `.names` 와 대조해 경고한다.

---

## 환경 메모 (삽질 방지)

- GPU = **RTX 2080 ×2 (각 8GB, Turing sm75)**, 폐쇄망 → 전부 로컬.
- **Turing 철칙: fp16 / 4bit 만. bf16·FP8 = 하드웨어 미지원.**
- bnb 양자화는 텐서병렬(TP) 미지원 → TP1(1장)로 떨어짐. 14B는 AWQ/GPTQ 권장.
