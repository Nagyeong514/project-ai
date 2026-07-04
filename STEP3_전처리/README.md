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

**(2026-07-04 갱신)** 위 데이터 흐름의 "(성긴검출→) motion-guided 샘플링"은 실제로는 미사용이다
— orchestrator(현재는 STEP4의 `step4_runner.py`)가 `.sample()`을 한 번도 호출한 적이 없다(config
값과 무관하게 항상 그랬음). 삭제하진 않고 `STEP4_YOLO_VLM관찰/step4_components/sampler_motion.py`에
"미사용"으로 표시만 해뒀다. 본선은 STT/YOLO/VLM 전부 STEP3가 균등 추출한 프레임을 그대로 쓴다.

**역할 분리(중요):** VLM은 '눈'으로 **관찰만**(observations: actor/action/objects_visible). 묶기·해석·
진단(예: LED 깜빡임 의미)은 **전부 LLM이** 한다. LED는 VLM이 "황색 1회+백색 3회"처럼 횟수만 적고,
진단 코드표(Dell 7920)는 LLM 프롬프트에만 둔다.

**확정된 실행 방식:**
- **VLM 입력 = 네이티브 비디오 모드(본선)** — 영상+fps를 Qwen3-VL에 직접. (프레임-리스트는 `input_mode: frame_list` 옵션)
- **단일 GPU 순차 실행** — VLM 추론 완료 → `unload()` → LLM 로드. (STEP4/5의 `step{N}_runner.py`가
  자동 처리. 2026-07-04 갱신: 예전엔 `tacit_pipeline/orchestrator.py` 하나가 이걸 다 했는데,
  STEP3/4/5 분할로 각 STEP의 runner가 자기 몫만 담당하게 나뉨 — 파이프라인_통합실행/에서
  4클립을 한 번에 돌릴 때도 이 세 runner를 그대로 순서대로 호출)
- **VLM·LLM 둘 다 4bit NF4.** 서버 증설 시 config로 양자화 해제.

---

## 설계 원칙 (코드에 반영됨)

1. **모델 교체 용이성** — STT/VLM/LLM/YOLO/샘플러/정렬 전부 추상 인터페이스(`tacit_common/interfaces/`)
   뒤에 두고, `config.yaml` 의 `impl` 한 줄로 구현을 갈아끼운다.
   새 모델 = 해당 STEP의 `step{N}_components/` 클래스 1개 + `step{N}_registry.py` 한 줄 + config
   한 줄. **기존 코드 무수정.** (2026-07-04 갱신: 예전엔 `interfaces/`/`components/`/`registry.py`가
   전부 `STEP3_전처리/tacit_pipeline/` 하나였는데, STEP3/4/5 분할로 공용 인터페이스는
   `tacit_common/`, 구현체/레지스트리는 각 STEP 소유로 나뉨.)
2. **스키마 = 단일 진실 공급원** — 최종 JSON 구조는
   `STEP5_LLM으로_암묵지_후보생성/step5_schema/tacit_schema.py` 한 곳에서만 정의(2026-07-04 갱신:
   예전 경로는 `tacit_pipeline/schema/tacit_schema.py` — LLM 융합 결과물이라 STEP5 소유로 이동).
   키 문자열 하드코딩 금지. `schema_version` 으로 버전 관리.
3. **타임스탬프가 뼈대** — 모든 중간 산출물은 공통 **초(float)** 단위. 프레임↔초 변환은 `FrameMeta(fps)`.
4. **할루시네이션 규율** — LLM은 입력에 없는 사실 생성 금지(묶기·분류·라벨링은 허용).
   추론은 `reasoning_origin="model_inferred"` 로 정직 태깅. **발화 근거인 척 위장 금지.**
   (`TacitKnowledgeCandidate.cross_check()` 가 위반 의심 지점을 경고로 출력)

---

## 폴더 구조

**(2026-07-04 갱신)** 이 README가 있는 `STEP3_전처리/`는 이제 STT+정제+프레임추출만 담당한다.
YOLO 검출+VLM 관찰은 `STEP4_YOLO_VLM관찰/`로, 정렬+LLM 융합은
`STEP5_LLM으로_암묵지_후보생성/`로 분할됐다(예전엔 이 폴더 하나가 파이프라인 전체였음). 아래는
그 분할 이후 기준 3-STEP 전체 구조.

`전처리_파이프라인_계획서.md`(확정 설계·실행 지침, 애매하면 이 문서 기준), 서버이관 핸드오프 메모,
STT/VAD 사전연구(`00_사전연구/`)는 project-ai 최상위 `docs/`, `00_사전연구/`에 있다(코드/실행
산출물과 문서를 분리).

```
project-ai/
├── tacit_common/                       # STEP3/4/5 공용: schema/intermediate, interfaces, config, artifacts
├── STEP3_전처리/                        # (이 폴더) STT+정제, 프레임 추출
│   ├── run_step3.py                    # 단독 엔트리포인트
│   ├── step3_registry.py / step3_runner.py / step3_preflight.py
│   ├── step3_components/               # stt_whisper.py, transcript_refine.py, frame_extract.py
│   ├── resources/                      # en_normalization.json
│   ├── transcripts/                    # STT 결과 JSON(정제까지 끝난 버전)
│   └── output/_frames/<video_id>/      # 추출 프레임 + frames_meta.json(STEP4가 읽음)
├── STEP4_YOLO_VLM관찰/                  # YOLO 검출 + VLM 관찰(영상 갈래 전용, transcript 의존 없음)
│   ├── run_step4.py
│   ├── step4_registry.py / step4_runner.py / step4_preflight.py
│   ├── step4_components/               # detector_yolo.py, detector_noop.py(비활성 대안),
│   │                                   #   vlm_qwen.py, sampler_motion.py(미사용)
│   ├── step4_prompts/                  # vlm_observation.py
│   ├── constants.py                    # YOLO 클래스 상수(실제 best.pt 기준 7개)
│   └── output/                         # detections/, vlm_observations/
├── STEP5_LLM으로_암묵지_후보생성/        # 정렬(Aligner) + LLM 융합 — 두 갈래 합류 지점
│   ├── run_step5.py
│   ├── step5_registry.py / step5_runner.py / step5_preflight.py
│   ├── step5_components/               # aligner.py, llm_fusion.py
│   ├── step5_prompts/, step5_schema/   # llm_fusion_prompt.py, tacit_schema.py(최종 출력 스키마)
│   └── output/                         # aligned_windows/, tacit_json/(최종 암묵지 후보)
└── 파이프라인_통합실행/                  # CLIP1~4 순차 통합 실행(모델 재로딩 최소화)
    ├── run_all_clips.py / run_one_clip.py
    ├── config.yaml / config.example.yaml    # STEP3/4/5 공용 단일 config
    ├── preflight.py                         # 통합용(STT+YOLO+VLM+LLM 넷 다 검사)
    ├── run_logs/
    └── .venv/                               # STEP3/4/5 단독 실행도 이 venv를 같이 씀
```

---

## 실행 순서

**(2026-07-04 갱신)** `run.py`/`run_all_clips.py`는 `파이프라인_통합실행/`으로 옮겨졌고,
`run.py`는 `run_one_clip.py`로 이름이 바뀜. STEP별 단독 실행 경로도 새로 생김.

```bash
# 1) 의존성 설치 (Turing sm75: fp16/4bit만) — 파이프라인_통합실행/.venv 기준
cd 파이프라인_통합실행
pip install -r requirements.txt

# 2) 설정 만들기
cp config.example.yaml config.yaml
#    config.yaml 에서 아래 TODO(fill) 자리를 채운다:
#    - detector.params.weights_path : best.pt 경로
#    - video_path (또는 --video 로 전달)

# 3-a) 통합 실행 — CLIP1~4 한 번에(모델 재로딩 최소화)
python run_all_clips.py

# 3-b) 통합 실행 — 클립 1개만
python run_one_clip.py --config config.yaml --video /path/to/CLIP1.mp4

# 3-c) STEP별 단독 실행(디버깅/재작업 시 — 이전 STEP 산출물이 파일로 이미 있어야 함)
cd ../STEP3_전처리        && python run_step3.py --config ../파이프라인_통합실행/config.yaml --video <영상경로>
cd ../STEP4_YOLO_VLM관찰   && python run_step4.py --config ../파이프라인_통합실행/config.yaml --video-id <video_id>
cd ../STEP5_LLM으로_암묵지_후보생성 && python run_step5.py --config ../파이프라인_통합실행/config.yaml --video-id <video_id>
```

산출물(위치는 폴더 구조 참고):
- `STEP3_전처리/transcripts/<video_id>.json` — STT 결과(정제까지 끝난 버전)
- `STEP3_전처리/output/_frames/<video_id>/` — 추출 프레임 + `frames_meta.json`
- `STEP4_YOLO_VLM관찰/output/detections/<video_id>.json`,
  `output/vlm_observations/<video_id>.observations.json`
- `STEP5_LLM으로_암묵지_후보생성/output/aligned_windows/<video_id>.windows.json`,
  `output/tacit_json/<video_id>.tacit.json` ← **최종 암묵지 후보**

---

## 채울 자리 (TODO)

**(2026-07-04 갱신)** 이 표는 원래 "서버 수령 전" TODO 목록이었다 — 경로만 새 구조로 갱신함.
`_infer_video()`/`_infer()`/`unload()` 항목은 실제로 이미 구현·검증된 것으로 보이나(네이티브
비디오 추론, hf_transformers 전환 등 — `docs/전처리_파이프라인_계획서.md` §8 실행 기록 참고),
행별로 다시 하나하나 확인한 건 아니라 완료 표시는 TODO로 남겨둠.

| 위치 | 무엇 |
|---|---|
| `파이프라인_통합실행/config.yaml` `detector.weights_path` | 튜닝된 YOLO `best.pt` 경로 |
| `파이프라인_통합실행/config.yaml` `video_path` | 명장 영상 경로 |
| `STEP4_YOLO_VLM관찰/step4_components/vlm_qwen.py` `_generate_mm()` | Qwen3-VL 네이티브 비디오 추론(video+fps, NF4 generate) |
| `STEP5_LLM으로_암묵지_후보생성/step5_components/llm_fusion.py` `_infer()` | Qwen2.5-14B 텍스트 추론(backend별) |
| `STEP3_전처리/resources/en_normalization.json` | 실제 STT 오인식 패턴 추가 |
| `STEP4_YOLO_VLM관찰/resources/videos_parts.example.json` | (옵션) 영상→부품명 주입 매핑(example 복사) |

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
- `constants.validate_model_names()` 가 로딩 시 실제 모델 `.names` 와 대조해 경고한다
  (2026-07-04 갱신: `constants.py`는 이제 `STEP4_YOLO_VLM관찰/`에 있다 — YOLO 검출만 쓰는 상수라
  STEP4 소유로 이동).

---

## 환경 메모 (삽질 방지)

- GPU = **RTX 2080 ×2 (각 8GB, Turing sm75)**, 폐쇄망 → 전부 로컬.
- **Turing 철칙: fp16 / 4bit 만. bf16·FP8 = 하드웨어 미지원.**
- bnb 양자화는 텐서병렬(TP) 미지원 → TP1(1장)로 떨어짐. 14B는 AWQ/GPTQ 권장.
