# 📌 먼저 읽으세요 — 이 프로젝트 상황 설명 (콜드스타트용)

> 다른 Claude/사람이 이 zip만 받고도 맥락을 잡도록 쓴 브리핑. 코드의 "무엇"이 아니라 "왜·어디까지·함정"을 적었습니다.

---

## 0. 한 줄 정의

스마트글래스로 찍은 명장(숙련자) 작업 영상에서 **영상(행동)** 과 **음성(설명)** 을 따로 뽑아 정제한 뒤, LLM으로 융합해 **암묵지 후보를 JSON으로 생성**하는 전처리 파이프라인.
범위는 "암묵지 후보 생성"까지 — **진짜 암묵지인지 판별은 다음 단계(품질검증 팀) 몫이라 우리는 안 함.**

```
영상 → YOLO(부품 위치) + VLM(행동 관찰)
오디오 → STT → transcript 정제
        ↓ 둘을 타임스탬프로 융합(LLM)
   암묵지 후보 JSON (schema_version 1.3)
```

---

## 1. 확정된 설계 결정 (이미 합의 — 다시 열지 말 것)

1. **VLM = 관찰만(버전 A).** 영상에서 보이는 사실만 기록. 묶기·해석·암묵지판단은 전부 융합 LLM이 함.
2. **모델 구성(현재 RTX 2080 8GB):** STT=faster-whisper large-v3-turbo(`condition_on_previous_text=False`), VLM=Qwen3-VL-8B-Instruct(4bit NF4), LLM=Qwen2.5-**7B**-AWQ(vLLM), YOLO=튜닝된 best.pt. 전부 config로 교체 가능.
3. **단일 GPU 순차 실행:** VLM 추론 → unload → LLM 로드 (8GB에 동시 적재 불가). 7B는 한 장에 들어감 실측됨.
4. **샘플링: 적응형 샘플러 안 만듦.** VLM은 균일 fps(네이티브 비디오), YOLO는 독립 높은 fps로 전체 훑어 **손-부품 접촉 이벤트를 백업 그물로 저장**(예정).
5. **타임스탬프 = 뼈대.** 영상 0:00 단일 기준. 각 단계가 부여한 시간을 그대로 들고 감(재계산 안 함).
6. **Transcript 정제 = (b)방식.** 정규식 근거추출은 한국어 구어체에서 0건이라 폐기. 정제는 **정규화 + 연속반복 환각감지만**. 근거성 판단은 융합 LLM이 raw 발화로 직접.
7. **LLM 융합 규칙:** 케이스 A(행동+발화)/B(행동만=action_only, ★누락금지)/C(발화만). 입력에 없는 사실 생성 금지(묶기·라벨링은 허용). 추론은 `reasoning_origin="model_inferred"`로 정직 태깅. LED 진단표는 LLM만 봄(VLM은 횟수만 관찰).
8. **출력 스키마:** Pydantic 단일 정의(`schema/tacit_schema.py`), 키 하드코딩 금지.

---

## 2. 환경 함정 (RTX 2080 = Turing sm75, conda) — 반드시 숙지

- **vLLM Turing 우회:** FlashAttention2(sm80+) 거부 → `VLLM_ATTENTION_BACKEND` 대신 `attention_backend="TRITON_ATTN"`, `dtype="half"`(bf16 금지), `enforce_eager=True`.
- **LLM은 vLLM으로:** 캐시 모델이 AWQ인데 autoawq 미설치 → transformers 불가, vLLM AWQ 네이티브.
- **libstdc++ 충돌:** 실행 시 `LD_LIBRARY_PATH=/home/piai/anaconda3/lib` 우선.
- **비디오 디코딩:** pyav/cv2/torchcodec 다 불안정 → **ffmpeg CLI subprocess로 jpg 추출**(`components/frame_extract.py`).
- **한글 경로 버그:** NFD/NFC 불일치로 파일 못 엶 → **영어 경로로 해결**(영상 `master/master_videos/clip1.mp4` 등).
- **VLM 실측:** 로드 20s·생성 35s·피크 6.88GB(8GB OK). 파라미터: max_pixels=192², repetition_penalty=1.2, do_sample=False.

실행 커맨드:
```bash
LD_LIBRARY_PATH=/home/piai/anaconda3/lib python run.py --config config.yaml --video <영상경로>
```

---

## 3. 현재 진행 상황 (어디까지 됐나)

**✅ 완료·검증됨**
- 4개 모델 환경 뚫고 개별 작동 확인.
- **음성 갈래 완성:** CLIP1~4 STT 완료(`transcripts/CLIP1~4.json`) + 정제본(`*.refined.json`). 정제 (b)방식 구현·검증.
  - 활성 정규화: 마더보이드/마더보드/메인보드 → motherboard 만(`resources/en_normalization.json`).
  - 대기 항목(`_candidates`/`_observed_ambiguous`): 실측 불충분/일반어 충돌로 비활성. **추측 활성화 금지(원문 훼손).**
  - 연속반복 환각: 1번째 보존, 2·3번째 `repeat_hallucination=true`.
- **VLM 관찰 프롬프트(버전 A) 반영 완료**(`prompts/vlm_observation.py`).
- **LLM 융합 프롬프트 전문 반영 완료**(`prompts/llm_fusion_prompt.py`, LED표 포함).

**❗ 아직 미반영 델타**
1. 스키마 `conflict`/`conflict_detail` 필드 (본 것 vs 들은 것 충돌 보존용) — **미반영**.
2. YOLO 독립 fps + 손-부품 접촉 이벤트 저장 — 미반영.
3. 융합 입력에 `yolo_contacts` + "접촉 있는데 VLM 행동 없으면 action_only로 살려라" — 미반영.

**🔲 미실행**
- CLIP1 풀런(STT→VLM→LLM→암묵지 JSON): 배선은 됨, 끝까지 안 돌려봄.
- YOLO 어댑터 실행 안 함.
- 커밋: 다 검증되면 한 번에 하기로 함.

---

## 4. 바로 다음 할 일 (순서)

1. **CLIP1 MVP 풀런** (YOLO/부품주입 잠시 빼고 STT→VLM→LLM부터). 단, **우리 실제 프롬프트**로 돌려야 의미 있음.
2. 나온 암묵지 JSON을 **「촬영 대본」의 암묵지 정답지 9개와 비교.** 특히 🔵 중요+침묵 구간 9개(action_only)를 다 잡는지가 핵심 시험대.
3. 결과 보고 프롬프트 튜닝 → YOLO 백업그물 통합 → 풀 통합.

> 핵심 주의: 우리 범위는 "후보 생성"까지. 후보가 진짜 암묵지인지 **판별은 안 한다.** 융합 LLM은 정직하게 후보를 보존·태깅만.

---

## 5. 폴더 안내 (이 zip 기준)

- `tacit_pipeline/` — 파이프라인 본체 (Protocol 인터페이스 + registry로 모델 교체 가능)
  - `interfaces/` — STT/VLM/LLM/Detector/Sampler Protocol
  - `components/` — 각 모델 어댑터 (stt_whisper, vlm_qwen, llm_fusion, detector_yolo, transcript_refine, aligner, frame_extract)
  - `prompts/` — **VLM 관찰 프롬프트 / LLM 융합 프롬프트 (지침의 핵심)**
  - `schema/` — `intermediate.py`(단계 간 중간 산출물), `tacit_schema.py`(최종 출력 v1.3)
  - `resources/en_normalization.json` — 용어 정규화 사전
- `config.yaml` — 이 머신 실제 실행값 / `config.example.yaml` — 템플릿
- `transcripts/` — CLIP1~4 STT 결과 + 정제본 (이미 검증된 음성 갈래 산출물)
- `run.py` — 진입점
