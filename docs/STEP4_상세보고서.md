# STEP4(YOLO 검출 + VLM 관찰) 상세 기술 보고서

> 목적 (STEP3 보고서와 동일)
> 1. **재디벨롭 참고 문서** — 설계 근거·파라미터·손잡이를 바로 찾는다.
> 2. **트러블슈팅 아카이브** — OOM·반복출력·부품오인 등 개발 중 시간을 잡아먹은 사건과 해결책.
>
> **작성 규칙(엄수):** 코드를 실제로 따라가 확인된 것만 기술. 확인 불가 항목은 본문에
> "**(확인 불가)**"로 명시하고 §7에 재수집. 모든 주장에 근거 위치(파일:라인 / config 키 / git 커밋).
> 작성 기준 시점: 2026-07-16. 기준 커밋: `45be634`.

---

## §0. 정본(canonical) 확정 — "어느 STEP4 얘기인가"

### 0-1. 디렉토리는 하나뿐이다

`find`로 전 트리를 훑은 결과 STEP4 계열 디렉토리는 **`STEP4_YOLO_VLM관찰/` 단 하나**다.
과거 존재했던 `STEP4_YOLO_VLM관찰_모션O/` 폴더는 **2026-07-13에 원본 STEP4로 병합·삭제**됐다
(`run_step4_motion.py:3-5` 주석: "STEP4_YOLO_VLM관찰_모션O 폴더를 원본 STEP4로 병합하며
run_step4_motion.py로 이관"). 즉 코드베이스는 하나이고, `step4_runner.py`·`step4_components/`가
정본이다(7/7 `segment_bounds` 로직 포함).

### 0-2. 계열은 "디렉토리"가 아니라 "실행 모드 + 출력 폴더"로 갈린다

하나의 코드베이스 위에서 **실행 진입점 2개 + 출력 폴더 2개**로 나뉜다. 이것이 사용자가 우려한
"계열"의 실체다.

| 구분 | 베이스(정본 본선) | 모션O(실험 계열) |
|---|---|---|
| 실행 진입점 | `run_step4.py`(단독) / 통합 `run_all_clips.py` | `run_step4_motion.py` |
| config | `파이프라인_통합실행/config.yaml` | `STEP4_YOLO_VLM관찰/config_motion.yaml` |
| 입력 | STEP3 `output/_frames/<video_id>/frames_meta.json`(원본 영상 1개당) | STEP3 `output/motion_guided_clips/*.mp4`(서브클립들) |
| 프레임 시각 | STEP3가 부여한 값 그대로(uniform=0기준, motion=원본좌표) | 각 서브클립을 **독립 영상으로** 재추출 → **서브클립-로컬 0기준** |
| 청크 경계 | frames_meta에 `segments` 있으면 `hard_breaks`, 없으면 시간청크 | `segments` 없음 → `hard_breaks=None`(기본 동작) |
| detections 출력 | `output/detections/<video_id>.json` | `output_motion/detections/<클립>.json` |
| observations 출력 | `output/vlm_observations/<video_id>.observations.json` | `output_motion/vlm_observations/<클립>.observations.json` |
| 원본 좌표 복원 | **불필요**(입력이 이미 원본 좌표) | **필요**(§4 — 리포에 스크립트 없음, 재현 불가 경고) |
| 코드 차이 | — | **없음.** `step4_runner`/`step4_components` 그대로 재사용. 차이는 입력 소스·경로뿐(`run_step4_motion.py:7-12`) |

- **config.yaml vs config_motion.yaml 실측 diff**: `detector`~`stt` 구간이 **완전 동일**(diff 0줄).
  차이는 `paths:`의 frames/detections/observations 3개를 `output_motion/`으로 격리한 것뿐
  (`config_motion.yaml:2-5` 헤더 주석과 실측 일치).
- **E2E 본선이 실제로 쓰는 것**: `main.py → run_all_clips.py → Step4Runner(config.yaml)`이므로
  **정본 산출물은 `output/`**. `output_motion/`은 모션가이드 서브클립 실험 계열이다.

### 0-3. "모션"이 두 갈래라는 점(중요 혼동 주의)

이 리포에는 모션가이드를 반영하는 **서로 다른 두 경로**가 있다. 섞으면 안 된다.

| 모션 경로 | 방식 | 좌표 | 재현성 | 산출물 |
|---|---|---|---|---|
| **(가) STEP3 sampling.impl="motion"** | STEP3 `MotionSampledFrameSource`가 서브클립 프레임을 뽑아 **원본좌표(`start_sec+t`)로 복원**한 frames_meta(+segments) 생성 → step4_runner가 `hard_breaks`로 청크 | 원본좌표 | **재현 가능**(segments가 좌표 운반) | `output/`(video_id=원본) |
| **(나) run_step4_motion.py** | motion_guided_clips 서브클립 mp4를 **각각 독립 영상으로** STEP4 처리 | 서브클립-로컬 | **재현 불가**(원본좌표 remap 스크립트 부재, §4) | `output_motion/`(video_id=서브클립) |

현재 디스크의 `output/` 관찰들은 (가)로 생성된 것이다(관찰 chunk가 `16-42/52-78/83-109`로
STEP3 motion segments 경계와 일치, 실측). `output_motion/`은 (나)다.

**근거 위치(§0):** `run_step4_motion.py:1-22`, `config_motion.yaml:1-21`, `파이프라인_통합실행/config.yaml`
paths, `main.py:78`, `run_all_clips.py:44-55`, 실측 `output/vlm_observations/CLIP2.mp4.observations.json`.

> **이 보고서의 기준**: 별도 명시가 없으면 **베이스 본선(=`output/`, config.yaml)** 기준으로 서술하고,
> 모션O 특유의 시간축 이슈는 §4에서 따로 다룬다.

---

## §1. 전체 구조 개관

### 1-1. 엔트리포인트 → 산출물 순서도

```
[입력] STEP3 산출물
   output/_frames/<video_id>/frames_meta.json  (frame_paths[], times[], fps, duration, segments?)
   output/_frames/<video_id>/frame_%04d.jpg
        │
        ▼  Step4Runner.run(video_id)  (step4_runner.py:83-143)
   fmeta 로드 → hard_breaks = [s.end_sec for s in segments[:-1]] (motion이면), 아니면 None
        │
        ├──────── [YOLO 축] (step4_runner.py:96-125) ─────────────────────────
        │   본선: yolo_subprocess_entry.py를 subprocess로 (CUDA_VISIBLE_DEVICES 격리)
        │      UltralyticsYOLODetector.detect(frame_refs, meta)  (detector_yolo.py:78-107)
        │      · 프레임마다 model.predict(conf,iou,imgsz,device)
        │      · Detection(timestamp=fr.timestamp, frame_idx, cls, conf, bbox)
        │   → save_detections → output/detections/<video_id>.json
        │   검출 0건 & impl!=noop → RuntimeError (파이프라인 중단, step4_runner.py:120-125)
        │
        ├──────── [VLM 축] (step4_runner.py:127-141) ────────────────────────
        │   injected = _injected_parts(video_id, flat_dets)  (videos_map 손지정 or YOLO 클래스)
        │   vlm.observe_frames(frame_paths, times,
        │        injected_parts=injected,
        │        detections=None if in_map else flat_dets,   ← 청크별 부품주입 소스
        │        segment_bounds=hard_breaks)                  (vlm_qwen.py:180-286)
        │      · times를 chunk_sec(40s) 구간으로 분할(hard_breaks 우선)
        │      · 청크마다 build_video_observation_messages(parts) + 프레임 PIL 리스트로 generate
        │      · 파싱 → snap_to_grid → ActionDescription(timestamp=t0+ts, ...)
        │      · dedup_merge + _collapse_screen_runs (repeat_count/end_timestamp로 압축)
        │   → save_observations → output/vlm_observations/<video_id>.observations.json
        │      (+ raw_by_chunk: 청크당 VLM 원출력 1건)
        │
[산출] output/detections/<video_id>.json
       output/vlm_observations/<video_id>.observations.json
        │
        ▼ (STEP4 밖) STEP5 aligner가 transcript(STT)와 ±4초 윈도우로 합류
```

- YOLO가 **먼저**, VLM이 **나중**. 순차 실행(§5-1의 GPU 오염 격리 + OOM 회피). YOLO는 서브프로세스라
  끝나면 프로세스 종료로 GPU 반납, VLM은 `unload()`(step4_runner.py:140-141).
- STT/transcript 의존 **없음** — STEP4는 영상 갈래 전용(step4_runner.py:2-3).

### 1-2. 두 축의 역할 분담 원칙과 코드 구현

| 원칙 | 명문 위치 | 코드 구현 |
|---|---|---|
| **VLM = '눈', 관찰만(해석·진단 금지)** | `vlm_observation.py:4-15`, README.md:27-29 | 시스템 프롬프트 [절대 규칙] 1·2(vlm_observation.py:29-33), LED는 횟수만/진단표는 LLM(§3-4 규칙5) |
| **YOLO = 위치(백업 그물)** | `intermediate.py:161` "위치 힌트(코드레벨 판단용)", README.md:20 | detections에 bbox/conf 보존. **단 하류 실현은 부분적**(§2-4) |
| **묶기·해석·진단은 전부 LLM** | README.md:27-29 | STEP5 llm_fusion이 담당 |

- **"YOLO는 촘촘한 백업 그물, VLM은 눈"의 현실**: 설계 의도는 YOLO가 이벤트 증거를 촘촘히
  보존하는 것이지만, **현재 YOLO와 VLM은 STEP3가 뽑은 동일한 프레임 세트(같은 0.5fps jpg)를
  공유**한다. YOLO 전용 고밀도 추출 경로는 없다(§1-3). 따라서 "촘촘한 그물" 의도는 fps 레벨에서
  실현돼 있지 않다. detections의 하류 활용도 제한적이다(§2-4). — **솔직한 평가는 §2-4**.

### 1-3. YOLO fps와 VLM fps는 독립적으로 설정 가능한가? → **아니오**

- YOLO(`yolo_subprocess_entry.py:53-59`)와 VLM(`vlm_qwen.py:180,210-214`) **둘 다 동일한
  `frames_meta.json`의 `frame_paths`/`times`를 소비**한다. STEP4에는 프레임 추출 fps 설정이 없다
  — fps는 전적으로 STEP3의 `frame_extraction`(현재 0.5) 소관이다.
- 즉 **YOLO fps ≡ VLM fps ≡ STEP3 추출 fps**. 독립 조정 불가. YOLO만 촘촘히 돌리려면 STEP3에서
  별도 고밀도 추출 경로를 새로 만들어야 한다(현재 없음). — 재디벨롭 후보(§6).

### 1-4. STEP4 관련 config 키 전수 목록 (config.yaml, config.py 기준)

#### (a) `detector` (config.py:90, detector_yolo.py:24-33)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `detector.impl` | `"yolo_ultralytics"` | `yolo_ultralytics` / `noop`(검출 없이) | noop이면 0건 가드 우회(VLM 단독 검증) |
| `.params.weights_path` | `.../dell7920_v1/weights/best.pt` | 커스텀 학습 가중치 절대경로 | 다른 학습본으로 교체(클래스 대조 경고) |
| `.params.device` | `"cuda:0"` | YOLO 추론 디바이스 | VLM과 같은 가시 디바이스로(오염 회피, config 주석) |
| `.params.conf` | `0.25` | 신뢰도 임계값 | 올리면 검출↓(놓침↑), 낮추면 검출↑(오검출↑) |
| `.params.iou` | `0.7`(코드 기본) | NMS IoU. **config에 미노출→기본값** | 겹친 박스 병합 강도 |
| `.params.imgsz` | `640` | 추론 입력 크기 | 올리면 작은 부품↑·느림 |
| `.params.bbox_format` | `"pixel_xywh"` | `pixel_xywh`/`norm_xywh` | norm이면 0~1 정규화(meta.w/h 필요) |
| `.params.strict_names` | `false` | names 불일치 시 예외 여부 | true면 학습본 교체 시 즉시 중단 |

#### (b) `vlm` (config.py:90, vlm_qwen.py:51-71)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `vlm.impl` | `"qwen3_vl"` | 레지스트리 키(현재 유일) | 새 VLM은 클래스+registry 추가 |
| `.params.model_name` | `"Qwen/Qwen3-VL-8B-Instruct"` | 모델 ID | 다른 VLM |
| `.params.backend` | `"hf_transformers"` | (vLLM 불가 노드, CLAUDE.md) | — |
| `.params.device` | `"auto"` | `auto`(2-GPU device_map) / `cuda:0` | auto는 `--gres=gpu:2` 필요(§5-1) |
| `.params.input_mode` | `"native_video"` | 명칭. 실제는 jpg→PIL 리스트를 video로 전달(§5-4) | — |
| `.params.quantization` | `"nf4"` | 4bit NF4 / None(fp16) | 서버 증설 시 해제(§6) |
| `.params.max_memory_gib` | `{0:21, 1:17}` | GPU별 상한(비대칭, KV캐시 여유) | §5-1 |
| `.params.max_pixels` | `36864`(=192²) | 프레임 최대 픽셀 | 낮추면 부품오인↑(§5-3), 올리면 VRAM↑ |
| `.params.chunk_sec` | `40` | 관찰 청크 길이(초) | 0이면 전체1회(비권장, §3-3) |
| `.params.repetition_penalty` | `1.2` | 반복 억제 | §5-2 |
| `.params.max_new_tokens` | `4000` | 생성 상한 | 낮추면 JSON 잘림(salvage로 부분복구) |
| `.params.do_sample` | `false` | 그리디(재현성) | true면 확률적 |
| `.params.part_injection` | `true` | [고정 사실] 부품주입 on/off | false면 모델 자체 인식 |
| `.params.videos_map_path` | `null` | 영상→부품 수동매핑 JSON | 주면 그 목록 전구간 고정주입 |

> `frame_extraction.*`(fps 등)는 STEP3 소관이지만 STEP4가 그 결과를 소비한다. VLM 관련 OOM
> 튜닝 이력이 `config.yaml`의 `frame_extraction.fps_long`/`max_memory_gib` 주석에 있다(§5).

**근거 위치(§1):** `step4_runner.py:83-143`, `yolo_subprocess_entry.py:53-59`, `vlm_qwen.py:180-286`,
`detector_yolo.py:24-33`, `tacit_common/schema/intermediate.py:161`, `config.py:90`, `config.yaml`.

---

## §2. YOLO 검출 축

### 2-1. 가중치와 추론 파라미터

- **모델**: 커스텀 학습 `best.pt`. config 경로
  `.../yolo_local_v13_0629.../runs/detect/dell7920_v1/weights/best.pt`(config.yaml `detector.weights_path`).
  Dell 7920 부품 학습본이다.
- **로딩**: 지연 로딩. `_load()`가 `ultralytics.YOLO(weights_path)` 생성 후
  `validate_model_names()`로 클래스 대조(detector_yolo.py:45-56).
- **추론**(detector_yolo.py:85-88): `model.predict(source, conf=0.25, iou=0.7, imgsz=640,
  device="cuda:0", verbose=False)`. bbox는 ultralytics `xywh`(중심)를 좌상단 xywh로 변환
  (detector_yolo.py:109-116).

### 2-2. 클래스 목록 (실물 확인 — 사용자 가정 정정)

**실제 모델 클래스는 7개**다(`constants.py:8-14`, `best.pt`의 `names` 직접 추출, 2026-06-30):

```
0:GPU  1:RAM  2:RAM_slot  3:eraser  4:hand  5:monitor  6:power_button_LED
```

> **정정**: 프롬프트에 적힌 8개(…+`motherboard`)는 스펙 표 기준이다. **`motherboard`는 불필요로
> 판단해 의도적으로 학습 제외**(팀 확인) — 누락이 아니라 설계 결정(constants.py:12-13,34-36).
> 즉 정본은 **7개**이며 motherboard 상수/좌표기준점 로직은 두지 않는다.

- 의미 분류(constants.py:53-62): 손=`[hand]`, 부품=`[RAM,RAM_slot,GPU]`, 도구=`[eraser]`(접점세척
  암묵지 도구), 진단=`[power_button_LED]`, 출력=`[monitor]`.
- **대조 검증 함수 존재함**: `validate_model_names(model_names)`(constants.py:65-78) — 모델 `.names`를
  `EXPECTED_NAMES`와 대조해 불일치 메시지 리스트 반환. `_load()` 직후 호출, `strict_names=False`면
  경고만(detector_yolo.py:51-56).

### 2-3. 검출 산출물 스키마 + 실제 예시

`output/detections/<video_id>.json`(artifacts.py:76-82) = `{video_id, detections:[Detection]}`.
`Detection`(intermediate.py:88-96):

| 필드 | 타입 | 의미 |
|---|---|---|
| `timestamp` | float | 초(원본좌표). `frames_meta.times[frame_idx]`로 부착 |
| `frame_idx` | int | 프레임 인덱스 |
| `cls` | str | 클래스명(모델 names) |
| `conf` | float | 신뢰도 |
| `bbox` | {x,y,w,h} | 픽셀 xywh(좌상단 기준) |

**실제 발췌**(`output/detections/CLIP2.mp4.json`, 114건, cls분포 GPU54/hand45/RAM9/LED6):
```json
{"timestamp": 16.0, "frame_idx": 0, "cls": "power_button_LED", "conf": 0.808,
 "bbox": {"x": 254.03, "y": 249.41, "w": 47.85, "h": 52.38}}
{"timestamp": 16.0, "frame_idx": 0, "cls": "hand", "conf": 0.585,
 "bbox": {"x": 299.28, "y": 212.10, "w": 60.72, "h": 125.18}}
```

- **timestamp 부착 구조**: `yolo_subprocess_entry.py:58` `FrameRef(frame_idx=i, timestamp=times[i], ...)`
  → detector가 `Detection.timestamp=fr.timestamp`(detector_yolo.py:97). 즉 **YOLO는 시각을
  계산하지 않고 frames_meta의 `times[frame_idx]`를 그대로 단다.**
- **모션 샘플링 시 등간격 아님**: `times`가 모션 경로(가)에서는 원본좌표라 등간격이 아니다
  (예 CLIP2 `times=[16,18,...,44, 52,...]` — 세그먼트 사이 점프). detection timestamp도 그대로
  비등간격. (uniform이면 `i/fps`로 등간격.)

### 2-4. 하류 소비 실태 감사 (재디벨롭 1순위)

**detections는 STEP5 aligned_windows에 저장되지만 LLM 프롬프트에는 안 들어간다** — 코드로 확정:

1. **정렬 단계**: STEP5 aligner가 detections를 윈도우에 배치
   (`aligner.py:65,72,85` `detections=w_dets`) → `AlignedWindow.detections`에 담김.
2. **디스크 저장됨**: `save_aligned_windows`가 `w.model_dump()`로 저장하므로 aligned_windows JSON에
   detections가 남는다(artifacts.py:129-135, `AlignedWindow.detections` 필드 intermediate.py:161).
3. **LLM payload에서 제외됨**: `llm_fusion._serialize()`(llm_fusion.py:187-221)가 LLM에 넘기는
   payload는 `window_id/case/window_start/window_end/actions/utterances`뿐이다. **`w.detections`를
   직렬화하는 코드가 없다.** actions의 `objects_visible`(VLM이 적은 객체명)만 들어간다.

**⇒ 솔직한 평가**: "YOLO 백업 그물"의 현재 실현 정도는 **부분적**이다. YOLO가 실제로 기여하는 것은:
- (실현됨) **VLM 부품주입 소스**: 청크별 실제 검출 클래스를 `[고정 사실]`로 VLM에 주입
  (`_parts_for_chunk`, vlm_qwen.py:449-459) — RAM 영상에서 GPU 환각 방지 등.
- (실현됨) **파이프라인 건강 가드**: 검출 0건이면 즉시 RuntimeError(환경고장 조기탐지, step4_runner.py:120).
- (미실현) **이벤트 증거(bbox/conf/시각)의 LLM 활용**: aligned_windows에 박제만 되고 LLM은 안 봄.
  즉 "촘촘한 그물로 이벤트를 보존해 판단에 쓴다"는 의도는 데이터 보존까지만 됐고 활용은 안 됨.
  → **detections를 LLM payload에 넣거나, 코드레벨 판단(예: LED 옆 손 위치)에 쓰는 것이 개선 1순위.**

**근거 위치(§2):** `detector_yolo.py:24-116`, `constants.py:8-78`, `yolo_subprocess_entry.py:53-74`,
`tacit_common/schema/intermediate.py:88-96,161`, `tacit_common/artifacts.py:76-82,129-135`,
`STEP5_.../aligner.py:65,72,85,116`, `STEP5_.../llm_fusion.py:187-221`, 실측 `output/detections/CLIP2.mp4.json`.

---

## §3. VLM 관찰 축

### 3-1. 모델·양자화

- **모델**: `Qwen/Qwen3-VL-8B-Instruct`, `Qwen3VLForConditionalGeneration`(vlm_qwen.py:123,153).
- **양자화**: 4bit **NF4**(BitsAndBytes). `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`
  (vlm_qwen.py:126-129). **compute_dtype=fp16**(Turing sm75 제약). `quantization=None`이면
  `torch_dtype=float16`(vlm_qwen.py:130-131). config로 on/off 노출됨(`vlm.params.quantization`).
- **attention**: `attn_implementation="sdpa"`(Turing FA2 불가, vlm_qwen.py:155).

### 3-2. 추론 파라미터와 각 값의 근거

| 파라미터 | 값 | 근거(코드/주석/git) |
|---|---|---|
| `max_pixels` | `192²`(36864) | 부품 오인 대응. 코드 주석 "8GB 검증용 하향 원복"(config.yaml), "해상도 낮추면 작은 부품 오검출 위험"(config.yaml `fps_long` 주석). **낮추면 부품오인↑**(§5-3) |
| `repetition_penalty` | `1.2` | 동일 블록 무한반복 억제(vlm_qwen.py:10 "팀 검증 파라미터"). 단 greedy+1.2로도 자기복제 루프는 못 막아 청크분할로 근본 해결(vlm_qwen.py:188-192) |
| `max_new_tokens` | `4000` | CLIP1 4000에서 완결(config.yaml 주석). 초과 잘림은 `_salvage_observations`로 부분복구 |
| `do_sample` | `false` | 재현성(그리디). 빈 청크 재시도 때만 temp=0.3 샘플링(vlm_qwen.py:555-556) |
| `chunk_sec` | `40` | 전체1회 붕괴 4종 대응(§3-3) |
| fps | STEP3의 0.5 공유 | §1-3 — STEP4 자체 fps 없음. fps 하향(0.5→0.3/0.15)은 관찰0건 버그 방아쇠라 폐기(§5-3) |

### 3-3. 청크 처리 (40초 구간)

- **구조**(vlm_qwen.py:180-286): 영상을 `chunk_sec=40`초 구간으로 잘라 **구간마다 generate() 따로**.
  구간당 ~20프레임(0.5fps)/관찰 한 자릿수. `chunk_sec=0/None`이면 전체 1회(비권장).
- **전체 1회가 무너진 4종**(vlm_qwen.py:187-192, git 8c83615): (a)~15관찰 후 자기복제 루프
  (b)등간격 타임스탬프 날조 (c)관찰0건 동결(CLIP4) (d)KV캐시 OOM. 청크 분할로 넷 다 구조적 해소.
- **청크 경계에서 행동 잘림 대응**:
  - 꼬리 구간이 3프레임 미만이면 직전 구간에 병합(`_chunk_by_time`, vlm_qwen.py:406-409) —
    프레임 1~2장짜리 '영상'은 관찰 의미 없고 processor 불안정.
  - 모션 경로(segments 있음): `hard_breaks`로 **세그먼트를 먼저 확정한 뒤 그 안에서만** chunk_sec
    적용(`_chunk_ranges`, vlm_qwen.py:412-447). 세그먼트 프레임이 한 청크에 섞이는 것 구조적 방지
    (CLIP4 18초 중첩 처리 중 발견, vlm_qwen.py:420-427).
  - 청크 경계를 넘는 반복 관찰은 통합 후 dedup으로 접음(§3-6).
  - **(확인 불가)** 40초 경계에서 하나의 연속 동작이 두 청크로 쪼개져 별개 관찰로 기록될
    가능성 자체를 막는 로직은 없다(경계는 세그먼트/시간 기준으로만 정해짐). 실측 빈도 미확인.

### 3-4. 프롬프트 해부 (vlm_observation.py)

- **위치**: `STEP4_YOLO_VLM관찰/step4_prompts/vlm_observation.py`. 본선이 쓰는 빌더는
  `build_video_observation_messages()`(vlm_qwen.py:216에서 호출). per-frame용
  `build_observation_messages()`는 죽은 경로(본선 미사용).
- **시스템 프롬프트 전문**(vlm_observation.py:25-70)의 [절대 규칙] 요지 + 요청 기법 대조표:

| 사용자 지목 기법 | 프롬프트에 있나 | 위치 |
|---|---|---|
| ① 역할 고정("관찰만, 해석 금지") | ✅ | 시스템 [절대규칙] 1 (vlm_observation.py:29-30), 헤더(:26) |
| ② 추상동사 금지(확인/점검/검사/진단) | ✅ | [절대규칙] 2 (vlm_observation.py:31-33) |
| ③ 부품주입 `[고정 사실]` 블록 + on/off | ✅ | `injected_parts` → "[고정 사실] …(이 식별을 신뢰하라)"(vlm_observation.py:120-124, 154-157). on/off는 `part_injection` config + `_parts_for_chunk` |
| ④ few-shot 출력형식 고정 | ✅ | `_FEWSHOT_EXAMPLES`+`_fewshot_block()`(vlm_observation.py:77-104). "형식만 따라하고 내용 베끼지 마라" 경고 포함(예시 복사 사고 대응, :74-76) |
| ⑤ 손가락 단위 분리 | ✅ | [절대규칙] 3 "어느 손/어느 손가락이/무엇을"(vlm_observation.py:34) |
| ⑥ 수치요청 금지(단 LED 횟수 허용) | ✅ | [절대규칙] 4 "측정값 추정 마라, 단 셀 수 있는 것(LED 횟수)은 허용"(vlm_observation.py:35-36) |
| ⑦ LED 횟수만 관찰, 진단표는 LLM | ✅ | [절대규칙] 5 "색·횟수만, 의미(원인/진단) 절대 금지"(vlm_observation.py:37-42). 진단표는 프롬프트에 없음(:14-15) |

- 추가 규칙: 화면/모니터 상태도 관찰 대상(규칙7, actor="시선"), 화면내용 추측금지(규칙9),
  같은 문장 반복금지(규칙8), 촬영이 'RAM 탈거 시뮬'이란 메타정보는 모델에 숨김(:17).

### 3-5. timestamp 원칙 검증 (VLM은 시각을 계산하지 않는다)

- VLM이 뱉은 `timestamp` 문자열 → `hhmmss_to_seconds()`로 초 변환 → **`snap_to_grid(ts, local)`로
  구간 내 실제 프레임 그리드 중 최근접값으로 교정**(vlm_qwen.py:264-266). 최종 저장은
  `timestamp = t0 + ts`(청크 시작 전역시각 + 교정된 로컬시각, vlm_qwen.py:270-271).
- 파싱 실패/누락 시 0:00 떼몰림 대신 청크 그리드를 관찰 순서대로 균등 배분(`_even_grid_ts`,
  vlm_qwen.py:268-269, 564-576).
- 즉 **모델 자유 계산이 아니라 우리가 부여한 그리드 값을 강제**한다(snap_to_grid docstring
  vlm_qwen.py:30-38, 원칙 5.6). 프롬프트도 "주어진 프레임 시각을 쓰라"고 지시(vlm_observation.py:114,150-151).

### 3-6. 후처리: dedup + 파싱 재시도

- **dedup(4-B)** `dedup_merge`(vlm_qwen.py:294-323): 전체 청크 통합 후 시간순 정렬 → **최근 4건**과
  정규화 action(`_norm_action`: 공백/비단어 제거+소문자) 완전일치면 접음. 삭제가 아니라
  **`repeat_count += `, `end_timestamp = max(...)`**. 비교창 2→4 확대 근거: CLIP1 160~196s에서
  위/좌/우/아래만 바꾼 4문장 순환 퇴화가 2건 창을 통과(vlm_qwen.py:304-308).
- **화면 run 압축(4-E)** `_collapse_screen_runs`(vlm_qwen.py:342-386): 같은 청크 내 화면상태 관찰
  (`monitor 단독 + actor 시선`)이 연속이면 대표1건으로 접음. 단 **숫자 포함 화면관찰
  (`_has_concrete_info`, 예 "65536 MB")은 실제 정보라 접지 않음**(vlm_qwen.py:331-340, 375-376).
- **JSON 파싱 재시도**: `_parse_observations`(vlm_qwen.py:602-627) — ①코드펜스 제거→ `{`~`}` 잘라
  `observations` 추출(빈 배열도 인정) ②실패 시 `_salvage_observations`(brace 매칭으로 완성된
  관찰만 부분복구, 잘린 마지막 객체는 버림, vlm_qwen.py:578-600) ③완전실패 시 **빈 결과+경고**
  (예전 raw 덩어리를 관찰1건으로 밀어넣어 STEP5 오염시키던 폴백 제거, vlm_qwen.py:623-626).
- **빈 청크 재시도(4-C)**: greedy가 `observations:[]`로 얼면 그 청크만 `do_sample=True, temp=0.3`
  1회 재생성(vlm_qwen.py:250-260, 555-556).

### 3-7. 산출물 스키마 + 실제 예시

`output/vlm_observations/<video_id>.observations.json`(artifacts.py:97-114) =
`{video_id, n_observations, observations:[ActionDescription], raw_by_chunk?}`.
`ActionDescription`(intermediate.py:130-147):

| 필드 | 타입 | 의미 |
|---|---|---|
| `timestamp` | float | 초(청크시작+그리드스냅). 원본좌표(베이스) |
| `actor` | str? | 행동주체(오른손/왼손/양손/시선) |
| `action` | str | 관측 구체동작(추상동사 금지) |
| `objects` | list[str] | `objects_visible` — 보이는 객체명(**VLM 자유서술**, YOLO 클래스와 불일치 가능) |
| `end_timestamp` | float? | 접힌 반복의 마지막 시각(반복 없으면 null) |
| `repeat_count` | int | 관찰 총 횟수(1=반복없음) |
| `chunk` | str? | 이 관찰이 나온 청크 "t0-t1"(초) |

**실제 발췌**(`output/vlm_observations/CLIP2.mp4.observations.json`, 16건):
```json
{"timestamp": 16.0, "actor": "오른손", "action": "손가락으로 컴퓨터 케이스 우측 면의 원형 전원 버튼을 누르는 듯한 움직임을 한다", "objects": ["power_button_LED", "computer_case"], "end_timestamp": null, "repeat_count": 1, "chunk": "16-42"}
{"timestamp": 30.0, "actor": "오른손", "action": "GPU 카드의 금속 클립을 왼손으로 잡고, 오른손으로 GPU를 앞으로 당기는 동작을 한다", "objects": ["GPU_card", "metal_clips"], "end_timestamp": 36.0, "repeat_count": 2, "chunk": "16-42"}
```
> `objects`의 `GPU_card`/`metal_clips`/`computer_case`는 YOLO 7클래스에 없다 — **VLM이 자유 서술한
> 객체명**이다(few-shot도 `battery_connector` 등 자유명 사용, vlm_observation.py:82). 부품주입은
> 클래스 신뢰를 유도할 뿐 objects 값을 YOLO명으로 강제하진 않는다.
> `raw_by_chunk` 키 = `["16-42","52-78","83-109"]`(청크당 VLM 원출력 1건, 디버깅용).

**근거 위치(§3):** `vlm_qwen.py:51-71,123-155,180-386,449-459,555-627`, `vlm_observation.py:25-168`,
`tacit_common/schema/intermediate.py:130-147`, `tacit_common/artifacts.py:97-114`, git 8c83615,
실측 `output/vlm_observations/CLIP2.mp4.observations.json`.

---

## §4. 모션가이드 계열의 시간축 재매핑 (재현성 감사)

> ⚠️⚠️⚠️ **경고 박스 — 재현 불가능 지점** ⚠️⚠️⚠️
>
> `run_step4_motion.py`(모션O 경로 (나))는 서브클립을 **각각 독립 영상**으로 처리해
> `output_motion/vlm_observations/<서브클립>.observations.json`을 만든다. 이 산출물의 timestamp는
> **서브클립-로컬(0기준)**이다. 이것을 원본 영상 시간축으로 재매핑한
> `output_motion/vlm_observations_original_ts/<원본>.observations.json`이 존재하고,
> STEP5 분석(예: `docs/report_packet/scripts_trace/trace_stage1.py:13`)이 **이 original_ts를 입력으로
> 소비**한다. **그러나 original_ts를 생성한 재매핑 스크립트는 리포지토리에 존재하지 않는다**
> (`grep -rn "original_ts" --include=*.py` 결과: 생성 코드 0건, 소비 코드만 존재).
>
> 이 4개 파일은 **커밋 `331c772`("original_ts 4파일 추적(유일 백업)")로 산출물만 박제**됐다 —
> 커밋 메시지 자체가 "유일 백업"임을 인정한다. **즉 세션 작업으로 만들어졌고 스크립트가
> 유실된, 재현 불가능한 산출물이다.**

### 4-1. 재매핑 규칙 복원 (산출물 역추적)

original_ts와 서브클립 산출물·CSV를 대조해 규칙을 복원했다.

- **오프셋 소스 = `clip_timestamps.csv`의 `start_sec`** (파일:
  `STEP1_모델연구/01_모션기반샘플링/02_모션가이드샘플링_프로토타입/04_샘플링 영상/<stem>/clip_timestamps.csv`).
- CLIP1 실측 대조:

  | | 서브클립 clip01 로컬 | original_ts | CSV start_sec |
  |---|---|---|---|
  | 첫 관찰 timestamp | 0.0 | 4.0 | clip01=**4.0** |
  | chunk | "0-36" | "4-40" | — |
  | clip04 마지막 관찰 | 로컬 ~20/24 | 168.0/172.0 | clip04=**148.0** (148+20=168 ✓) |

- **복원된 규칙**: `original_ts = 서브클립_로컬_ts + clip_timestamps.csv[start_sec]`. chunk 문자열도
  같은 오프셋으로 이동. (end_timestamp도 동일.) — **이것이 정확한 원본좌표 복원**이다.

### 4-2. `merge_by_group.py`는 이것과 다르다 (혼동 금지)

- `scripts/merge_by_group.py`(merge_by_group.py:1-15)는 **"가상 이어붙임" 시간축**이다.
  각 서브클립을 순서대로 이어붙인 가상 영상 기준으로 `t_off += fmeta["duration"]` 누적 오프셋을
  적용(merge_by_group.py:78-93). 즉 clip01=+0, clip02=+(clip01 길이), …
- CLIP1 대조: merge_by_group 오프셋 = 0, 36, 72, 108(각 36초). **실제 원본좌표 = 4, 53, 100, 148**.
  → **완전히 다른 시간축.** merge_by_group 산출물은 `output_motion/*_by_group/`이며 원본좌표가 아니다.
- **결론**: 원본좌표가 필요하면 `_original_ts`(스크립트 유실), 가상연결이면 `_by_group`
  (스크립트 있음). 둘을 섞으면 안 된다. STEP5 분석이 쓴 것은 `_original_ts`다.

### 4-3. 재매핑 과정에서 소실된 것들

- **`source:"human_observation"` 필드 소멸(Pydantic extra ignore)**: 중간타입
  `ActionDescription`(intermediate.py:130-147)에는 `source` 필드가 **없다**. `BaseModel`은 기본이
  extra="ignore"라 `source` 같은 추가 키는 **로드 시 조용히 버려진다**. `load_observations`가
  `ActionDescription.model_validate`로 파싱하므로(artifacts.py:117-121), 관찰을 한 번이라도
  로드→저장 라운드트립하면 `source`가 사라진다. 실측: `_original_ts` 파일의 관찰 필드 union =
  `[action,actor,chunk,end_timestamp,objects,repeat_count,timestamp]` — **source 없음**.
  (확인 불가: 원래 어떤 관찰에 `source:"human_observation"`가 있었는지는 유실된 원본에서만 알 수
  있어 코드로 재구성 불가.)
- **`out_of_coverage` 관찰 누락**: **(확인 불가)** — 코드/현존 산출물에 `out_of_coverage` 필드나
  그 처리 로직이 없다(grep 0건). 모션 서브클립은 원본의 일부 구간만 커버하므로(유지율 70% 목표)
  커버되지 않은 구간의 관찰은 애초에 생성되지 않는다(구조적 누락). 이 누락을 명시 기록한
  산출물은 리포에 없다.

**근거 위치(§4):** `run_step4_motion.py`, `scripts/merge_by_group.py:1-15,78-93`,
`tacit_common/schema/intermediate.py:130-147`, `tacit_common/artifacts.py:117-121`, git `331c772`,
`docs/report_packet/scripts_trace/trace_stage1.py:13`, 실측 `output_motion/.../CLIP1...clip01`,
`_original_ts/CLIP1...`, CSV `.../CLIP1_정상조립과정/clip_timestamps.csv`.

---

## §5. 트러블슈팅 아카이브

### 5-1. 단일 GPU OOM + CUDA_VISIBLE_DEVICES 오염 (YOLO→VLM)

- **증상**: CLIP1 99프레임에서 VLM이 `device_map="auto"`인데도 GPU 1장에 갇혀 21.97GiB 단일 OOM.
- **원인**: ultralytics `select_device()`가 `.predict()` 최초 호출 시 `os.environ["CUDA_VISIBLE_DEVICES"]`를
  무조건 덮어씀(ultralytics/utils/torch_utils.py:222 실측) → 같은 프로세스의 뒤 VLM이 1장에 갇힘
  (yolo_subprocess_entry.py:4-9, step4_runner.py:8-14).
- **최종 해결**: **YOLO를 서브프로세스로 완전 격리**(git 95cd6c2). 자식은 os.environ 복사본을 받으므로
  덮어써도 부모(VLM 로드 프로세스)에 안 샌다. 이후 VLM `device="auto"`가 2-GPU를 실제로 잡음
  (`hf_device_map` 로그로 검증, vlm_qwen.py:159-165).
- **방어 장치**: 서브프로세스 격리(항상, 명시 detector 주입 테스트만 예외), `hf_device_map` 로그,
  `max_memory_gib` 비대칭 `{0:21,1:17}`(KV캐시가 GPU1에 쏠리는 실측 대응, vlm_qwen.py:132-151),
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(vlm_qwen.py:120).
- **재발 시**: `--gres=gpu:2`인지(gpu:1이면 auto라도 1장, config.yaml 주석), `[VLM-DEVICE]` 로그가
  2개 GPU 보이는지, `max_memory` 배분(nf4+auto는 배치 재조정 불가 실측 — config.yaml `max_memory_gib` 주석).

### 5-2. 동일 행동 무한 반복 출력

- **증상**: ~15번째 관찰 이후 앞 문장 자기복제 루프, A/B/A/B 교대 반복, 위/좌/우/아래만 바꾼
  템플릿 순환(CLIP1 160~196s).
- **원인**: 출력이 길수록 greedy 디코딩이 패턴 반복으로 퇴화. 유사 프레임 과다도 기여.
- **실패한 시도**: `repetition_penalty=1.2`+greedy로도 자기복제 루프를 **못 막음**(vlm_qwen.py:188).
- **최종 해결**: (1)**청크 분할**(40초)로 청크당 관찰 한 자릿수 → 루프 진입 전에 종료(§3-3, git 8c83615).
  (2)**dedup 후처리 신설**(4-B `dedup_merge`, git 8cc0d12) — 삭제 대신 `repeat_count`/`end_timestamp`
  압축, 비교창 최근 4건. (3)화면 run 압축(4-E).
- **방어 장치**: chunk_sec, repetition_penalty, dedup_merge(최근4건), _collapse_screen_runs.
- **재발 시**: `[VLM-DEDUP]` 로그의 압축 건수, 특정 청크가 여전히 루프면 chunk_sec 축소 검토.

### 5-3. 부품 오인식(특히 RAM)

- **증상**: RAM 영상인데 VLM이 GPU로 환각 식별.
- **원인**: (a)해상도 부족으로 작은 부품 구분 실패 (b)YOLO 검출이 3일간 빈 파일이라 부품주입이
  통째로 꺼진 채 실행(§5-6과 연결).
- **최종 해결**: (1)**max_pixels 상향**(128→192²) — 낮추면 오검출이라 일부러 안 낮춤
  (config.yaml `fps_long`/`max_pixels` 주석). (2)**부품주입 `[고정 사실]` 프롬프트**
  (vlm_observation.py:120-124) — 청크별 실제 검출 클래스만 주입(전구간 고정주입은 그 구간에
  없는 부품 환각 유도, vlm_qwen.py:449-459, step4_runner.py:129-136).
- **방어 장치**: part_injection, `_parts_for_chunk`(구간 검출 클래스 우선), videos_map 손지정 옵션.
- **재발 시**: `[VLM-CHUNK] 주입 부품=` 로그, YOLO 검출이 실제로 비지 않는지(§5-6).

### 5-4. 네이티브 비디오 vs 프레임 리스트 입력 (최종 정착)

- **증상/맥락**: 이 환경에서 torchcodec/pyav/cv2 Python 디코더 불안정(vlm_qwen.py:8).
- **최종 정착(현재 코드 확정)**: **ffmpeg CLI로 추출한 jpg를 PIL 리스트로 읽어 `video` 타입으로
  전달**(vlm_qwen.py:214,230 `{"type":"video","video":frames}`, frames=PIL 리스트). config의
  `input_mode:"native_video"`는 **명칭일 뿐** — 원본 mp4를 디코더에 직접 넣는 게 아니다.
  per-frame 이미지 모드(`build_observation_messages`)와 `describe_actions`는 죽은 경로
  (vlm_qwen.py:461-463 `NotImplementedError`).
- **비디오 인식 방어(중요)**:
  - `do_sample_frames=False`(vlm_qwen.py:533) — 안 주면 transformers가 55프레임을 24fps로 오인해
    4프레임만 남김(video_grid_thw T=2 실측).
  - `video_metadata`에 `fps`(실제 프레임 간격) 명시(vlm_qwen.py:236-242) — 없으면
    replace_video_token이 fps=24 가정 → 관찰0건 유력 원인.
  - `video_grid_thw` T값 hard assert(4-D, vlm_qwen.py:544-551) — 프레임 조용한 축소 재발 시 즉사.
  - `_log_vision_input`으로 매 호출 vision 토큰 수 로깅(vlm_qwen.py:481-512).

### 5-5. VLM 출력 잡티(가타카나 혼입 등)

- **STEP4 처리**: 가타카나/한자 등 이물 문자에 대한 **전용 처리 로직은 STEP4에 없다**(grep 0건).
  VLM raw는 `[VLM-RAW]` 로그와 `raw_by_chunk`에 보존되고, `_parse_observations`가 JSON `{`~`}`
  바깥 텍스트를 잘라내므로 배열 밖 잡티는 자연 제거된다(vlm_qwen.py:611). action 문자열 내부에
  섞인 이물은 그대로 통과할 수 있다.
- **하류(STEP5) 처리**: `_looks_derailed`(llm_fusion.py:233-238)가 **한자(一-鿿)·대체문자(�)**를
  탈선 신호로 탐지. 단 정규식 `[一-鿿�]`은 **가타카나(0x30A0-0x30FF)를 포함하지 않는다** —
  가타카나 자체는 이 검출기로 안 걸린다(한자/�만). 이건 LLM 출력 대상이고 VLM 출력엔 미적용.
- **(확인 불가)** 가타카나 혼입의 실제 발생 빈도 — 코드/현존 로그에 빈도 집계가 없다. 처리 방식은
  위 일반적 JSON 파싱 제거에 의존.

### 5-6. YOLO 조용한 실패(빈 detections 둔갑) + cuDNN ABI 불일치

| 사건 | 증상/원인 | 해결 | 근거 |
|---|---|---|---|
| **YOLO 조용한 실패** | `No module named pyparsing` 등 환경문제를 삼키고 빈 detections 저장 → "검출 0건"과 구분 불가, 3일간 4클립 부품주입 꺼진 채 실행 | 예외 전파(exit!=0), 빈 파일 저장 금지(yolo_subprocess_entry.py:18-20,60-63); 검출0건이면 RuntimeError(step4_runner.py:120-125); requirements에 pyparsing 명시 | git 8c83615, 8c8361533 메시지 |
| **cuDNN 코어/서브라이브러리 불일치** | torch가 venv libcudnn.so.9(9.24) 선로딩 후 서브라이브러리를 LD_LIBRARY_PATH의 다른 버전(9.16)에서 dlopen → 첫 conv2d에서 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. STT 먼저 돈 프로세스는 우연히 생존, YOLO 서브프로세스만 죽음 | 자식 env의 LD_LIBRARY_PATH 맨 앞에 `nvidia.cudnn`의 lib 추가(출처 일치 강제, 덮어쓰기 금지) | `_cudnn_consistent_env`(step4_runner.py:46-69), git bc8835e |
| **few-shot 예시 복사** | 예시의 지우개/LED 문장이 timestamp까지 관찰로 복제됨(CLIP1 청크1) | 예시를 Dell과 무관한 소재(노트북 배터리/나사)로 교체 + "형식만 참고" 경고 | git 07d9709, 2dedc1d, vlm_observation.py:74-76,102-104 |
| **max_memory 1-GPU 안전화** | 2-GPU용 `{0:21,1:17}`가 1-GPU 할당에서 없는 장치 idx로 로딩 깨짐 | 실존 장치만 필터(vlm_qwen.py:140-148) | git 1c43983 |

**근거 위치(§5):** `vlm_qwen.py:8,120,132-165,188-192,214-242,449-627`, `yolo_subprocess_entry.py:4-20,60-63`,
`step4_runner.py:8-14,46-69,120-125`, `vlm_observation.py:74-76`, `STEP5_.../llm_fusion.py:233-238`,
git 95cd6c2/8c83615/bc8835e/8cc0d12/07d9709/2dedc1d/1c43983, config.yaml 주석.

---

## §6. 재디벨롭 가이드

### 6-1. 실행 커맨드

```bash
# 클립 1개(STEP3 산출물 있어야 함) — YOLO+VLM
cd project-ai/STEP4_YOLO_VLM관찰
srun -p RTX6000 -w n5 --gres=gpu:2 ../파이프라인_통합실행/.venv/bin/python3 \
  run_step4.py --config ../파이프라인_통합실행/config.yaml --video-id CLIP1_정상조립과정.mp4

# 전체 4클립(STEP3→4→5 통합)
cd project-ai/파이프라인_통합실행
srun -p RTX6000 -w n5 --gres=gpu:2 .venv/bin/python3 run_all_clips.py

# 모션O(서브클립 순회) — output_motion/으로 격리
cd project-ai/STEP4_YOLO_VLM관찰
srun -p RTX6000 -w n5 --gres=gpu:2 ../파이프라인_통합실행/.venv/bin/python3 \
  run_step4_motion.py --config config_motion.yaml            # 전부
  # --limit 1 (첫 클립만), --no-skip-existing (재실행)
```

- **YOLO만 따로**: `python3 step4_components/yolo_subprocess_entry.py --config <cfg> --video-id <id>`
  (step4_runner가 내부적으로 이걸 호출, step4_runner.py:77-78). 산출물 `output/detections/`.
- **VLM만 따로**: 정식 진입점 없음. `detector.impl=noop`로 두면 YOLO 건너뛰고 VLM만 돈다
  (검출0건 가드가 noop은 예외로 통과, step4_runner.py:120). 또는 `debug_vlm.py`/`scripts/debug_vlm_clip1.py`.

### 6-2. 개선 후보 우선순위

| 순위 | 항목 | 근거 |
|---|---|---|
| ① | **detections를 LLM에 전달**(백업 그물 미완성) — `_serialize`에 `w.detections` 추가하거나 코드레벨 판단(LED 옆 손 위치 등)에 활용 | §2-4 |
| ② | **`source:"human_observation"` 복원** — `ActionDescription`에 `source` 필드 추가(현재 extra ignore로 소멸) | §4-3 |
| ③ | **재매핑 스크립트 리포 편입** — `_original_ts` 생성 스크립트(§4-1 규칙: 로컬ts+CSV start_sec)를 정식 코드로 작성해 재현성 확보 | §4 경고박스 |
| ④ | **서버 증설 시 양자화 해제 + fps 상향** — 바꿀 config: `vlm.params.quantization`(nf4→null), `vlm.params.max_memory_gib`(상향/제거), `frame_extraction.fps_short/long`(0.5→상향, 단 §5-3 부품오인·§5-2 반복 재검증), `vlm.params.max_pixels`(192²→상향) | config.yaml |
| ⑤ | **YOLO 고밀도 독립 fps** — 백업 그물 의도 실현하려면 YOLO 전용 고밀도 프레임 경로 신설(현재 STEP3 fps 공유) | §1-3 |

### 6-3. 건드리면 위험한 불변식

1. **timestamp 원본 좌표계 유지.** YOLO/VLM 둘 다 `frames_meta.times`를 그대로 단다(계산 금지).
   `snap_to_grid`(vlm_qwen.py:264-266)와 `t0+ts`(vlm_qwen.py:270) 로직을 바꾸면 STEP5 ±4초 정렬이 깨진다.
   모션O는 로컬좌표→원본좌표 remap(§4-1)이 반드시 선행돼야 STEP5가 원본영상 단위로 정렬 가능.
2. **VLM "관찰만" 역할 경계.** 프롬프트에 해석·진단·묶기를 허용하면 STEP5 게이트 전제(관찰=사실,
   판단=LLM)가 무너진다. LED 진단표를 VLM 프롬프트에 넣지 말 것(vlm_observation.py:14-15).
3. **YOLO 서브프로세스 격리 유지.** 인프로세스로 되돌리면 CUDA_VISIBLE_DEVICES 오염으로 VLM이
   1-GPU에 갇혀 OOM(§5-1). detector 인자 주입은 테스트 전용.
4. **검출0건 = 즉시 실패(noop 제외).** 이 가드를 빼면 환경고장이 조용한 빈 결과로 둔갑(§5-6).
5. **파싱 완전실패 = 빈 결과.** raw 덩어리를 관찰로 밀어넣는 폴백을 부활시키면 STEP5 오염
   (vlm_qwen.py:623-626).
6. **do_sample_frames=False + video_metadata.fps 명시.** 빼면 프레임 조용한 축소/fps 오인 →
   관찰0건(§5-4).

**근거 위치(§6):** `run_step4.py`, `run_step4_motion.py`, `yolo_subprocess_entry.py:77-78`,
`step4_runner.py:120`, `vlm_qwen.py:264-271,533,623-626`, `vlm_observation.py:14-15`,
`STEP5_.../llm_fusion.py:187-221`, config.yaml.

---

## §7. "확인 불가" 항목 목록

1. **`_original_ts` 재매핑 스크립트** — 생성 코드가 리포에 없다(grep 0건). 산출물 4파일만 커밋
   `331c772`로 박제("유일 백업" 자인). 규칙은 §4-1로 역추적 복원했으나 스크립트 자체는 재현 불가.
2. **`source:"human_observation"`의 원래 분포** — `ActionDescription`에 `source` 필드가 없어
   라운드트립 시 소멸. 어떤 관찰에 있었는지는 유실된 원본에서만 확인 가능.
3. **`out_of_coverage` 관찰** — 필드·로직·명시 기록 산출물이 리포에 없다. 모션 서브클립 미커버
   구간의 관찰은 구조적으로 생성 안 됨(누락 자체를 기록한 산출물 부재).
4. **가타카나 혼입 발생 빈도** — 코드/현존 로그에 집계 없음. STEP4에 가타카나 전용 처리 없음.
5. **40초 청크 경계에서 연속 동작이 쪼개지는 실측 빈도** — 경계 분할 로직은 있으나(§3-3) 하나의
   동작이 두 청크로 갈려 별개 관찰이 되는 사례 빈도는 코드로 측정 불가.
6. **`iou`/일부 YOLO 파라미터의 실제 사용값** — config.yaml에 `iou` 미노출이라 코드 기본값(0.7)이
   쓰인다고 추정되나, config 병합 후 최종값을 런타임 로그로 확정하진 못함(코드 기본값 기준 서술).
7. **`debug_vlm.py`/`scripts/debug_*` 스크립트의 최신성** — 디버그용이며 본선 경로가 아니라 현재
   동작 여부는 미검증(본선은 run_step4/run_step4_motion).

---

*(끝) 본 보고서는 조사·설명 전용이며 코드를 수정하지 않았다.*
