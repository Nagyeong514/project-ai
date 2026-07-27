# STEP3(전처리 + STT) 상세 기술 보고서

> 목적 (2가지)
> 1. **재디벨롭 참고 문서** — 이 단계를 다시 손볼 때 설계 근거·파라미터·손잡이를 바로 찾는다.
> 2. **트러블슈팅 아카이브** — OOM·디코딩 실패 등 개발 중 시간을 잡아먹은 사건과 해결책 기록.
>
> **작성 규칙(엄수):** 코드를 실제로 따라가 확인된 것만 기술한다. 코드로 확인 불가능한 항목은
> 본문에 "**(확인 불가)**"로 명시하고, 보고서 말미(§7)에 목록으로 다시 모은다. 모든 주장에
> 근거 위치(파일:라인 / config 키 / git 커밋)를 붙인다.
> 작성 기준 시점: 2026-07-16. 기준 커밋: `45be634`.

---

## §0. 이 보고서에서 말하는 "STEP3"의 범위

`STEP3_전처리/`가 담당하는 것은 딱 두 가지다(`step3_runner.py:2` 주석).

1. **오디오 축**: STT(Whisper) → transcript 정제(정규화 + 반복/환각 태깅)
2. **비디오 축**: 프레임 추출(uniform | motion), STEP4(YOLO/VLM)가 소비할 `frames_meta.json` 생성

YOLO 검출·VLM 관찰은 STEP4, 정렬·LLM 융합은 STEP5 소유다(2026-07-04 분할, README.md:62-65).
단, §5의 트러블슈팅 사건 중 "VLM fps OOM"은 사용자 요청에 따라 STEP4 경계를 넘어 다루되,
STEP3의 프레임 추출 fps(`frame_extraction`)와 STEP4의 VLM 추론 fps는 **서로 다른 손잡이**임을
명확히 구분해 서술한다(§4·§5-3).

---

## §1. 전체 구조 개관

### 1-1. 엔트리포인트 → 산출물 텍스트 순서도

STEP3에 들어오는 진입점은 3가지다.

| 진입점 | 파일 | 용도 |
|---|---|---|
| 통합 실행(전체) | `파이프라인_통합실행/run_all_clips.py` | CLIP1~4 순차, 모델 1회 로드(`run_all_clips.py:44-55`) |
| 통합 실행(1개) | `파이프라인_통합실행/run_one_clip.py` | 클립 1개 STEP3→4→5(`run_one_clip.py:36-38`) |
| STEP3 단독 | `STEP3_전처리/run_step3.py` | STEP3만(디버깅/재작업, `run_step3.py:30-32`) |
| 오디오만 | `STEP3_전처리/run_audio_extract.py` | 영상→WAV만(STT와 무관한 별도 유틸, §3-5 주의) |

원샷 오케스트레이터 `파이프라인_통합실행/main.py`는 STEP3~7을 각 STEP venv로 subprocess
호출하는 최상위 진입점이며, 그 [1/3] 단계가 `run_all_clips.py`를 호출한다(`main.py:78`).

```
[입력] 원본 영상 파일 경로 (예: master/master_videos/CLIP1_정상조립과정.mp4)
   │   video_id = Path(video_path).name   ← 확장자 포함 파일명 그대로 (step3_runner.py:35)
   │
   ├─────────────── [오디오 축] ───────────────────────────────────────────
   │   Step3Runner.run() (step3_runner.py:37-42)
   │     stt.transcribe(video_path, video_id)         ← 원본 영상 경로를 그대로 STT에 전달
   │        WhisperTurboSTT (stt_whisper.py:96-144)   faster-whisper large-v3-turbo
   │        · vad_filter=True (무음 스킵, 타임스탬프는 원본 유지)
   │        · seg.start/seg.end = 원본 오디오 초(float)
   │     → Transcript(utterances=[Utterance(start,end,raw_text,normalized_text)])
   │        (transcribe() 안에서 정제 前 스냅샷을 1차 저장 — stt_whisper.py:143,146-153)
   │     refiner.refine(transcript)                   NormalizeRefiner (transcript_refine.py:76-105)
   │        · normalized_text = 용어정규화(마더보이드→motherboard 등)
   │        · repeat_hallucination = 연속반복 + 환각상투구 태깅
   │     artifacts.save_transcript(...)               ← 정제본으로 덮어쓰기 (step3_runner.py:40)
   │     stt.unload()                                  ← GPU 즉시 해제 (step3_runner.py:41-42)
   │   [산출] transcripts/<video_id>.json
   │
   └─────────────── [비디오 축] ───────────────────────────────────────────
       sampling.extract(video_path, video_id, frames_dir, frame_extraction)
          impl="uniform" → UniformFrameSource (frame_source.py:39-56)
          impl="motion"  → MotionSampledFrameSource (frame_source.py:59-217)
             ffmpeg CLI로 jpg 추출 (frame_extract.py:35-59)
       → {frame_paths, times, fps, duration, segments}
       artifacts.save_frames_meta(...)               (step3_runner.py:47-50)
   [산출] output/_frames/<video_id>/frame_%04d.jpg + frames_meta.json

[반환] video_id  ← STEP4/5가 이 문자열로 파일을 찾는다 (step3_runner.py:34,55)

────────── 이후 (STEP3 밖) ──────────
STEP4: frames_meta.json + jpg → YOLO 검출 + VLM 관찰
STEP5: transcript + 관찰 + 검출을 ±4초 윈도우로 정렬(aligner) → LLM 융합
```

### 1-2. 오디오/비디오 축이 갈라지고 다시 만나는 지점

- **분기점**: `Step3Runner.run()`. 한 함수 안에서 오디오(STT) 처리(`step3_runner.py:37-42`)와
  비디오(프레임) 처리(`step3_runner.py:44-50`)가 **완전히 독립적으로** 진행된다. 두 축은
  STEP3 내부에서 서로의 산출물을 참조하지 않는다.
- **공용 시간축**: 두 축 모두 산출물 timestamp를 **원본 영상 기준 초(float)** 로 맞춘다.
  - 오디오: `seg.start/seg.end` (원본 오디오 초, `stt_whisper.py:113-118`)
  - 비디오: `times[i]` (원본 영상 초, uniform은 `i/fps`, motion은 `start_sec + local_t`)
- **합류점(STEP3 밖)**: STEP5의 aligner가 두 축의 초(float) timestamp를 `±4초` 윈도우로
  묶는다(`config.yaml` `aligner.params.window_sec: 4.0`). 즉 **STEP3의 임무는 "두 축의 시각을
  같은 좌표계(원본 초)로 내보내는 것"** 이고, 실제 결합은 STEP5가 한다.

### 1-3. 디렉토리 / 파일 산출물 맵

| 산출물 | 경로 | 저장 함수 | 스키마(키) |
|---|---|---|---|
| STT transcript(정제본) | `STEP3_전처리/transcripts/<video_id>.json` | `artifacts.save_transcript` (artifacts.py:57-62) | `{video_id, language, model, utterances:[{start,end,raw_text,normalized_text,repeat_hallucination}]}` |
| 프레임 jpg | `STEP3_전처리/output/_frames/<video_id>/frame_%04d.jpg` | `extract_frames` (frame_extract.py:52-56) | (이미지 파일) |
| 프레임 메타 | `STEP3_전처리/output/_frames/<video_id>/frames_meta.json` | `artifacts.save_frames_meta` (artifacts.py:23-40) | `{video_id, fps, duration, frame_paths[], times[], (segments[])}` |
| (motion 전용) 서브클립 mp4 | motion_root의 `<stem>/clip_XX_*.mp4` + `clip_timestamps.csv` | (팀원 사전산출물 또는 auto_generate) | CSV: `clip_no,filename,start_sec,end_sec,anchor_sec,...` |
| (별도유틸) 오디오 WAV | `STEP3_전처리/output/audio/<video_id>.wav` | `extract_audio` (audio_extract.py:18-45) | 16kHz mono PCM WAV |

- `video_id`는 **확장자 포함 원본 파일명 그대로**(`Path(video_path).name`, `step3_runner.py:35`).
  그래서 `CLIP3_재부팅시도 전까지.mp4`처럼 공백/한글이 포함된 이름이 산출물 파일명에도
  그대로 남는다(경로 이스케이프 없음, OS 파일시스템이 공백/UTF-8을 허용하는 데 의존).
- 프레임은 `output/_frames/<video_id>/` **폴더 단위**로 분리 저장된다 — 예전 "모든 클립이 한
  폴더에 덮어써지던 버그"의 수정판(`step3_runner.py:6` 주석). 또한 매 추출마다 이전
  `frame_*.jpg`를 전부 삭제하고 새로 뽑는다(`frame_extract.py:48-49`) — 이전 클립 프레임이
  다음 추론에 섞이는 사고 방지.

### 1-4. 설계 원칙과 코드상 보장 지점

| 원칙 | 코드상 보장 지점 |
|---|---|
| **모든 timestamp = 원본 영상 좌표계** | 오디오: `vad_filter`가 오디오를 자르지 않아 `seg.start`가 원본 기준(§3-4). 비디오 uniform: `times=[i/fps]`가 원본 0초 기준(`frame_extract.py:58`). 비디오 motion: `collected.append((start_sec + t, ...))` 단 한 곳에서 로컬→원본 보정(`frame_source.py:166`), 사후 범위 assert(`frame_source.py:204-209`). |
| **timestamp는 한 번만 부여, 이후 그대로 운반** | `frame_extract.py:44-45` 주석("시간은 여기서 한 번만 부여"), 중간 산출물 공통 초 단위(`intermediate.py:8` 원칙 3). VLM은 시각을 재계산하지 않고 그리드로 스냅(`vlm_qwen.py:30-38` `snap_to_grid`). |
| **모션가이드는 비디오 축 전용, 오디오 축 독립** | `sampling`은 `Step3Runner`의 비디오 처리(`step3_runner.py:44-45`)에서만 소비. STT는 `sampling`을 참조하지 않고 항상 원본 `video_path`를 받음(`step3_runner.py:38`). 이 불변식은 별도 메모리(모션클립 STT 금지)로도 강제. |
| **모델 교체 = config 한 줄** | `step3_registry.py`의 딕셔너리(STTS/REFINERS/SAMPLING) + `config.impl`. 새 구현=클래스1개+레지스트리 한 줄. |
| **스키마 = 단일 진실 공급원** | 중간 타입은 `tacit_common/schema/intermediate.py`에서만 정의(`Transcript`, `Utterance`, `FrameMeta`). |

**근거 위치(§1):** `step3_runner.py:33-55`, `frame_extract.py:35-59`, `frame_source.py:166,204-209`,
`stt_whisper.py:96-144`, `tacit_common/artifacts.py:19-62`, `tacit_common/schema/intermediate.py:106-127`,
`README.md:41-56`.

---

## §2. 공통 앞단

### 2-1. 입력 영상 로드 / 클립 관리 / demux 위치

- **입력 영상 목록**: `run_all_clips.py:25-31`에 **4개 절대경로가 하드코딩**돼 있다.
  ```python
  VIDEO_DIR = "/home/ai_user/team_a2/members/안나경/master/master_videos"
  VIDEOS = [f"{VIDEO_DIR}/CLIP1_정상조립과정.mp4", ".../CLIP2.mp4",
            ".../CLIP3_재부팅시도 전까지.mp4", ".../CLIP4_재부팅및BIOS.mp4"]
  ```
  디렉토리 스캔·확장자 필터는 없다. 처리 대상은 이 리스트(전체) 또는 `--video`(1개)로
  **명시 지정**된다. 4개 파일 실재 확인(2026-07-16): 전부 존재(CLIP1 277MB, CLIP2 217MB,
  CLIP3 55MB, CLIP4 320MB).
- **클립 단위 관리(CLIP1~4)**: "클립"은 코드상 특별한 타입이 아니라 **영상 파일 1개 = 1클립**이다.
  `video_id`가 유일 키. (motion 모드에서 한 영상을 다시 서브클립으로 쪼개는 것은 §4-3의
  별개 개념이다 — 그건 비디오 축 내부의 샘플링 산출물이고, 시각은 원본으로 복원된다.)
- **demux(오디오-비디오 분리) 위치**: STEP3 본선(`step3_runner.py`)에는 **명시적 demux 단계가
  없다.** 오디오/비디오 분리는 각 소비자 내부에서 일어난다.
  - 오디오: `stt.transcribe(video_path)`에 **영상 파일 경로를 그대로** 넘긴다
    (`stt_whisper.py:101`). faster-whisper가 내부적으로 ffmpeg로 오디오 트랙을 뽑는다.
    STEP3 코드에는 별도 리샘플링/포맷 변환이 없다.
  - 비디오: `extract_frames`가 ffmpeg CLI로 프레임 jpg만 뽑는다(`-i video`, 아래 옵션 참고).
  - **명시적 demux가 필요한 곳은 별도 유틸 `audio_extract.py`뿐**이고, 이건 본선 러너가
    호출하지 않는다(§3-5).
- **사용 도구 = ffmpeg CLI subprocess.** 프레임 추출 실제 명령(`frame_extract.py:50-56`):
  ```
  ffmpeg -y -v error -i <video> \
    -vf "fps=<fps>,scale='if(gt(iw,ih),480,-2)':'if(gt(iw,ih),-2,480)'" \
    -q:v 5  <out_dir>/frame_%04d.jpg
  ```
  - `fps=<fps>`: 초당 추출 프레임(현재 0.5 = 2초에 1장).
  - `scale=...`: **긴 변을 480px로** 맞추고 짧은 변은 비율 유지(`-2`=2의 배수로 자동).
    가로영상이면 폭 480, 세로영상이면 높이 480.
  - `-q:v 5`: JPEG 품질(1~31, 낮을수록 고품질). 하드코딩(`jpeg_q=5`, config 미노출).
  - 오디오 추출 명령(`audio_extract.py:30-39`, 별도 유틸):
    `ffmpeg -y -i <video> -vn -acodec pcm_s16le -ar 16000 -ac 1 -loglevel error <out.wav>`.

### 2-2. config 로드 지점

- 로더: `tacit_common/config.py`의 `PipelineConfig.load()` (config.py:96-105). YAML → pydantic
  `BaseModel` 검증.
- 호출 지점: 각 진입점의 프리플라이트가 `check_config()`를 통해 로드
  (`preflight_base.py:40-47`, `run_step3.py:30`이 `step3_preflight.run_preflight()` 호출).
- 상대경로 기준: `PROJECT_ROOT = config.py의 부모의 부모` = `project-ai/` 루트(config.py:19).
  `paths:`의 상대경로는 `PathsConfig.resolve()`가 이 루트를 붙여 절대경로화(config.py:56-57).
  cwd가 STEP3든 통합실행이든 같은 절대 위치를 보게 하기 위함.

### 2-3. STEP3 관련 config 키 전수 목록 (재디벨롭용 손잡이)

값은 실행용 `파이프라인_통합실행/config.yaml`(2026-07-15) 기준. 정의 위치는 pydantic 모델.

#### (a) `frame_extraction` — 프레임 추출 정책 (config.py:29-40)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `fps_short` | `0.5` | 짧은 영상(<threshold)용 추출 fps | 올리면 프레임↑(관찰 밀↑, VRAM/토큰↑), 낮추면 반대 |
| `fps_long` | `0.5` | 긴 영상(≥threshold)용 추출 fps | 상동. **주의: 낮추면 VLM 관찰 0건 동결 버그 재발**(§5-3) |
| `long_video_threshold_sec` | `120.0` | 이 값 이상이면 `fps_long` 적용 | 짧/긴 영상 경계 이동 |
| `fps_override` | `null` | 지정 시 길이 무관 항상 이 fps | 값 주면 short/long 분기 무시 |
| `long_side` | `480` | 프레임 긴 변 픽셀 | 올리면 작은 부품 검출↑·VRAM↑, 낮추면 오검출 위험(config 주석: max_pixels는 일부러 안 건드림) |
| `ffmpeg_bin` | `.../imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2` | ffmpeg 바이너리 절대경로 | 서버 바뀌면 여기 수정(§5-1) |

> **현재 `fps_short == fps_long == 0.5`** 이므로 길이별 자동 분기는 실질적으로 무효(항상 0.5).
> 분기 메커니즘(`select_fps`, `frame_source.py:31-36`)은 살아있고 값만 같다.

#### (b) `sampling` — 프레임 소스 스위치 (config.py:87)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `sampling.impl` | `"uniform"` | `uniform`=균등추출(기준선) / `motion`=모션가이드 산출물 소비 | `motion`으로 바꾸면 §4-3 경로. **단 motion_root 기본경로가 stale**(§6·§7) |
| `sampling.params.motion_root` | (미지정→기본값) | motion CSV/서브클립 루트 | 실제 산출물 위치로 지정 필요 |
| `sampling.params.dedup_window_sec` | (기본 0.5) | 중첩 구간 프레임 제거 여유(초) | §4-3 dedup 민감도 |
| `sampling.params.auto_generate` | (기본 False) | CSV 없는 신규 영상 자동 샘플링 | True면 팀원 코드로 서브클립 생성(§4-3) |

#### (c) `stt` — Whisper (config.py:91, stt_whisper.py:35-46)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `stt.impl` | `"whisper_turbo"` | 레지스트리 키(현재 유일) | 새 STT는 registry+클래스 추가 |
| `stt.params.model_name` | `"mobiuslabsgmbh/faster-whisper-large-v3-turbo"` | 모델 ID | 다른 Whisper 변형으로 교체 |
| `stt.params.backend` | `"faster_whisper"` | `faster_whisper`↔`openai_whisper` (transcribe에 완전 분리 경로) | openai_whisper면 vad_filter 등 미적용(§3) |
| `stt.params.device` | `"cuda"` | 추론 디바이스 | `cpu`면 검증용(느림) |
| `stt.params.compute_type` | `"float16"` | 연산 정밀도. **Turing=fp16만**(bf16/FP8 금지) | int8=cpu 검증용이었음 |
| `stt.params.language` | `"ko"` | 언어 고정 | None이면 자동감지 |
| `stt.params.condition_on_previous_text` | `false` | 이전 세그먼트를 컨텍스트로 쓸지 | True면 반복/환각 전파 위험↑(§3-2) |
| `stt.params.vad_filter` | `true` | faster-whisper 내장 VAD(무음 스킵, 타임스탬프 원본유지) | False면 무음 구간 환각↑(§3-1) |
| `stt.params.vad_parameters.threshold` | `0.5` | Silero VAD 임계값 | config 주석: 0.2로 낮춰도 효과 무의미(STEP1 재검증) |
| `stt.params.transcript_dir` | `"transcripts"` | 저장 폴더(상대경로) | **런타임엔 runner가 절대경로로 덮어씀**(step3_runner.py:28-31) |

#### (d) `transcript_refine` — 정제 (config.py:92, transcript_refine.py:49-60)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `transcript_refine.impl` | `"normalize"` | 레지스트리 키 | (구 RegexRefiner 근거태깅은 폐기) |
| `.params.normalization_dict_path` | `null` | 정규화 사전 경로 | null이면 `resources/en_normalization.json` 기본(transcript_refine.py:63-64) |
| `.params.flag_repetitions` | `true` | 반복/환각 태깅 on/off | False면 `repeat_hallucination` 항상 False |
| `.params.hallucination_phrases` | (미지정→기본목록) | 환각 상투구 denylist | 목록 지정 시 덮어씀(transcript_refine.py:59) |

#### (e) `paths` — STEP3 산출물 경로 (config.py:43-54)

| 키 | 현재 값 | 의미 |
|---|---|---|
| `paths.step3_frames_dir` | `"STEP3_전처리/output/_frames"` | 프레임+메타 루트(PROJECT_ROOT 기준) |
| `paths.step3_transcript_dir` | `"STEP3_전처리/transcripts"` | transcript 저장 루트 |

> `fps_override`(최상위, config.py:65)와 `frame_extraction.fps_override`는 **다른 것**이다.
> 전자는 `FrameMeta.fps` 폴백용(STEP5), 후자는 ffmpeg 추출 fps 정책용(config.py:69-71 주석).

**근거 위치(§2):** `run_all_clips.py:25-31`, `frame_extract.py:50-56`, `audio_extract.py:30-39`,
`tacit_common/config.py:19,29-105`, `stt_whisper.py:35-46`, `config.yaml`(전 섹션),
`step3_runner.py:28-31`.

---

## §3. 오디오 축: VAD → STT

### 3-1. VAD (Voice Activity Detection)

- **방식**: 별도 VAD 라이브러리를 STEP3 코드에서 직접 호출하지 않는다. **faster-whisper 내장
  `vad_filter=True`** 를 켠다(`stt_whisper.py:107`, `vad_filter=self.vad_filter`).
- **엔진/파라미터**: faster-whisper의 내장 VAD는 Silero 기반. 파라미터는
  `vad_parameters={threshold: 0.5}`로 전달(`config.yaml` `stt.vad_parameters`,
  `stt_whisper.py:108` `vad_parameters=self.vad_parameters or None`).
- **동작(중요)**: 이 내장 필터는 **오디오를 잘라 이어붙이지 않는다.** 무음 구간을 디코딩에서만
  스킵하고, 세그먼트 `start/end`는 **원본 오디오 타임라인 기준으로 그대로 리포트**한다.
  (`stt_whisper.py:11-17` 주석에 명시.)
- **왜 "자르기+이어붙이기" 방식을 안 쓰는가(설계 근거)**: STEP1의 VAD 재검증 실험
  (`STEP1_모델연구/04_STT_VAD강건성/02_VAD재검증_v2/`)에서 Silero로 발화 구간만 잘라
  이어붙이면 결과 오디오가 짧아져(발화 비율 CLIP1~4 = 7~22%) `seg.start/end`가 원본
  타임라인과 어긋난다는 걸 확인. STEP5 aligner가 발화·행동을 ±윈도우로 직접 매칭하므로
  이 어긋남이 그대로 정렬 오류가 된다. → **트리밍 방식 폐기, 내장 필터 채택**(stt_whisper.py:11-17).
  실측: CER은 병합 방식과 동급 이상(평균 0.2362→0.1813), 세그먼트가 원본 전 구간에 분포.
- **threshold 튜닝 메모**: config 주석 — "0.2로 낮춰도 효과 무의미"(STEP1 06 재검증). 이 실험
  수치 자체는 STEP1 보고서 소관이고, STEP3 코드는 config 값을 그대로 faster-whisper에 전달만 한다.

### 3-2. STT

- **모델**: `faster-whisper large-v3-turbo`. config 실제 값 =
  `"mobiuslabsgmbh/faster-whisper-large-v3-turbo"`(HF 리포). 코드 기본값은 `"large-v3-turbo"`
  (`stt_whisper.py:37`).
- **로딩**: 지연 로딩. `transcribe()` 최초 호출 시 `_load()`가 `faster_whisper.WhisperModel(...)`
  생성(`stt_whisper.py:60-69`). 모듈 최상단 import 금지(CLAUDE.md §1) — preflight가 5초 안에
  죽을 수 있게.
- **백엔드 2경로**: `faster_whisper`(본선) / `openai_whisper`. `transcribe()` 안 if/else로 완전
  분리(`stt_whisper.py:100-138`).

**핵심 STT 옵션과 그 값을 정한 이유:**

| 옵션 | 값 | 근거 위치 | 이유 |
|---|---|---|---|
| `language` | `"ko"` | stt_whisper.py:104 | 한국어 고정(자동감지 오버헤드/오탐 제거) |
| `condition_on_previous_text` | `False` | stt_whisper.py:41,104 | **반복 환각 억제 목적** — 이전 세그먼트를 컨텍스트로 안 써서 반복 루프/환각 전파 차단. 파일 상단 docstring "스펙 5.4"·주석 "스펙 기본값", 심층분석 doc §STT-137에도 동일 서술 |
| `word_timestamps` | `False` | stt_whisper.py:105 | **하드코딩**(config 미노출). 세그먼트 단위 타임스탬프만 필요, 단어 단위는 요청 안 함 |
| `vad_filter` | `True` | stt_whisper.py:107 | §3-1 |
| `vad_parameters` | `{threshold:0.5}` | stt_whisper.py:108 | §3-1 |
| `compute_type` | `"float16"` | stt_whisper.py:39 | Turing(sm75) fp16만 지원 |

- **`repeat_hallucination` 필드 판정·기록 위치**: STT가 아니라 **정제기(NormalizeRefiner)** 가
  판정한다(`transcript_refine.py:82-98`). 두 조건 중 하나면 `True`:
  1. **(a) 직전 발화와 정규화 키가 동일**하면 반복(Whisper 끝부분 자기복제 환각 의심)
     — `transcript_refine.py:85-86`.
  2. **(b) 환각 상투구 denylist 매칭** — `transcript_refine.py:88-89`. denylist =
     `DEFAULT_HALLUCINATION_PHRASES`(transcript_refine.py:32-40): "다음 영상에서 만나요",
     "시청해 주셔서 감사합니다", "감사합니다", "구독", "아멘" 등(무음 구간 상투 환각, 4클립 실측).
  - **삭제하지 않고 플래그만** 단다(원문·타임스탬프 보존, transcript_refine.py:8). 융합
    LLM이 무시 힌트로 쓴다. 다운스트림 소비 확인: STEP5가 `repeat_hallucination=True`
    발화를 융합 입력에서 제외(`STEP5_.../step5_components/llm_fusion.py:256-257,299-300` 등).

- **정규화(normalized_text) 생성**: `NormalizeRefiner._normalize()`(transcript_refine.py:69-74).
  `resources/en_normalization.json`의 오인식 사전으로 치환. **긴 키부터, 대소문자 무시**
  (부분매칭 충돌 방지). 현재 활성 항목은 `마더보이드/마더보드/메인보드 → motherboard` 3건만
  (나머지는 `_candidates`로 비활성 — 추측 치환이 원문 훼손하는 것 방지, en_normalization.json:11).

### 3-3. 최종 transcript 스키마 (Transcript / Utterance, intermediate.py:106-127)

| 필드 | 타입 | 의미 |
|---|---|---|
| `video_id` | str | 원본 파일명(확장자 포함) |
| `language` | str? | STT가 반환한 언어(보통 "ko") |
| `model` | str? | 사용 모델명 |
| `utterances[]` | list | 발화 세그먼트 목록 |
| `utterances[].start` | float | 시작 초(원본 좌표계) |
| `utterances[].end` | float | 끝 초(원본 좌표계) |
| `utterances[].raw_text` | str | **STT 원문(절대 손실 금지)** |
| `utterances[].normalized_text` | str | 용어 정규화 적용본 |
| `utterances[].repeat_hallucination` | bool | 반복/환각 태그(기본 False) |

**실제 파일 발췌** (`transcripts/CLIP2.mp4.json`, 3~5줄):
```json
{"start": 80.46, "end": 89.94, "raw_text": "램은 슬롯이 총 6개로 구분되어 있는데 색상으로 쉽게 구분을 할 수가 있습니다", "normalized_text": "램은 슬롯이 총 6개로 구분되어 있는데 색상으로 쉽게 구분을 할 수가 있습니다", "repeat_hallucination": false}
{"start": 89.94, "end": 95.3,  "raw_text": "램을 뺄때는 양쪽을 동시에 눌러 줘야 합니다", "normalized_text": "램을 뺄때는 양쪽을 동시에 눌러 줘야 합니다", "repeat_hallucination": false}
{"start": 95.3,  "end": 99.98, "raw_text": "안그러면은 슬롯이 파손될 수가 있습니다", "normalized_text": "안그러면은 슬롯이 파손될 수가 있습니다", "repeat_hallucination": false}
```
> 위 예시에서 "램"(RAM 오인식)이 `normalized_text`에도 그대로 남아 있다 — 현재 활성 정규화
> 사전에 "램"이 없기 때문(`_candidates`에 비활성 대기, en_normalization.json:5,23). **일반어
> 충돌** 위험(랩=wrap/lab, 매치=match) 때문에 의도적으로 비활성이며, 맥락 해소는 융합 LLM에 위임.

> **(확인 불가/이상 징후)** `transcripts/CLIP2.mp4.json`의 일부 발화(상위 2건)와
> `transcripts/_merged_all.json`에 **`split_from_original: true`** 라는 필드가 있다. 이 필드는
> 현재 STEP3 코드 어디서도 생성되지 않으며(`grep` 결과 0건), `Utterance` 모델에도 없다
> (intermediate.py:106-119). CLIP1/3/4에는 없다. → **수동 편집 흔적으로 추정.** 코드 재실행 시
> 이 필드는 사라진다. `_merged_all.json`을 만드는 스크립트도 저장소에 없다(§7).

### 3-4. 좌표계 검증 (이 축의 입력이 원본 오디오 전체임을 코드로 확인)

1. **입력 = 원본 영상 전체**: `Step3Runner.run()`이 `self.stt.transcribe(video_path, video_id)`를
   호출하고(`step3_runner.py:38`), `video_path`는 진입점에서 받은 **원본 영상 경로**
   (`run_all_clips.py:52` `step3.run(video)`, `video`∈`VIDEOS` 원본 4클립). motion 서브클립·프레임
   등 어떤 가공물도 STT에 들어가지 않는다. faster-whisper가 이 원본에서 오디오 전체를 디코딩한다.
2. **타임스탬프 = 별도 보정 없음**: `stt_whisper.py:113-118`에서 `Utterance(start=float(seg.start),
   end=float(seg.end), ...)` — faster-whisper가 준 값을 **그대로** 넣는다. 오프셋 가감·스케일링
   코드가 없다. `vad_filter`는 무음을 스킵할 뿐 좌표를 옮기지 않으므로(§3-1), `seg.start`는
   원본 오디오 초 그대로다. 실측 근거: `CLIP2.mp4.json` 첫 발화 `start=24.27`처럼 0이 아닌 값이
   원본 영상 24초 지점을 그대로 가리킨다(트리밍 방식이었다면 0 근처로 쏠렸을 것).

### 3-5. (주의) `audio_extract.py` / `run_audio_extract.py`는 본선이 아니다

- `run_audio_extract.py`는 영상→16kHz mono WAV(`output/audio/<id>.wav`)를 만드는 **독립 유틸**
  이다(GPU 불필요, 로그인 노드 실행 가능). **본선 러너(`step3_runner.py`)는 이 WAV를
  사용하지 않는다** — STT는 원본 영상 경로를 직접 받는다(§3-4). `output/audio/*.wav` 4개가
  존재하지만 파이프라인 데이터 흐름상 소비처가 없다(디버그/사전연구 잔재로 보임).

**근거 위치(§3):** `stt_whisper.py:11-17,37-46,60-69,100-138`, `transcript_refine.py:8,32-40,69-74,82-98`,
`tacit_common/schema/intermediate.py:106-127`, `resources/en_normalization.json`, `config.yaml` stt 섹션,
`STEP5_.../llm_fusion.py:256-257`, 실제 파일 `transcripts/CLIP2.mp4.json`, `run_audio_extract.py`.

---

## §4. 비디오 축: 프레임 추출 + 모션가이드 샘플링

### 4-1. sampling 분기 (uniform | motion)

- 스위치: `config.sampling.impl`. 레지스트리 매핑(`step3_registry.py:19-22`):
  `"uniform" → UniformFrameSource`, `"motion" → MotionSampledFrameSource`. 빌드는
  `build_sampling()`(step3_registry.py:42-43), 실행은 `Step3Runner.run()`의
  `self.sampling.extract(...)`(step3_runner.py:45).
- 두 구현 모두 같은 인터페이스: `extract(video_path, video_id, frames_dir, fe_cfg)
  → {frame_paths, times, fps, duration, segments}`. 차이는 `segments`(uniform=None, motion=구간목록)와
  프레임 시각 산출 방식.
- **현재 config는 `uniform`** 이지만, **디스크의 frames_meta.json 4개는 `segments`를 포함**한다
  (2026-07-16 확인) → 마지막 실행은 **motion 모드**였다. times가 0이 아니라 첫 segment의
  `start_sec`부터 시작(예 CLIP2 `times[0]=16.0` = `segments[0].start_sec=16.0`)하는 것으로
  motion 좌표복원이 실제로 동작했음이 확인된다. (모션 채택 배경: 2026-07-07 4클립 전부 모션
  우세, frame_source.py:5-11.)

### 4-2. uniform 경로

- `UniformFrameSource.extract()`(frame_source.py:46-56):
  1. `dur = probe_duration(video)` — ffmpeg `-i`의 stderr에서 `Duration:` 파싱(frame_extract.py:24-32).
  2. `fps = select_fps(fe_cfg, dur)` — fps 결정 로직(frame_source.py:31-36):
     ```python
     if fe_cfg.fps_override is not None: return fe_cfg.fps_override
     return fe_cfg.fps_long if dur >= fe_cfg.long_video_threshold_sec else fe_cfg.fps_short
     ```
     **길이별 매핑 테이블(현재 값 기준):**

     | 조건 | 사용 fps |
     |---|---|
     | `fps_override` 지정 | 그 값(현재 null) |
     | `duration ≥ 120.0s` | `fps_long` = **0.5** |
     | `duration < 120.0s` | `fps_short` = **0.5** |

     → 현재는 길이와 무관하게 **항상 0.5fps**(short==long). "자동조정 함수"는 있으나 값이 같아
     실질 무효.
  3. `extract_frames(video, fps, out_dir, long_side=480)` — §2-1의 ffmpeg 명령 실행.
  4. `times = [i/fps for i in range(len(paths))]`(frame_extract.py:58) — **0초 기준 등간격**.
     원본 좌표계 = 원본이 곧 입력이므로 보정 불필요.

### 4-3. motion 경로

- **알고리즘 요약** (팀원 사전 산출물 소비형 어댑터):
  1. `MotionSampledFrameSource.extract()`(frame_source.py:119-217)는 `motion_root/<stem>/
     clip_timestamps.csv`와 서브클립 mp4를 읽는다. 오프셋의 **단일 진실 = CSV 컬럼**(start_sec 등),
     파일명 파싱 금지(반올림 불일치 실측, frame_source.py:63-64).
  2. 각 서브클립을 `extract_frames`로 로컬 프레임 추출 → 로컬 시각 `t`에
     **`start_sec + t`** 를 더해 원본 좌표로 복원(frame_source.py:161-166).
  3. 중첩 구간 dedup: 시간근접이 아니라 **구간중첩** 기준. "직전 clip의 `end_sec +
     dedup_window_sec` 이하면 버림"(frame_source.py:180-189). 실측 CLIP4에서 홀/짝 그리드
     어긋남으로 시간근접 기준이 안 걸러진 사건의 수정판(frame_source.py:170-179 주석).
  4. 채택 프레임을 원본 시각순 `frame_%04d.jpg`로 재명명 복사, 임시 폴더 정리
     (frame_source.py:191-201).
  5. **좌표 검증 assert**: 모든 times가 어떤 segment 범위 안에 드는지 확인, 아니면 AssertionError
     (frame_source.py:204-209) — 조용한 오프셋 실수 즉시 탐지.

- **CSV(`clip_timestamps.csv`) 역할**: 서브클립↔원본 좌표 매핑의 단일 진실. 컬럼 =
  `clip_no, filename, start_sec, end_sec, anchor_sec, start_ts, end_ts, duration_sec, motion_score`
  (extract_video.py:70-80). `start_sec`이 로컬→원본 보정의 유일 오프셋 소스.

- **모션 샘플링 원 알고리즘**(서브클립을 처음부터 만들 때, `auto_generate=True` 또는 원 프로토타입):
  - `motion_score.py`: 1초 간격 샘플링 → 인접 프레임 **absdiff 평균**을 motion score로. GaussianBlur로
    노이즈 억제, `noise_threshold=1.5` 미만은 0 처리(motion_score.py:12-78).
  - `select_clip.py`: 3전략 비교 후 **종합점수 최고 전략 자동 선택**(select_clip.py:144-238).
    - A: Threshold+AKS(상위75% 임계 → ±window → 병합 → 최소간격 필터)
    - B: CDF 역변환 샘플링(MGSampler, 모션 밀도 비례)
    - C: AKS 시간균등분할 + 윈도우 ArgMax
    - 종합점수 = `0.40*커버리지 + 0.35*(모션관련도/100) + 0.25*균등성`(select_clip.py:66-70).
  - `extract_video.py`: 선택 구간을 ffmpeg 재인코딩(libx264, crf20, `-an` 오디오 제거)으로 서브클립
    추출 + CSV 기록(extract_video.py:31-92). `n_clips=max(3, dur/40)`, 유지율 70% 목표로 윈도우
    자동 확대(abc_pipeline_runner.py:155-171).

- **이식 이력** (`motion_sampling/__init__.py:1-11`):
  - 원본: `00_사전연구/04_모션가이드샘플링_프로토타입/01_코드/`(팀원 제공). 이식일 2026-07-08,
    계획서 `docs/계획서_모션가이드_STEP3_이식.md`.
  - **알고리즘 무변경**: `abc_pipeline_runner.py`의 import 3줄(상대경로 전환)만 바뀌고,
    `motion_score.py`/`select_clip.py`/`extract_video.py`는 **원본과 바이트 단위 동일**하다고 명시.
  - **(확인 불가)** 원 작성자 실명은 `__init__.py:2`에서 "계획서 승인 후 실행 시점에 확정"으로
    남겨져 있고 채워지지 않았다.
  - **(확인 불가/경로 드리프트)** `__init__.py` 주석과 config의 `motion_root` 기본값은 여전히
    `00_사전연구/04_...`를 가리키지만, 실제 산출물은 폴더 정리로
    `STEP1_모델연구/01_모션기반샘플링/02_모션가이드샘플링_프로토타입/04_샘플링 영상/`으로 이동했다
    (2026-07-16 확인). 즉 **motion 모드를 지금 그대로 켜면 기본 motion_root가 존재하지 않아
    FileNotFoundError** — `sampling.params.motion_root`를 새 경로로 지정해야 한다(§6·§7).

### 4-4. 좌표 보정 감사 (이중 보정 위험 지점)

- **로컬→원본 보정은 코드 전체에서 정확히 한 곳**: `frame_source.py:166`
  `collected.append((r["start_sec"] + t, r["clip_no"], p))`. 이 외에 times에 오프셋을 더하거나
  빼는 곳은 없다(uniform은 애초에 원본=입력이라 보정 없음).
- **이중 보정 없음 근거**:
  - motion에서 만든 `times`(이미 원본 좌표)를 STEP4/5가 그대로 소비한다. STEP4는 `frames_meta`의
    `times`를 그대로 VLM 관찰 시각으로 부착(VLM은 시각 재계산 안 함, `vlm_qwen.py:30-38`
    `snap_to_grid`는 그리드 값으로 스냅만).
  - VLM이 뱉은 시각 문자열을 초로 되돌릴 때 `hhmmss_to_seconds`(intermediate.py:29-48)를 쓰지만,
    이는 STEP4 내부 변환이고 STEP3의 `start_sec` 보정과 무관(서브클립 offset을 다시 더하지 않음).
  - `frames_meta`의 `segments`는 원본 좌표 그대로 저장되고(artifacts.py:36-37), 하류에서 재보정
    코드가 없다.
- **감사 결론**: 서브클립 로컬 시간→원본 좌표 변환은 `frame_source.py:166` 단일 지점, 사후 assert로
  검증(frame_source.py:204-209). **이중 보정 위험 지점 없음**(코드 추적 기준).

### 4-5. frames_meta 스키마와 하류 소비 필드

`frames_meta.json`(artifacts.py:23-40):

| 필드 | 타입 | 의미 | 하류 소비 |
|---|---|---|---|
| `video_id` | str | 원본 파일명 | STEP4/5 키 |
| `fps` | float | 사용 fps(=0.5) | `FrameMeta.fps`(STEP5 `step5_runner.py:50`) |
| `duration` | float | 원본 길이(초) | 메타데이터 |
| `frame_paths[]` | list[str] | jpg 절대경로 | STEP4 VLM/YOLO 입력 |
| `times[]` | list[float] | 각 프레임 원본 시각(초) | 관찰 시각 부착·정렬 |
| `segments[]` | list? | motion일 때만: `{clip_no,filename,start_sec,end_sec,overlaps_next}` | 좌표검증/구간 후처리 |

- 실측(CLIP2, motion): `fps=0.5, duration=113.39, n_frames=42, times=[16.0,18.0,20.0,...],
  segments=3개`. `segments`는 uniform이면 키 자체가 없다(포맷 100% 하위호환, artifacts.py:28-29,36-37).
- STEP5 소비 확인: `step5_runner.py:50-51`이 `fmeta["fps"]`, `len(fmeta["frame_paths"])`로
  `FrameMeta` 구성. `times`/`frame_paths`는 STEP4가 VLM 관찰 시각 부착에 사용.

**근거 위치(§4):** `step3_registry.py:19-22,42-43`, `frame_source.py:31-36,46-56,119-217`,
`frame_extract.py:24-59`, `motion_sampling/{motion_score,select_clip,extract_video,abc_pipeline_runner}.py`,
`motion_sampling/__init__.py:1-11`, `tacit_common/artifacts.py:23-40`, `STEP5_.../step5_runner.py:50-51`,
실측 `output/_frames/CLIP2.mp4/frames_meta.json`.

---

## §5. 트러블슈팅 아카이브

> 각 사건: 증상 → 원인 → 실패한 시도 → 최종 해결 → 코드에 남은 방어 장치 → 재발 시 확인.
> 흔적 출처: git log, 코드 주석, config 방어적 기본값, try/except·empty_cache 등.

### 5-1. 한글 경로 + Python 디코더 3종(pyav/torchcodec/cv2) 불안정 → ffmpeg CLI 우회

- **증상**: pyav / torchcodec / cv2 3종 Python 비디오 디코더가 이 환경에서 전부 불안정
  (경로 인코딩 / libavutil / libnvrtc 관련).
- **원인(코드 주석 기준)**: 이 환경의 Python 디코더 라이브러리 문제
  (`frame_extract.py:1-7`: "torchcodec/pyav/cv2 디코딩이 불안정", "libnvrtc/libavutil/경로인코딩").
- **실패한 시도의 흔적**: **(확인 불가)** pyav/torchcodec/cv2를 실제로 호출했던 코드는 현재
  저장소에 남아 있지 않다. 이들 디코더의 시도·실패는 **주석 진술로만** 존재한다
  (frame_extract.py:1-7, vlm_qwen.py:8). "영어 경로로 변경"이라는 수정 시도도 **코드/커밋
  흔적 없음** — 현재 영상 파일명은 여전히 한글(`CLIP3_재부팅시도 전까지.mp4`)이고 정상 동작한다.
- **최종 해결(현재 코드 기준 확정)**: **ffmpeg CLI subprocess로 프레임을 jpg로 추출**하는 방식이
  최종 채택됐다(`frame_extract.py:35-59`, `audio_extract.py`도 동일 이유 subprocess).
  ffmpeg CLI가 경로 인코딩/코덱 문제를 통째로 우회하며 한글 경로도 그대로 처리한다.
  - **"네이티브 비디오 모드로 복귀"는 하지 않았다.** config의 `vlm.input_mode: "native_video"`는
    **명칭일 뿐**, 실제로 VLM도 ffmpeg로 추출한 jpg를 PIL 리스트로 받는다
    (`vlm_qwen.py:8` "프레임은 ffmpeg CLI로 추출... → PIL 리스트로 전달"). 즉 **최종은 jpg 우회
    방식**이며, 원본 mp4를 디코더에 직접 넣는 경로는 쓰지 않는다.
- **코드에 남은 방어 장치**:
  - ffmpeg 3단 폴백: `ffmpeg_bin(config) or $FFMPEG_BIN or DEFAULT_FFMPEG`(frame_extract.py:20-21).
  - preflight가 ffmpeg 바이너리 존재를 로드 전에 검증(step3_preflight.py:34-39).
  - `-v error`/`-loglevel error`로 조용한 실패 방지 + `check=True`(frame_extract.py:53,56).
- **재발 시 확인**:
  1. `config.frame_extraction.ffmpeg_bin`이 실재 바이너리인지. **`DEFAULT_FFMPEG =
     /home/piai/anaconda3/envs/deep/bin/ffmpeg`(frame_extract.py:17)는 이 서버에 없다** —
     반드시 config로 오버라이드해야 함(현재 config는 imageio_ffmpeg 정적 바이너리로 지정).
  2. Python 디코더(pyav/torchcodec/cv2)를 다시 도입하지 말 것 — 우회가 정답.

### 5-2. 8GB GPU에서 STT/VLM 동시 적재 시 OOM → 순차 실행 + 즉시 언로드

- **증상**: STT(cuda)가 언로드 안 된 채 VLM→LLM으로 이어지면 뒤 단계 OOM 여유 감소
  (특히 VLM 관찰이 많아 LLM 입력이 커진 경우). 2026-07-02 실측(stt_whisper.py:78-83 주석).
- **원인**: 단일 GPU에 STT/VLM/LLM을 동시에 얹으려는 것. 8GB(구 RTX2080)에서는 특히 치명.
- **최종 해결(순차 실행 설계)**:
  - **STT 완료 → 즉시 `unload()`**: `step3_runner.py:41-42`
    ```python
    if hasattr(self.stt, "unload"): self.stt.unload()
    ```
  - `WhisperTurboSTT.unload()`(stt_whisper.py:77-94): `self._model=None; gc.collect();
    torch.cuda.empty_cache()`. STT는 파이프라인 초반 1회만 쓰므로 끝나면 바로 비운다.
  - 같은 패턴이 STEP4(VLM unload), STEP5(LLM unload)에도 있어(`step5_runner.py:53-54`),
    한 GPU에 한 번에 한 모델만 상주하게 순차 실행된다.
- **코드에 남은 방어 장치**: `hasattr(x, "unload")` 덕타이핑 가드(어댑터가 unload 미구현이어도
  안 깨짐), `unload()` 안 `try/except`로 torch import/empty_cache 실패 무시(stt_whisper.py:88-94).
- **재발 시 확인**: unload 호출이 각 STEP 끝에 살아있는지, GPU가 실제로 비는지(`nvidia-smi`).
  현재는 서버 증설로 RTX6000 23.6GB×2라 여유가 크지만(README.md:183) 순차 설계는 유지.

### 5-3. 긴 영상에서 VLM fps로 인한 OOM (+ 관찰 0건 동결 버그)

> **STEP3와 STEP4의 fps는 다른 손잡이다.** STEP3 `frame_extraction.fps_*`는 ffmpeg 추출 밀도.
> 아래 OOM은 그 프레임을 소비하는 **STEP4 VLM 추론**에서 터졌고, 회피책으로 STEP3의 추출 fps를
> 낮췄다가 다른 버그를 만난 사건이다. config.yaml `frame_extraction.fps_long` 주석에 전말이 있다.

- **증상**: CLIP1(99프레임, fps=0.5)에서 CUDA OOM(21.00GiB 사용 중 1.64GiB 추가 요청 실패).
- **실패한 시도(fps 자동 하향)**: `fps_long`을 0.5→0.3으로 낮춰 60~70프레임대로 회피 시도.
  → **그런데 0.3에서 CLIP1/CLIP4가 vision 토큰은 정상(3600~4200개)인데 관찰 0건
  (`{"observations": []}`)으로 얼어붙는 별개 버그** 발생. fps=0.4에서도 재발
  (config.yaml `fps_override` 주석). → **"fps 낮추기는 버릴 카드"로 결론.**
- **최종 해결(근본, config.yaml `fps_long` 주석 + 커밋 8c83615)**:
  1. **YOLO를 서브프로세스로 격리**(`step4_components/yolo_subprocess_entry.py`) —
     ultralytics의 `select_device()`가 `CUDA_VISIBLE_DEVICES`를 무조건 덮어써 부모 프로세스를
     오염시키던 것을 차단.
  2. **VLM `device: "auto"`** 로 2-GPU(≈47GB) 확보(`--gres=gpu:2` 필요).
  3. **VLM 청크 분할 관찰**(`chunk_sec=40`): 영상을 40초 구간으로 잘라 구간마다 `generate()`.
     전체 1회 생성이 유발하던 (a)자기복제 루프 (b)등간격 타임스탬프 날조 (c)관찰 0건 동결
     (d)KV캐시 OOM을 전부 완화(커밋 8c83615 메시지).
  4. 이 조합으로 **`fps_long`을 다시 0.5로 복구** — fps를 낮추는 대신 메모리/구조를 고쳤다.
     `max_pixels`(192²)는 일부러 안 낮춤(해상도↓ = 작은 부품 오검출 위험).
- **fps 값의 출처**: `fps_short=fps_long=0.5`는 스펙값(검증용 0.25→0.5 원복,
  config.yaml `fps_short` 주석). 0.3/0.4는 폐기된 회피값(주석에 실패 기록으로 보존).
- **코드에 남은 방어 장치**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (vlm_qwen.py:120, OOM 파편화 완화), 청크 분할(chunk_sec), 2-GPU device_map,
  YOLO 서브프로세스 격리.
- **미해결(config.yaml `max_memory_gib`/`fps_override` 주석)**: **CLIP4(115프레임)** 만
  fps=0.5+현재 max_memory에서 OOM, 21/15로도 반대쪽 GPU OOM, fps=0.4로 낮추면 관찰 0건 재발.
  → **별도 처리 예정**(fps 문턱 재탐색 또는 max_pixels 하향).
- **재발 시 확인**: `--gres=gpu:2`로 잡았는지(gpu:1이면 auto라도 1장), YOLO 서브프로세스 격리가
  살아있는지, 관찰 0건이면 fps 낮추기로 대응하지 말 것(이 버그의 방아쇠).

### 5-4. 그 외 STEP3 범위 git log / 주석에 남은 흔적

| 사건 | 증상/원인 | 해결/방어 | 근거 |
|---|---|---|---|
| **프레임 폴더 덮어쓰기 버그** | 모든 클립이 한 폴더에 섞여 덮어써짐 | `output/_frames/<video_id>/` 폴더 단위 분리 + 매 실행 이전 jpg 삭제 | step3_runner.py:6, frame_extract.py:48-49 |
| **정규식 근거판단 폐기** | 한국어 구어체 노하우("~하면 됩니다" 등)를 정규식이 0건 잡음 | 근거판단을 융합 LLM에 통째 위임, 정제는 정규화+반복태깅만(결정적) | 커밋 26c9a22, transcript_refine.py:1-13 |
| **STT transcript 이중 저장(정제 전/후)** | STT가 transcribe() 안에서 정제 前 스냅샷 저장 → STEP5가 정제 전 버전 볼 위험 | runner가 refine() 후 `save_transcript`로 **덮어써** 항상 정제본만 파일에 남김 | step3_runner.py:40, artifacts.py:49-62 |
| **transcript_dir 상대경로 cwd 의존** | STT 어댑터의 `transcript_dir`이 상대경로라 cwd(STEP3/통합실행)에 따라 위치가 달라짐 | runner가 PROJECT_ROOT 기준 절대경로로 강제 덮어씀 | step3_runner.py:25-31 |
| **motion dedup 시간근접→구간중첩 전환** | CLIP4 홀/짝 그리드 어긋남으로 시간근접 dedup이 98/99 프레임 안 걸러짐 | 직전 clip `end_sec+window` 이하 프레임 버리는 구간중첩 기준으로 교체 | frame_source.py:170-189 |
| **ffmpeg_bin 경로 서버이관** | 옛 `/home/piai/anaconda3/...` 경로가 새 서버에 없음 | config를 imageio_ffmpeg 정적 바이너리로 지정(2026-07-02) | config.yaml `ffmpeg_bin` 주석 |
| **preflight: 무거운 import 지연** | (STEP6 사고 교훈) 최상단 import가 preflight 도달 전에 죽임 | 모델 import를 `_load()` 안으로 지연, config 선택 impl만 검사 | stt_whisper.py:19-20,60-69, step3_preflight.py:19-31, CLAUDE.md §1 |
| **motion auto_generate 도입** | 신규 영상에 CSV/서브클립이 없으면 실패 | `auto_generate=True`면 팀원 코드(run_abc_pipeline) 호출해 자동 생성. 기존 4클립은 False로 검증 보존 | frame_source.py:71-84,131-147 |

**근거 위치(§5):** `frame_extract.py:1-7,17,20-21,48-49`, `vlm_qwen.py:8,120`, `stt_whisper.py:77-94`,
`step3_runner.py:6,25-31,40-42`, `frame_source.py:71-84,131-147,170-189`, `config.yaml`(fps/max_memory/ffmpeg_bin 주석),
git 커밋 `8c83615`, `26c9a22`, `536d341`.

---

## §6. 재디벨롭 가이드

### 6-1. 처음부터 다시 돌리는 실행 커맨드

전제: GPU 노드(n5)에서, `파이프라인_통합실행/.venv` 사용. STEP3만 GPU가 필요한 부분은 STT(cuda).

```bash
# (A) STEP3 단독 — 클립 1개
cd project-ai/STEP3_전처리
srun -p RTX6000 -w n5 --gres=gpu:1 \
  ../파이프라인_통합실행/.venv/bin/python3 run_step3.py \
  --config ../파이프라인_통합실행/config.yaml \
  --video /home/ai_user/team_a2/members/안나경/master/master_videos/CLIP1_정상조립과정.mp4

# (B) STEP3~5 통합 — 클립 1개
cd project-ai/파이프라인_통합실행
srun -p RTX6000 -w n5 --gres=gpu:2 \
  .venv/bin/python3 run_one_clip.py --config config.yaml \
  --video /.../master_videos/CLIP1_정상조립과정.mp4

# (C) 전체 4클립(STEP3~5, 모델 1회 로드)
cd project-ai/파이프라인_통합실행
srun -p RTX6000 -w n5 --gres=gpu:2 .venv/bin/python3 run_all_clips.py

# (D) STEP3~7 원샷(오케스트레이터)
cd project-ai/파이프라인_통합실행
srun -p RTX6000 -w n5 --gres=gpu:2 .venv/bin/python3 main.py --run-tag myrun

# (오디오만, GPU 불필요, 로그인 노드 가능 — 본선과 무관한 별도 유틸)
cd project-ai/STEP3_전처리 && python3 run_audio_extract.py
```
> `--gres=gpu:2`는 VLM(STEP4)의 device=auto 때문 필수(§5-3). STEP3 단독은 gpu:1로 충분.
> `--pty` 인터랙티브 셸은 자동화에서 못 쓰므로 항상 `srun ... python3` 형태로(CLAUDE.md §2).

### 6-2. 새 영상(클립) 추가 시 체크리스트

1. **영상 배치**: `master/master_videos/`에 mp4를 둔다.
2. **`run_all_clips.py:26-31`의 `VIDEOS` 리스트에 절대경로 추가**(하드코딩이라 자동 스캔 안 됨).
   1개만 돌릴 땐 `run_one_clip.py --video`로 대체 가능.
3. **uniform 모드면 끝**(추가 산출물 불필요).
4. **motion 모드로 돌릴 거면**:
   - `sampling.impl: "motion"` + `sampling.params.motion_root`를 **실제 존재하는 경로**로 지정
     (기본값은 stale, §7). CSV/서브클립이 이미 있으면 `<motion_root>/<stem>/clip_timestamps.csv`
     형태여야 함.
   - 없으면 `sampling.params.auto_generate: true`로 자동 생성(팀원 코드 호출).
     단 자동 생성은 A/B/C 전략을 다시 뽑으므로 기존 검증이 무효화될 수 있음(frame_source.py:74-77).
5. **정규화 사전 갱신**: 새 영상에서 STT 오인식이 나오면 `resources/en_normalization.json`의
   `_candidates`에서 최상위로 승격(추측 활성화 금지 — 원문 훼손 위험, en_normalization.json:11).
6. **스모크 검증**: transcript utterances 개수 > 0, frames_meta frame_paths 개수 > 0, motion이면
   좌표 assert 통과(에러 없이 끝남만으론 불충분, CLAUDE.md §1-5).

### 6-3. 현재 설계의 알려진 한계 / 개선 후보

| 항목 | 내용 | 근거 |
|---|---|---|
| **`normalized_text`가 융합 입력에 거의 안 쓰임** | STEP5 융합은 `raw_text`만 LLM에 전송하고 `normalized_text`는 2026-07-06부터 **미전송**(raw와 거의 동일 판단). 유일 소비처는 STEP5 `cross_check`의 허용텍스트 목록(step5_runner.py:61-62)뿐. 즉 STEP3의 정규화 노력(마더보이드→motherboard)이 사실상 하류에 반영되지 않음 | STEP5 `llm_fusion.py:212-214` 주석, `step5_runner.py:61-62` |
| **motion_root 기본경로 stale** | config·`__init__.py` 기본값이 `00_사전연구/04_...`인데 실제는 `STEP1_모델연구/01_모션기반샘플링/02_...`로 이동. motion 모드 즉시 켜면 FileNotFoundError | §4-3, §7 |
| **fps 자동조정이 실질 무효** | `fps_short==fps_long==0.5`라 길이별 분기가 항상 같은 값. 긴 영상 최적화 여지 있으나 fps 낮추기는 관찰 0건 버그 방아쇠(§5-3)라 신중히 | config.py:35-36 |
| **CLIP4 VLM OOM 미해결** | 115프레임에서 fps=0.5 OOM, fps 낮추면 관찰 0건. 별도 과제 | config.yaml `max_memory_gib` 주석 |
| **transcript 수동편집 흔적** | `split_from_original` 필드·`_merged_all.json`은 코드가 안 만듦. 재실행 시 사라져 재현 불가 | §3-3, §7 |
| **audio WAV 산출물 소비처 없음** | `output/audio/*.wav`는 별도 유틸 산출물이고 본선이 안 씀 | §3-5 |
| **DEFAULT_FFMPEG 하드코딩 stale** | 폴백 경로 `/home/piai/...`가 이 서버에 없음. config 오버라이드 의존 | frame_extract.py:17 |

### 6-4. 건드리면 위험한 불변식 (깨지면 하류 전체가 무너짐)

1. **timestamp = 원본 영상 초(float) 단일 좌표계.** 오디오·비디오 두 축이 이걸 지켜야 STEP5
   aligner의 ±4초 매칭이 성립. 특히:
   - STT에 원본이 아닌 트리밍/서브클립 오디오를 넣지 말 것(좌표 어긋남, §3-1). 모션클립은
     관찰 전용, STT는 항상 원본(별도 메모리 제약).
   - motion 로컬→원본 보정은 `frame_source.py:166` 한 곳. 여기 외에 오프셋을 더하는 코드를
     추가하면 이중 보정. assert(frame_source.py:204-209)를 지우지 말 것.
2. **`video_id = 원본 파일명(확장자 포함)`.** STEP4/5가 이 문자열로 파일을 찾는다. 파일명
   정규화 규칙을 바꾸면 하류가 파일을 못 찾음(step3_runner.py:35).
3. **정제 후 저장이 최종.** runner가 refine 후 `save_transcript`로 덮어써야 STEP5가 정제본을
   본다. transcribe() 내부 1차 저장(정제 전)만 남기면 안 됨(step3_runner.py:40).
4. **STT 끝 즉시 unload.** 순차 실행 전제. 지우면 VLM/LLM 단계 OOM 여유 감소(§5-2).
5. **frames_meta 포맷 하위호환.** uniform은 `segments` 키 없음. 항상 넣도록 바꾸면 기존 uniform
   소비 코드와 어긋날 수 있음(artifacts.py:28-29).
6. **raw_text 절대 손실 금지.** 원문은 정규화·태깅 후에도 보존(intermediate.py:115,
   transcript_refine.py:95). 정제가 원문을 덮어쓰면 융합 근거(source_utterance)가 훼손됨.

**근거 위치(§6):** `run_step3.py`, `run_one_clip.py`, `run_all_clips.py:26-31`, `main.py`,
`config.py:35-36`, `frame_source.py:74-77,166,204-209`, `step3_runner.py:35,40`, `stt_whisper.py:77-94`,
`STEP5_.../llm_fusion.py:212-214`, `STEP5_.../step5_runner.py:61-62`, `en_normalization.json:11`, CLAUDE.md §1-2.

---

## §7. "확인 불가" 항목 목록 (코드로 검증 불가능)

1. **pyav/torchcodec/cv2 실패의 실제 코드 흔적** — 3종 디코더를 호출했던 코드는 저장소에 없다.
   실패·시도는 주석 진술로만 존재(frame_extract.py:1-7, vlm_qwen.py:8). 재현 불가.
2. **"영어 경로 변경" 수정 시도** — 코드/커밋에 한글→영어 파일명 변경 흔적 없음. 현재 영상은
   여전히 한글명이고 ffmpeg CLI로 정상 동작. "영어 경로 변경 후 정착 디코더"라는 전제 자체가
   코드상 확인되지 않음(정착 = ffmpeg jpg 우회, §5-1).
3. **디스크의 motion frames_meta를 생성한 정확한 config** — 현 config.yaml/config_motion_obs.yaml
   둘 다 `sampling.impl: uniform`인데 디스크 산출물은 `segments` 포함(motion). motion 실행에
   쓰인 정확한 config/motion_root 오버라이드는 저장소에 남아있지 않다(산출물 존재로 motion
   실행 사실만 역추정).
4. **`split_from_original` 필드 / `_merged_all.json`의 출처** — 생성 코드가 저장소에 없다
   (grep 0건). 수동 편집·임시 스크립트 추정. 재실행 시 사라짐.
5. **모션 샘플링 원 작성자 실명** — `motion_sampling/__init__.py:2`에 "확정 대기"로 남고
   채워지지 않음.
6. **VAD threshold 실험 수치("0.2로 낮춰도 무의미")의 재현** — STEP1 재검증 보고서 소관.
   STEP3 코드는 값을 전달만 하므로 이 수치의 검증은 STEP3 범위 밖.
7. **`fps=0.5`가 스펙값이라는 "스펙" 문서 원본** — 코드 주석이 "스펙 5.4/5.6"을 참조하나 그
   스펙 원문 자체는 이 보고서에서 대조하지 않음(주석 진술 기준).
8. **CLIP 원본 길이 불일치** — STEP1 VAD 보고서의 CLIP1 원본 길이(198.5s)와 STEP3 산출물의
   duration이 다를 수 있음(서로 다른 컷/소스 가능성). 어느 쪽이 "정본"인지 코드로 확정 불가.

---

*(끝) 본 보고서는 조사·설명 전용이며 코드를 수정하지 않았다.*
