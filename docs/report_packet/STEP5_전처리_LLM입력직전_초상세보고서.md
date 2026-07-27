# STEP5 전처리 초상세 보고서 — 원재료 4파일 → LLM 최종 프롬프트 문자열까지

- 작성일: 2026-07-11
- 범위: **LLM에 입력이 들어가기 직전까지의 전처리만.** LLM 추론·검증(게이트)·재조립(`_rebuild_from_draft`)·Pass 2 응집은 범위 밖.
- 기준 실행: **CLIP4_재부팅및BIOS.mp4, 모션관찰 실전런(2026-07-10, 잡 2437)**
  - 실행 명령: `run_step5.py --config 파이프라인_통합실행/config_motion_obs.yaml --video-id "CLIP4_재부팅및BIOS.mp4"` (`run_logs/step5_motion_obs_r2.sbatch`, 로그 `step5_motion_obs_r2_2437.log`)
  - 본 보고서의 모든 수치는 **실제 산출물 파일을 로드해 코드 경로를 그대로 재실행**해 얻은 값이다. 재계산한 윈도우 9개는 저장본 `output/aligned_windows_motion_obs/CLIP4_재부팅및BIOS.mp4.windows.json`과 **바이트 단위 일치**를 확인했다.

## 추적 대상 (끝까지 따라가는 데이터 2건)

| 구분 | 데이터 | 원재료 상의 위치 |
|---|---|---|
| 행동 A | ts=81.0 "검지와 엄지를 사용하여 키보드의 Enter 키를 누름" | observations 파일 5번째 관찰 |
| 발화 U | 82.35~84.01초 "화면이 켜졌다고 다가 아니에요." | transcript 파일 2번째 발화 |

두 건은 시간상 0.65초 차이로 붙어 있어(행동→1.35초 뒤 발화 시작) 같은 윈도우에 묶이는 과정을 관찰하기 좋다.

---

# STAGE 0. 원재료 4개 파일

로드 주체: `step5_runner.py:39-43` (`Step5Runner.run`) — 네 파일을 `tacit_common/artifacts.py`의 로더로 읽는다. 경로는 전부 config가 결정한다(`config_motion_obs.yaml`의 `paths:` 섹션).

```python
transcript = artifacts.load_transcript(self.transcript_dir, video_id)   # step5_runner.py:39
flat_dets  = artifacts.load_detections(self.detections_dir, video_id)  # :40
actions    = artifacts.load_observations(self.observations_dir, video_id)  # :41
fmeta      = artifacts.load_frames_meta(self.frames_dir, video_id)     # :42
```

## 0-1. transcript — `STEP3_전처리/transcripts/CLIP4_재부팅및BIOS.mp4.json`

- **통계: 발화 7건**, repeat_hallucination=true **0건**, language=ko, model=`mobiuslabsgmbh/faster-whisper-large-v3-turbo`
- 필드 생성처(STEP3):
  - `start/end/raw_text`: faster-whisper STT 세그먼트 (`step3_components/stt_whisper.py`, STT 시점엔 `normalized_text=text` 임시복사 — `stt_whisper.py:118`)
  - `normalized_text`: 정제기 치환 (`step3_components/transcript_refine.py:78-96`, 사전 `resources/en_normalization.json` — 활성 규칙 3개: 마더보이드/마더보드/메인보드→motherboard)
  - `repeat_hallucination`: 직전 발화 동일 반복 or 환각 상투구 denylist 매칭 시 true (`transcript_refine.py:80-90`, denylist는 `:31-40` "다음 영상에서 만나요" 등)
- 실제 데이터(전체 7건 중 앞 2건 — 2번째가 추적 발화 U):

```json
{"start": 22.48, "end": 24.6, "raw_text": "됐습니다. 화면이 나왔습니다.",
 "normalized_text": "됐습니다. 화면이 나왔습니다.", "repeat_hallucination": false},
{"start": 82.35, "end": 84.01, "raw_text": "화면이 켜졌다고 다가 아니에요.",
 "normalized_text": "화면이 켜졌다고 다가 아니에요.", "repeat_hallucination": false}
```

- ⚠️ 정합성 주의: 7건 중 1건(130.74초)은 `raw_text`("채널 **5인식**")와 `normalized_text`("채널 **오인식**")가 다른데, 이 치환은 **현행 활성 사전 3규칙으로는 재현되지 않는다**. 파일 수정시각(2026-07-10 01:16)으로 보아 실전런 직전 수동(세션) 보정된 것이다. 이 보정본이 LLM에 전달되는지는 STAGE 2에서 다룬다(결론: **전달 안 됨**).

## 0-2. detections — `STEP4_YOLO_VLM관찰/output/detections/CLIP4_재부팅및BIOS.mp4.json`

- **통계: 검출 76건** (monitor 58, power_button_LED 13, hand 5), timestamp 0.0~189.0초
- 필드 생성처(STEP4 YOLO): `step4_components/detector_yolo.py:97-105` — 커스텀 가중치(dell7920_v1 best.pt, conf 0.25)로 프레임별 검출. `timestamp`는 frames_meta의 `times[frame_idx]`를 그대로 부착(모션 샘플링이라 등간격이 아님 — 실측 `times[28]=73.0`).
- 실제 데이터(앞 1건):

```json
{"timestamp": 0.0, "frame_idx": 0, "cls": "power_button_LED", "conf": 0.8555187582969666,
 "bbox": {"x": 169.92447662353516, "y": 265.1715393066406, "w": 28.728744506835938, "h": 30.09222412109375}}
```

## 0-3. observations — `STEP4_YOLO_VLM관찰/output_motion/vlm_observations_original_ts/CLIP4_재부팅및BIOS.mp4.observations.json`

- **통계: 관찰 17건** (actor: 시선 10, 오른손 7), `repeat_count>1` 접힘 3건: (ts 0.0, ×17, ~32.0), (75.0, ×2, ~77.0), (103.0, ×2, ~105.0)
- 필드 생성처(STEP4 VLM, 모션O 계열):
  - `action/actor/objects`: Qwen3-VL-8B 관찰 로그(`STEP4_YOLO_VLM관찰/step4_components/vlm_qwen.py`, 40초 청크 단위 생성)
  - `end_timestamp/repeat_count`: STEP4 dedup(4-B)이 동일 문장 반복 관찰을 접은 흔적
  - `chunk`: 그 관찰이 나온 청크 "전역t0-전역t1"
  - **이 파일 자체는 모션 하위클립별 산출물(clip01~05)을 원본 영상 시간축으로 재매핑·병합한 것**이다(예: clip04의 ts 7.0 → 122.0, clip05의 22.0 → 175.0). 재매핑 스크립트는 리포지토리에 없다(2026-07-10 세션 작업 산출물 — `scripts/merge_by_group.py`는 "가상 이어붙임" 시간축이라 별개).
- 실제 데이터(추적 행동 A):

```json
{"timestamp": 81.0, "actor": "오른손", "action": "검지와 엄지를 사용하여 키보드의 Enter 키를 누름",
 "objects": ["hand", "keyboard"], "end_timestamp": null, "repeat_count": 1, "chunk": "73-107"}
```

## 0-4. frames_meta — `STEP3_전처리/output/_frames/CLIP4_재부팅및BIOS.mp4/frames_meta.json`

- **통계: fps 0.5(샘플링 fps), duration 229.33초, frame_paths 84개, times 84개, segments 5개**(모션가이드 서브클립 구간표)
- 생성처(STEP3): 프레임 추출/모션 샘플링(`step3_components/frame_extract.py`·`motion_sampling/`), 저장은 `tacit_common/artifacts.py:23-40` `save_frames_meta`
- STEP5가 실제로 쓰는 값은 **fps, duration, frame_paths 개수 셋뿐**이다(`step5_runner.py:43,50-51`). `times`·`segments`·프레임 경로 자체는 STEP5 전처리에서 안 쓴다(duration은 LLM 이후 `_finalize_metadata`의 clip_end용 — 범위 밖).

## 0-5. 로드 시 Pydantic 검증에서 버려지거나 변형되는 것 (전수)

로더는 전부 `model_validate`를 거친다. Pydantic v2 기본값 `extra='ignore'`라 **스키마에 없는 키는 조용히 소멸**한다.

| # | 대상 | 무슨 일이 일어나나 | 근거 |
|---|---|---|---|
| 1 | observations 최상위 `n_observations` | 읽지 않음(리스트 길이로 대체 가능, 무손실) | `artifacts.py:117-121`은 `raw["observations"]`만 사용 |
| 2 | observations 최상위 `raw_by_chunk`(균일런 파일에 존재) | 읽지 않음 — VLM 원문은 STEP5에 안 들어감 | 동상 |
| 3 | 관찰 항목의 `source: "human_observation"` | **필드 자체가 `ActionDescription`에 없어 로드 즉시 소멸** (실측: `model_validate({... 'source':'human_observation'})` → dump에 source 없음) | `intermediate.py:130-147`에 source 필드 없음 |
| 4 | detections 최상위 `video_id` | 읽지 않음 | `artifacts.py:85-89` |
| 5 | transcript의 `language`/`model` | Transcript 객체엔 실리지만 STEP5 어디서도 사용 안 함 | `step5_runner.py` 전체 |
| 6 | 값 변형 | **없음** — 네 로더 모두 필드 값을 바꾸지 않는다(형변환 없음, 기본값 채움 없음 — CLIP4 파일들은 전 필드가 명시돼 있음) | |

⚠️ #3 은 실질 의미가 있다: 사람 추가 관찰(`STEP4_YOLO_VLM관찰/output_motion/vlm_observations/CLIP4_재부팅및BIOS_clip04·05...json`엔 `source:"human_observation"` 3건 실존)이 STEP5에 오는 경로에서 **두 번 지워진다** — (1) original_ts 병합본을 만들 때 이미 제거됐고, (2) 남아 있었더라도 Pydantic이 버린다. 그래서 시스템 프롬프트의 `[source="human_observation" 행동의 취급]` 규칙(`llm_fusion_prompt.py:151-155`)은 **현재 데이터 경로에서는 절대 발동하지 않는 죽은 규칙**이다. 같은 이유로 clip05의 `chunk:"out_of_coverage"` 관찰 1건("우분투 로그인 화면", 원본 ~224초)은 original_ts 병합본에 아예 없어 STEP5 입력에서 빠졌다.

---

# STAGE 1. 윈도우 묶기 — `step5_components/aligner.py` `WindowAligner.align`

파라미터(전부 `config_motion_obs.yaml` `aligner.params`): `window_sec=4.0`, `merge_overlapping=true`, `merge_cap_sec=20.0` (`aligner.py:32-40`에서 수신).

## 1-1. 추적 대상의 t0/t1 계산 (숫자 그대로)

**행동 A (ts=81.0)** — 행동은 앵커다. `aligner.py:57-58`:

```
t0 = max(0.0, 81.0 - 4.0) = 77.0
t1 = 81.0 + 4.0           = 85.0
→ 앵커 윈도우 [77.0, 85.0] 생성
```

**발화 U (82.35~84.01)** — 발화는 앵커가 아니라 **겹치는 모든 행동 윈도우에 복사**된다(`aligner.py:59-64`, `_overlaps`: `a0<=b1 and b0<=a1`). U는 앵커 4개와 겹친다:

| 앵커 행동 ts | 윈도우 | U(82.35~84.01)와 겹침? |
|---|---|---|
| 81.0 (추적 행동 A) | [77, 85] | ✅ (82.35 ≤ 85 and 77 ≤ 84.01) |
| 83.0 | [79, 87] | ✅ |
| 85.0 | [81, 89] | ✅ |
| 87.0 | [83, 91] | ✅ |

즉 병합 전 단계에서 이미 **발화 U는 서로 다른 윈도우 4곳에 중복으로 들어간다**(이 중복이 최종 프롬프트까지 살아남는지는 1-3 참조).

## 1-2. CLIP4 실측 집계: 17 → 20 → 9

- 행동 앵커 윈도우 **17개** (관찰 17건, 행동 1건당 1개 — `aligner.py:54-74`)
- 어느 앵커에도 안 걸린 leftover 발화 **3건** → utterance_only 윈도우 3개 추가 (`aligner.py:77-87`; 윈도우 구간은 발화의 [start, end] 그대로):
  - [22.48, 24.60] "됐습니다. 화면이 나왔습니다."
  - [127.38, 130.04] "여기서 용량이 맞게 잡혔는지 봐야 해요."
  - [130.74, 134.76] "화면이 켜졌어도 채널 5인식이면 성능이 절반으로 날아가거든요."
- 시간순 정렬 후 **20개** → `_merge`(`aligner.py:98-119`)에서 **병합 11회 발생 → 최종 9개**

병합 로그 전수(재실행 실측, `overlap = w.start <= last.end`, `capped = 병합 후 전체 span <= 20.0`):

```
[0,4]+[0,4]           → MERGE (span  4s)                        → W01
[22.48,24.60]         → SPLIT (겹침 없음)                        → W02
[69,77]+[71,79]       → MERGE (span 10s)
  +[77,85]            → MERGE (span 16s)   ← 추적 행동 A의 앵커가 여기서 흡수됨
  +[79,87]            → MERGE (span 18s)   ← utt dedup 발생: 2+2→2
  +[81,89]            → MERGE (span 20s)   ← utt dedup 발생: 2+2→2   → W03 [69,89]
[83,91]               → SPLIT (cap 초과: span 22s > 20)  ★20초 상한 분리 사례 1
  +[87,95]+[89,97]+[91,99]+[93,101] → MERGE ×4 (12→18s)          → W04 [83,101]
[97,105]              → SPLIT (cap 초과: span 22s > 20)  ★상한 분리 사례 2
  +[99,107]           → MERGE (span 10s)                          → W05 [97,107]
[111,119]+[118,126]   → SPLIT 후 MERGE (span 15s)                 → W06 [111,126]
[127.38,130.04]       → SPLIT                                     → W07
[130.74,134.76]       → SPLIT                                     → W08
[171,179]             → SPLIT                                     → W09
```

- **왜 W03/W04가 병합됐나**: BIOS 진입 구간(69~107초)의 키 입력 관찰이 2초 간격으로 촘촘해 ±4초 윈도우가 도미노처럼 겹쳤기 때문. cap=20이 없었으면 69~107초 전체가 윈도우 1개로 뭉쳤을 것(2026-07-06 5-A가 막은 사고 유형, `aligner.py:36-39` 주석).
- **dedup 실사례**: W03 체인에서 [79,87]과 [81,89]를 흡수할 때, 각각이 갖고 있던 발화 2건(82.35 / 84.67)이 이미 W03에 있어 `_extend_unique_utt`(`aligner.py:121-127`, `(start,end)` 키)가 **2회에 걸쳐 발화 4건 추가를 0건으로 걸렀다** (2+2→2가 두 번).
- **20초 상한 분리 실사례**: 위 로그의 ★ 2곳. [83,91]은 W03과 겹치지만(83 ≤ 89) 병합 시 span이 69→91 = 22초 > 20이라 분리됐고, 그 결과 **발화 U가 W03과 W04 양쪽에 들어갔다** (윈도우 간 dedup은 병합될 때만 일어나므로, 분리된 윈도우끼리는 같은 발화를 각자 보유).

## 1-3. 추적 대상의 최종 귀속

| 대상 | 최종 위치 |
|---|---|
| 행동 A (81.0 Enter 키) | **W03** [69, 89] (case=fusion, 행동 5·발화 2·검출 11) — 여기 한 곳뿐 |
| 발화 U (82.35) | **W03과 W04 두 곳** (W04 [83,101]: 행동 5·발화 2·검출 3) |

최종 9개 요약: W01 action_only / W02 utterance_only / W03 fusion / W04 fusion / W05 action_only / W06 action_only / W07 utterance_only / W08 utterance_only / W09 fusion.

## 1-4. windows.json 저장 시점과 실제 윈도우 1개 전문

저장 시점: **LLM 로딩보다 먼저**, 정렬 직후다 — `step5_runner.py:46-47` (`align` → `save_aligned_windows`). 저장 형식은 `{"video_id", "windows": [AlignedWindow.model_dump()...]}` (`artifacts.py:129-135`). 즉 windows.json은 "LLM이 보기 전" 상태의 완전 보존본이고, LLM 입력은 여기서 한 번 더 깎인다(STAGE 2).

`output/aligned_windows_motion_obs/CLIP4_재부팅및BIOS.mp4.windows.json`의 W02 **전문**(가장 짧은 윈도우 — leftover 발화가 윈도우가 된 모습과, detections까지 저장되는 것을 그대로 보여줌):

```json
{
  "window_start": 22.48,
  "window_end": 24.6,
  "actions": [],
  "utterances": [
    {"start": 22.48, "end": 24.6, "raw_text": "됐습니다. 화면이 나왔습니다.",
     "normalized_text": "됐습니다. 화면이 나왔습니다.", "repeat_hallucination": false}
  ],
  "detections": [
    {"timestamp": 24.0, "frame_idx": 12, "cls": "power_button_LED", "conf": 0.8116534352302551,
     "bbox": {"x": 173.37749481201172, "y": 333.85247802734375,
              "w": 23.597305297851562, "h": 25.5447998046875}}
  ]
}
```

---

# STAGE 2. 직렬화 — `llm_fusion.py:187-221` `QwenLLMFusion._serialize`

호출 지점: `fuse()` 서두(`llm_fusion.py:296`) — `payload = self._serialize(windows)`. 윈도우 리스트가 dict 리스트로 바뀌는 유일한 지점.

## 2-1. W03: windows.json 저장본 vs 직렬화 payload (나란히 diff)

**before — windows.json의 W03** (추적 행동 A·발화 U 항목만 발췌, 나머지 행동 3건·검출 11건은 동일 구조):

```json
{
  "window_start": 69.0,
  "window_end": 89.0,
  "actions": [
    {"timestamp": 75.0, "actor": "시선", "action": "모니터 화면이 꺼져 있다",
     "objects": ["monitor"], "end_timestamp": 77.0, "repeat_count": 2, "chunk": "73-107"},
    {"timestamp": 81.0, "actor": "오른손", "action": "검지와 엄지를 사용하여 키보드의 Enter 키를 누름",
     "objects": ["hand", "keyboard"], "end_timestamp": null, "repeat_count": 1, "chunk": "73-107"}
  ],
  "utterances": [
    {"start": 82.35, "end": 84.01, "raw_text": "화면이 켜졌다고 다가 아니에요.",
     "normalized_text": "화면이 켜졌다고 다가 아니에요.", "repeat_hallucination": false}
  ],
  "detections": [
    {"timestamp": 81.0, "frame_idx": 32, "cls": "monitor", "conf": 0.8425486087799072,
     "bbox": {"x": 16.09820556640625, "y": 161.36398315429688,
              "w": 253.26141357421875, "h": 159.36715698242188}},
    "…(총 11건, 이하 생략)"
  ]
}
```

**after — `_serialize`가 만든 payload의 W03** (같은 항목, 실측 원문):

```json
{
  "window_id": "W03",
  "case": "fusion",
  "window_start": "00:01:09",
  "window_end": "00:01:29",
  "actions": [
    {"timestamp": "00:01:15", "actor": "시선", "action": "모니터 화면이 꺼져 있다",
     "objects_visible": ["monitor"], "repeat_count": 2, "repeat_until": "00:01:17"},
    {"timestamp": "00:01:21", "actor": "오른손",
     "action": "검지와 엄지를 사용하여 키보드의 Enter 키를 누름",
     "objects_visible": ["hand", "keyboard"]}
  ],
  "utterances": [
    {"timestamp": "00:01:22", "raw_text": "화면이 켜졌다고 다가 아니에요.",
     "repeat_hallucination": false}
  ]
}
```

### diff 전수표

**빠진 필드 (전부):**

| 필드 | 왜 뺐나 |
|---|---|
| `detections` 배열 통째(W03 기준 11건) | `_serialize`에 detections 처리 코드 자체가 없다. 스키마 주석대로 "위치 힌트(코드레벨 판단용)"(`intermediate.py:161`) — LLM에겐 YOLO 클래스명이 관찰의 `objects_visible`로 간접 전달된다는 설계 |
| `utterances[].end` | 발화 시작 시각만 보냄 — STEP6 0단계 대조 기준이 시작 시각이라 종료 시각은 불필요 |
| `utterances[].normalized_text` | **2026-07-06부터 의도적 미전송** (`llm_fusion.py:212-214` 주석: raw와 거의 동일한 문장 2벌이 프롬프트만 키워 CLIP4 6,330토큰 prefill OOM의 일부였음. source_utterance는 어차피 raw_text 원문만 사용) |
| `actions[].chunk` | 디버깅용 청크 표식 — 코드가 안 담음 |
| `actions[].end_timestamp`·`repeat_count` (repeat_count==1일 때) | 조건부 생략(`llm_fusion.py:202-205`): 반복이 아니면 둘 다 기본값이라 정보량 0 — 프롬프트 다이어트 |

**바뀐 필드 (전부):**

| before | after | 변환 규칙 |
|---|---|---|
| `window_start: 69.0` | `"00:01:09"` | `seconds_to_hhmmss` — **초 단위 버림**(`intermediate.py:18-26`, `int()` 절사) |
| `window_end: 89.0` | `"00:01:29"` | 동일 |
| `actions[].timestamp: 81.0` | `"00:01:21"` | 동일 |
| `utterances[].start: 82.35` | `timestamp: "00:01:22"` | **키 이름도 start→timestamp로 변경** + 0.35초 절사 |
| `objects` | `objects_visible` | 키 이름 변경(`llm_fusion.py:199`) |
| `end_timestamp: 77.0` (repeat_count>1일 때만) | `repeat_until: "00:01:17"` | 키 이름 변경 + HH:MM:SS 변환(`llm_fusion.py:203-204`) |

**새로 생긴 필드 (전부):**

| 필드 | 부여 규칙 |
|---|---|
| `window_id: "W03"` | 윈도우 순번(1-base)을 `f"W{idx:02d}"`로 (`llm_fusion.py:183-185` `_window_id` — 직렬화와 LLM-후 재조립이 같은 규칙을 공유하는 게 결박의 핵심) |
| `case: "fusion"` | `AlignedWindow.case` property(`intermediate.py:163-174`): 행동+발화=fusion / 행동만=action_only / 발화만=utterance_only |

## 2-2. repeat 접힘이 실제로 전달된 데이터 (W01)

repeat_count 17짜리 응시 관찰(0.0~32.0초 지속)이 payload에서 이렇게 나간다:

```json
{"timestamp": "00:00:00", "actor": "시선",
 "action": "컴퓨터 케이스의 전면 패널에 붙은 스티커가 보이고, 그 아래에 작은 원형 로고가 있다",
 "objects_visible": ["sticker", "logo"], "repeat_count": 17, "repeat_until": "00:00:32"}
```

(시스템 프롬프트의 `[repeat_count · end_timestamp 해석]` 섹션이 "N회 반복이 아니라 약 N초간 지속"으로 읽으라고 지시 — `llm_fusion_prompt.py:143-149`.)

## 2-3. repeat_hallucination=true 발화의 취급

**전달된다(제외 아님).** `_serialize`는 발화를 거르지 않고 `repeat_hallucination` bool을 그대로 payload에 실어 보낸다(`llm_fusion.py:215`). 무시는 프롬프트 지시로 처리한다("repeat_hallucination=true 인 발화는 STT 끝부분 환각이니 **무시**하라" — `llm_fusion_prompt.py:79`). 다만 전처리 시점에 이미 두 갈래로 차별 취급이 준비된다:

- 탈선 검증 기준선 `input_utts`(발화 커버리지 검사용)에서 **제외**: `llm_fusion.py:299-300` — `if not u.repeat_hallucination`
- (범위 밖 참고: 재조립·접지검증에서도 제외 — `llm_fusion.py:529`, `:257`)

CLIP4는 true 발화가 0건이고, **현재 4클립 transcript 전체에도 true는 0건**이라 실데이터 예시는 없다(전 클립 grep 실측).

---

# STAGE 3. 메시지 조립 — `llm_fusion_prompt.py:187-208` `build_fusion_messages`

호출 지점: `llm_fusion.py:297` — `messages = build_fusion_messages(meta.video_id, payload)`. 반환은 `[{"role":"system",...},{"role":"user",...}]` 2건.

## 3-1. system 프롬프트 구조 (6,112자, 실측 3,403토큰)

`FUSION_SYSTEM_PROMPT`(`llm_fusion_prompt.py:47-184`, f-string 상수 — 모든 클립에 동일한 고정 문자열). 섹션 목차:

| # | 섹션 | 요약 |
|---|---|---|
| 1 | 역할 정의 (3줄) | "암묵지 후보를 서술하는 정제 도우미" — 판별은 다음 단계, 버리지 마라 |
| 2 | ★후보의 단위 (7줄) | 후보 1건=윈도우 1개, 모든 window_id 전수 귀속, 인접 2개만 병합 예외, 클립 요약 금지 |
| 3 | [너는 서술만 쓴다] (9줄) | LLM 몫: window_ids·situation·tacit_insight·reasoning·reasoning_origin·conflict·metadata. diagnostic_steps/timestamp/source_utterance 출력 금지(시스템이 채움) |
| 4 | 입력 구조 설명 (3줄) | 각 window에 행동 서술+발화(raw_text+태그), 행동·발화 비동시성 정상 |
| 5 | 발화 근거성 판단 (4줄) | 구어체 노하우 해석은 전적으로 LLM 몫, rh=true 발화 무시 |
| 6 | LED 진단 코드표 (11줄, `:35-45`) | Dell 7920 깜빡임 횟수→원인 해석표 + 발화 해석 우선 규칙. **LLM 프롬프트에만 있고 VLM엔 없음(역할 분리)** |
| 7 | [절대 규칙 — 할루시네이션 금지] (15줄) | 5개: 없는 사실 금지 / 묶기·분류 허용 / utterance 태깅 조건 / model_inferred 정직 태깅·위장 금지 / 본것·들은것 충돌 시 conflict=true 둘 다 보존 |
| 8 | [reasoning 접지] (9줄, 07-08 추가) | 발화가 있으면 발화 표현을 살려 utterance 태깅 우선 — model_inferred 도피 차단 |
| 9 | [발화 없는 window] (8줄) | 침묵 암묵지(응시·촉각 확인·순서 준수) 서술 요령 |
| 10 | [insight vs reasoning 구분] (5줄) | 두 필드 동일 내용 금지 |
| 11 | [출력 언어] (2줄) | 한국어 |
| 12 | [출력] + 견본 JSON (~20줄) | JSON만, placeholder 금지, 윈도우 1개짜리 견본 1건(내용 베끼기 금지) |
| 13 | ★metadata 역할 분담 (6줄) | LLM: task/keywords/scenario_title. 시스템: id/scenario_id/equipment/source — 아예 쓰지 마라 |
| 14 | [repeat_count 해석] (7줄) | 반복 횟수 아님 — "약 N초간 지속"으로만 |
| 15 | [source=human_observation] (5줄) | 사람 추가 관찰 취급 규칙 — **STAGE 0-5에서 본 대로 현 데이터 경로에선 죽은 규칙** |
| 16 | [지연 발화] (9줄) | 이 정비공의 "작업 먼저, 수십 초 뒤 설명" 습관 — 인접 병합 적극 검토 |

## 3-2. user 메시지 실제 앞부분 (원문 그대로)

```
video_id: CLIP4_재부팅및BIOS.mp4
아래는 시간순 정렬된 구간 9개다(각 구간: window_id, case, 시간, actions, utterances).
모든 window_id(W01~W09)를 정확히 한 후보에 귀속시켜, 규칙을 지켜 암묵지 후보 JSON을 생성하라.

[{"window_id":"W01","case":"action_only","window_start":"00:00:00","window_end":"00:00:04","actions":[{"timestamp":"00:00:00","actor":"오른손","action":"손가락으로 컴퓨터 케이스 앞 패널 상단의 파워 버튼을 누르는 듯한 움직임을 한다","objects_visible":["power_button_LED","computer_case"]},{"timestamp":"00:00:00","actor":"시선","action":"컴퓨터 케이스의 전면 패널에 붙은 스티커가 보이고, 그 아래에 작은 원형 로고가 있다","objects_visible":["sticker","logo"],"repeat_count":17,"repeat_until":"00:00:32"}],"utterances":[]},{"window_id":"W02",…
```

- 헤더 3줄은 `llm_fusion_prompt.py:196-200`이 조립(윈도우 수와 W01~W09 범위를 명시해 전수 귀속을 헤더에서도 못박음).
- 추적 발화 U는 이 user 문자열 안에 **2회 등장**한다(실측 offset 1,579자·2,511자 지점 — W03 소속 1회 + W04 소속 1회). 추적 행동 A는 1회(W03).

## 3-3. compact JSON 직렬화와 토큰 절감 (실측)

`llm_fusion_prompt.py:201-203`: `json.dumps(payload, ensure_ascii=False, separators=(',',':'))` — 2026-07-06에 `indent=2` → compact로 바꾼 지점(주석: "들여쓰기 공백만으로 프롬프트가 ~900토큰 커져 CLIP4가 prefill OOM 문턱을 넘었다").

CLIP4 이번 payload 실측(Qwen3-14B 토크나이저):

| | 문자 수 | 토큰 수 |
|---|---|---|
| payload indent=2 | 6,303자 | 2,280토큰 |
| payload compact (실제 전송) | 4,051자 | 1,606토큰 |
| **절감** | **2,252자** | **674토큰** |

(주석의 "~900토큰"은 균일런 CLIP4 — 윈도우 11개, 발화 19건 — 기준. 이번 모션런은 payload가 더 작아 절감폭도 비례해 작다.)

## 3-4. chat template 적용 — 모델에 들어가는 진짜 최종 문자열

적용 지점: `llm_fusion.py:419` (`_infer`) — `self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.

**enable_thinking=False 주입 방식**: `_load()`가 모델명에 "qwen3"이 들어 있으면 토크나이저의 `apply_chat_template`를 몽키패치해 **모든 호출에 `enable_thinking=False`를 setdefault로 끼워 넣는다**(`llm_fusion.py:115-123`, 로그 `[LLM-DEVICE] Qwen/Qwen3-14B: enable_thinking=False 주입`). 호출부(`_infer`)는 아무것도 모른 채 기본 동작이 바뀐다. Qwen2.5 등 다른 모델엔 no-op.

Qwen3-14B 템플릿 적용 결과(실측 원문):

```
<|im_start|>system
당신은 숙련 정비공(명장)의 작업 영상에서 '암묵지 후보'를 서술하는 정제 도우미다.
…(system 6,112자 전체)…<|im_end|>
<|im_start|>user
video_id: CLIP4_재부팅및BIOS.mp4
…(user 4,214자 전체)…<|im_end|>
<|im_start|>assistant
<think>

</think>

```

- 끝부분이 핵심이다: `add_generation_prompt=True`가 `<|im_start|>assistant\n`을 열고, **enable_thinking=False가 빈 `<think>\n\n</think>\n\n` 블록을 미리 채워** 모델이 thinking을 건너뛰고 바로 JSON을 생성하게 만든다. (미지정 시 기본 True — 끝이 `<|im_start|>assistant\n`에서 멈춰 모델이 `<think>`부터 생성함. 실측 diff로 확인.)
- 이 문자열이 `self._tok(text, return_tensors="pt")`로 토크나이즈되어 GPU로 가는 것(`llm_fusion.py:425`)까지가 전처리의 끝이다.

---

# STAGE 4. 숫자로 보는 요약 (CLIP4, 모션관찰 실전런, 전부 실측)

| 단계 | 값 |
|---|---|
| 원재료 | 발화 **7건** / VLM 관찰 **17건** / YOLO 검출 **76건** / frames_meta(84프레임, 229.33초) |
| 행동 앵커 윈도우 | 17개 |
| + leftover 발화 윈도우 | 3개 → 병합 전 20개 |
| 병합 | 11회 (dedup 2회, 20초 상한 분리 2회) |
| **최종 윈도우** | **9개** (fusion 3 / action_only 3 / utterance_only 3) → windows.json 19,140바이트 |
| payload (compact) | **4,051자 / 1,606토큰** (detections 76건 전량 미포함) |
| user 메시지 | 4,214자 / 1,696토큰 |
| system 프롬프트 | 6,112자 / 3,403토큰 (고정) |
| **최종 프롬프트(chat template 후)** | **10,425자 / 5,116토큰** |

(참고: 이 5,116토큰 prefill에 이어지는 생성 예산은 max_new_tokens=4096 — 범위 밖.)

---

# 부록 A. 정보가 손실되는 지점 전수 목록

| # | 손실 지점 | 발생 스테이지 | 괜찮은 이유 / 리스크 |
|---|---|---|---|
| 1 | **YOLO detections 76건 전량 미전송** (cls/conf/bbox/frame_idx) | STAGE 2 (`_serialize`에 코드 없음) | 설계 의도: 위치 힌트는 코드레벨용(`intermediate.py:161`), 객체명은 VLM 관찰의 `objects_visible`로 간접 전달. 리스크: LLM이 "화면에 monitor가 실제로 검출됐는가" 같은 교차 확인 불가 — conflict 판단 재료가 관찰문에만 의존 |
| 2 | **normalized_text 미전송** | STAGE 2 (`llm_fusion.py:212-214`) | OOM 실측에 근거한 의도적 다이어트. **CLIP4 실측 리스크 1건 실존**: 130.74초 발화의 보정본("채널 **오인식**")은 버려지고 LLM은 raw("채널 **5인식**")만 봤다 — STT 오인식 교정이 융합에 반영 안 되는 실사례 |
| 3 | **sub-second 절사** (82.35→"00:01:22") | STAGE 2 (`seconds_to_hhmmss` 버림, `intermediate.py:18-26`) | 1초 미만 정밀도는 진단 서사에 무의미. 참고: LLM-후 재조립은 `round()`를 써서(STEP6 대조 일치용) 0.5초대에서 payload 표기와 1초 어긋날 수 있음(`llm_fusion.py:546-549` 주석) — 전처리 산출물과 최종 문서의 시각이 다를 수 있는 근원 |
| 4 | `utterances[].end` 미전송 | STAGE 2 | 발화 길이 정보 소실 — STEP6 0단계는 시작 시각만 대조하므로 무해 |
| 5 | `actions[].chunk` 미전송 | STAGE 2 | 디버그 메타 — 무해 |
| 6 | `end_timestamp/repeat_count` 생략(repeat_count==1) | STAGE 2 (`llm_fusion.py:202-205`) | 기본값 생략 — 정보량 0, 무손실 |
| 7 | **`source:"human_observation"` 소멸** | STAGE 0 (병합본에서 제거 + Pydantic extra ignore) | **리스크 실존**: 프롬프트의 human_observation 취급 규칙(`llm_fusion_prompt.py:151-155`)이 발동 불가한 죽은 규칙이 됨. 사람 관찰과 VLM 관찰의 신뢰도 구분이 LLM에게 안 보임. 살리려면 `ActionDescription`에 `source` 필드 추가 + `_serialize` 전달 필요 |
| 8 | **out_of_coverage 관찰 1건 미포함** ("우분투 로그인 화면", clip05) | STAGE 0 이전 (original_ts 병합본 생성 시) | 원본 229.33초 밖(~224초는 범위 내인데도 빠짐) — 재부팅 완료 확인이라는 마무리 관찰이 후보 재료에서 소실. 병합 스크립트가 리포지토리에 없어 규칙 확인 불가한 것 자체가 재현성 리스크 |
| 9 | transcript `language`/`model`, 최상위 `video_id`류 | STAGE 2~3 | video_id는 user 헤더로 별도 전달, 나머지는 무해 |
| 10 | frames_meta의 `times`/`segments`/프레임 경로 | STAGE 0 (runner가 안 씀) | fps/duration/개수만 필요 — 무해(모션 구간표는 STEP4까지의 관심사) |
| 11 | rh=true 발화의 예정된 무시 | STAGE 2~3 (전달은 되나 무시 지시 + 커버리지 기준선 제외) | 환각 억제 설계. 현 데이터엔 true 0건이라 실손실 0 |
| 12 | (손실의 반대) **윈도우 겹침 중복 전달** — 발화 U가 프롬프트에 2회 | STAGE 1→3 | 손실이 아니라 **중복 왜곡**: W03·W04 두 후보가 같은 발화를 각자 근거로 서술할 수 있음. LLM-후 재조립의 dedup(`seen_utt`)은 한 후보 안에서만 작동 — 후보 간 중복은 STEP6 몫 |

# 부록 B. 전처리 단계에서 값이 결정되는 파라미터 전수

| 파라미터 | 값 (이번 실행) | 결정 위치 |
|---|---|---|
| `paths.step3_transcript_dir` | `STEP3_전처리/transcripts` | `config_motion_obs.yaml` `paths:` |
| `paths.step3_frames_dir` | `STEP3_전처리/output/_frames` | 〃 |
| `paths.step4_detections_dir` | `STEP4_YOLO_VLM관찰/output/detections` | 〃 (모션런도 YOLO는 균일런 것 재사용) |
| `paths.step4_observations_dir` | `STEP4_YOLO_VLM관찰/output_motion/vlm_observations_original_ts` | 〃 (**모션런과 균일런의 유일한 입력 차이**) |
| `paths.step5_aligned_windows_dir` | `…/output/aligned_windows_motion_obs` | 〃 |
| `aligner.params.window_sec` | 4.0 | `config_motion_obs.yaml` `aligner.params` → `aligner.py:32` |
| `aligner.params.merge_overlapping` | true | 〃 → `aligner.py:32` |
| `aligner.params.merge_cap_sec` | 20.0 | 〃 → `aligner.py:33-40` |
| window_id 형식 | `W{idx:02d}` (1-base) | 코드 고정 `llm_fusion.py:183-185` |
| payload JSON 형식 | compact `separators=(',',':')`, `ensure_ascii=False` | 코드 고정 `llm_fusion_prompt.py:203` |
| system 프롬프트 전문·LED 코드표 | 고정 상수 | `llm_fusion_prompt.py:35-184` |
| `llm.params.model_name` | `Qwen/Qwen3-14B` | `config_motion_obs.yaml` `llm.params` — **전처리엔 토크나이저/chat template 선택으로만 작용** |
| enable_thinking | False (qwen3 계열 자동) | 코드 고정 `llm_fusion.py:115-123` (config 노출 안 됨) |
| `min_utterance_coverage` | 0.5 | config에 없음 → 코드 기본값 `llm_fusion.py:48` (검증용이지만 그 기준선 `input_utts`는 전처리 시점에 계산 — `llm_fusion.py:299-300`) |
| (참고, 생성 직전 경계) `max_new_tokens`/`temperature`/`max_retries`/`quantization`/`max_memory_gib` | 4096 / 0.2 / 5 / nf4 / {0:20,1:20} | `config_motion_obs.yaml` `llm.params` — 프롬프트 문자열 자체에는 무영향(추론 단계 소관) |

# 부록 C. 재현 방법

```bash
# 로그인 노드에서 GPU 없이 재현 가능(aligner·직렬화는 순수 로직, 토크나이저는 CPU)
cd /home/ai_user/team_a2/members/안나경/project-ai
python3 docs/report_packet/scripts_trace/trace_stage1.py                        # STAGE 1: 병합 로그 재현 + 저장본 바이트 대조
파이프라인_통합실행/.venv/bin/python docs/report_packet/scripts_trace/trace_stage23.py  # STAGE 2·3: 직렬화·메시지·chat template·토큰 실측(venv에 transformers 필요)
```

저장본 일치 검증: `재계산 windows.model_dump() == windows.json["windows"]` → **True**.
