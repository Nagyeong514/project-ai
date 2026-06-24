# CLAUDE.md — STT 엔진 비교 연구

---

## 연구 목적

한국어 STT 3종(large-v3 / large-v3-turbo / whisper-medium-ko)을 동일 오디오·동일 지표로 비교해 폐쇄망 배포에 적합한 최선 모델을 정량적으로 선정한다.

---

## 진행 상태 (2026-06-24)

| 단계 | 상태 |
|------|------|
| 파이프라인 전체 구현 | ✅ |
| whisper-medium-ko CT2 변환 (`models/whisper-medium-ko/`) | ✅ |
| 평가 지표 구현 | ✅ |
| 데이터 수집 (5파일) | ✅ |
| **전체 실험 (5파일 × 3모델)** | ✅ **완료** |
| **최선 모델 선정** | ✅ **`faster_whisper_large_v3_turbo`** |
| 통계 분석 (Wilcoxon) · 시각화 | 🔲 미구현 (n=5, 검정력 낮아 선택적) |

---

## 비교 모델

| 키 | 모델 | 상태 |
|----|------|------|
| `faster_whisper_large_v3` | Whisper large-v3 | 실험 가능 |
| `faster_whisper_large_v3_turbo` | Whisper large-v3-turbo | 실험 가능 |
| `whisper_medium_ko` | whisper-medium-ko (KsponSpeech 파인튜닝) | 실험 가능 |
| kospeech / clova / kakao | — | checkpoint·API키 없음, 보류 |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| `cer` | 문자 오류율 — **주지표** |
| `wer` | 단어 오류율 — 보조 |
| `substitutions / deletions / insertions / hits` | 오류 유형 분해 |
| `ins_rate` | insertions / ref_words — 환각 경향 |
| `del_rate` | deletions / ref_words — 누락 경향 |
| `length_ratio` | hyp_words / ref_words — 1 초과 시 과다 생성 |
| `cer_early / cer_late` | 전반부·후반부 CER — 롱폼 안정성 |
| `cer_degradation` | cer_late − cer_early — 양수면 후반 악화 |
| `cs_wer` | 영어 토큰만 추출한 WER — 한영 혼용 처리 능력 |
| `cs_ref_tokens` | GT 내 영어 토큰 수 (3개 미만이면 cs_wer 무효) |
| `rtf_mean / rtf_std` | 실시간 처리 배율 — 낮을수록 빠름 |

---

## 실험 결과 (n=5, 2026-06-24 완료)

**선정 룰 적용 결과** — RTF ≤ 0.10 통과 후 CER 최소:

| file_id | utterance_type | 선정 모델 | CER | RTF |
|---------|---------------|-----------|-----|-----|
| xsbdRlpLYhc | 대화 (레거시) | turbo | 0.1155 | 0.0475 |
| F02 | 낭독 | turbo | 0.0437 | 0.0278 |
| F03 | 낭독 | turbo | 0.0750 | 0.0272 |
| F04 | 낭독 | large-v3 | 0.1127 | 0.0706 |
| F05 | 낭독 (롱폼) | turbo | 0.1254 | 0.0382 |

**turbo 4회 선정 / large-v3 1회 선정 → 최종: `faster_whisper_large_v3_turbo`**

주요 발견:
- large-v3: RTF 5파일 중 4파일 0.10 초과 탈락
- medium-ko: F04에서 CER 75.5% (del_rate 72.6%) 붕괴 — 배포 불가
- turbo: 전 파일 RTF 0.10 이하, CER 평균 10.8%로 압도적 우세

상세 보고서: `results/STT_EXPERIMENT_REPORT.md`

---

## 데이터셋 조건

Wilcoxon 검정에 최소 **5개** 필요. 현재 1개 → **4개 추가 목표**.

**수집 계획**:

| utterance_type | 길이 | 목표 | 비고 |
|----------------|------|------|------|
| 낭독 | 30분 이상 (롱폼) | 2개 | **최우선** — cer_degradation 측정 |
| 낭독 | 10~30분 (일반) | 1개 | 낭독 샘플 수 확보 |
| 대화 | 자유 | 1개 | 유형 균형 |

### utterance_type 라벨 (metadata.csv)

| 값 | 의미 | 예시 |
|----|------|------|
| `낭독` | 원고 기반, 준비된 발화 | 뉴스 앵커, 오디오북, 강의, 다큐 내레이션 |
| `대화` | 즉흥·구어체, 인터뷰·토크 | 세바시45, 팟캐스트 |

> 유형별 Wilcoxon 검정력은 파일 수가 적어 약하지만, 라벨은 탐색 분석(낭독 vs 대화 기술통계)을 위해 반드시 기록.
> 핵심 질문 중 하나: "whisper-medium-ko가 낭독체에서 역전하는가?"

### 파일 분류 기준

| 구분 | 조건 | 목표 수 |
|------|------|---------|
| **일반** | 10~30분, 단일 화자 주도 | 1개 이상 |
| **롱폼** | 30분 이상 | 2개 이상 — cer_degradation 의미 있게 측정하려면 필수 |

> xsbdRlpLYhc(72.9분, 대화, 2인)는 레거시 파일 — 1인 조건 도입 전 수집. 결과 해석 시 참고용으로 유지.
> cer_degradation은 롱폼 파일에서만 유의미하게 측정됨 (10~30분 파일은 후반 드리프트 거의 없음).

### 필수 조건

| 항목 | 기준 |
|------|------|
| 자막 종류 | YouTube **수동 자막(cc)** — 자동 생성 절대 금지 |
| 자막 언어 | 한국어 |
| 자막 완전성 | 영상 내 **모든 발화**가 자막으로 처리됨 (일부만 달린 것 금지 → ins_rate 오염) |
| 자막 방식 | **축어** (요약·의역 금지 → del_rate 오염) |
| 비발화 주석 | `[웃음]` `(박수)` 등 없는 것 |
| 화자 수 | **1인** — 평가 지표가 화자 수에서 추가 이득 없음, 크로스토크 원천 차단, Stage 2 Diarization과 경계 명확 |
| 음향 | 조용한 실내, 배경음악 없음 또는 매우 작음 |

### 권장 조건

- 파일 간 화자 중복 없음 — 특정 목소리에 결과가 묶이는 샘플 편향 방지 (화자별 CER 측정 목적 아님)
- 남녀 혼합 — 동일 이유
- 낭독 3개 확보 — medium-ko 낭독 역전 검증용

### 금지

- YouTube 자동 생성 자막 (`ko-ko`, `ko-orig` 등)
- 2인 이상 대화·인터뷰·패널·Q&A
- 사투리 전용 영상 (GT 품질 불안정)

---

## 실험 실행

### 환경 변수 (매 세션 필수)

```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
```

### Step 1 — 오디오 + 자막 다운로드

```bash
conda run -n base yt-dlp \
  --no-write-auto-subs --write-subs --sub-lang ko --sub-format vtt \
  --output "/home/piai/project-ai/stt_comparison_research/data/raw/F02.%(ext)s" \
  "https://youtu.be/XXXX"
```

### Step 2 — webm → WAV 변환 (ffmpeg 없음, PyAV 사용)

```bash
conda run -n base python -c "
import av, wave
container = av.open('/home/piai/project-ai/stt_comparison_research/data/raw/F02.webm')
resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
pcm = []
for frame in container.decode(audio=0):
    for f in resampler.resample(frame): pcm.append(bytes(f.planes[0]))
container.close()
raw = b''.join(pcm)
with wave.open('/home/piai/project-ai/stt_comparison_research/data/raw/F02.wav', 'wb') as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
    wf.writeframes(raw)
print(f'완료: {len(raw)/2/16000:.1f}s')
"
```

### Step 3 — VTT → Ground Truth TXT

```bash
conda run -n base python \
  /home/piai/project-ai/stt_comparison_research/scripts/prepare_ground_truth.py \
  --vtt /home/piai/project-ai/stt_comparison_research/data/raw/F02.ko.vtt \
  --file-id F02 \
  --gt-dir /home/piai/project-ai/stt_comparison_research/data/ground_truth
```

### Step 4 — metadata.csv에 행 추가

```
file_id,utterance_type,duration_s,wav_path,url
F02,낭독,1800.0,/home/piai/project-ai/stt_comparison_research/data/raw/F02.wav,https://youtu.be/XXXX
```

> `wav_path` 절대경로 필수. `utterance_type`은 `낭독` 또는 `대화`.

### Step 5 — 실험 실행

```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

conda run -n base env \
  PYTHONPATH=/home/piai/project-ai/stt_comparison_research \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  python /home/piai/project-ai/stt_comparison_research/scripts/run_experiment.py \
  --metadata /home/piai/project-ai/stt_comparison_research/data/metadata.csv \
  --config /home/piai/project-ai/stt_comparison_research/configs/experiment_config.yaml \
  --output /home/piai/project-ai/stt_comparison_research/results/raw/results.csv \
  --gt-dir /home/piai/project-ai/stt_comparison_research/data/ground_truth \
  --models faster_whisper_large_v3,faster_whisper_large_v3_turbo,whisper_medium_ko \
  --skip-api
```

> results.csv는 append 모드. 이미 실험한 파일은 metadata에서 제거 후 실행.

---

## 알려진 이슈

| 항목 | 내용 |
|------|------|
| ffmpeg 미설치 | Step 2의 PyAV 방법으로 대체 |
| conda run 경로 | cwd를 바꾸지 않음 — 모든 경로 절대경로 필수 |
| LD_LIBRARY_PATH | 매 세션마다 export 필요 (libcublas.so.12 문제) |
| whisper-medium-ko model_id | `configs/experiment_config.yaml` → `/home/piai/project-ai/stt_comparison_research/models/whisper-medium-ko` |
| results.csv 중복 | 같은 파일 두 번 실행하면 중복 행 생김 |

---

## 파일 구조

```
stt_comparison_research/
├── CLAUDE.md
├── configs/experiment_config.yaml         # 수정 금지
├── data/
│   ├── metadata.csv
│   ├── raw/                               # WAV, webm, vtt (git 제외)
│   └── ground_truth/{file_id}.txt
├── models/whisper-medium-ko/              # CT2 변환본 (git 제외)
├── pipeline/stt/
│   ├── base.py                            # STTSegment, STTResult, BaseSTT
│   ├── faster_whisper_runner.py
│   ├── kospeech_runner.py                 # 보류
│   └── api_runner.py                      # 보류
├── evaluation/
│   ├── normalizer.py
│   └── metrics.py                         # Metrics 데이터클래스, evaluate()
├── experiments/run_all_models.py
├── analysis/
│   ├── statistical_tests.py               # Wilcoxon
│   └── plot_generators.py                 # 데이터 수집 후 구현
├── results/raw/results.csv               # 누적 결과 (git 제외)
└── scripts/
    ├── run_experiment.py                  # 메인 진입점
    ├── prepare_ground_truth.py
    └── download_audio.py                  # ffmpeg 필요 — 미사용
```

---

## 전체 연구 로드맵

**선정 룰**: RTF ≤ 0.10을 통과한 모델 중 CER 최소. 동점 시 RTF 우선.

```
[Stage 1 — 완료] STT 비교
  선정 모델: faster_whisper_large_v3_turbo
               ↓ 선정 모델을 STT 백엔드로 고정
[Stage 2] VAD 연구 (vad_stt_research/)
  Phase 1: 단일 화자 롱폼 — Silero VAD 전처리 효과 정량화 (AI Hub 데이터 대기 중)
  Phase 2: 다중 화자 — PyAnnote Diarization → 화자별 구간 분리 → Stage 1 모델 전사
```

- `vad_stt_research/`와 코드 분리 운영. STT 연구 완료 후 최선 모델 확정 시 VAD 연구에 투입.
- PyAnnote는 VAD를 내부 포함 → Phase 2에서 Silero VAD 대신 PyAnnote가 VAD 역할까지 담당.
- xsbdRlpLYhc WAV는 `vad_stt_research/data/raw/`를 공유 참조.
