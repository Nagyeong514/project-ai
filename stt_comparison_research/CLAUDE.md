# CLAUDE.md — STT 엔진 비교 연구

새 세션을 시작하면 이 파일만 읽으면 됩니다. 현재 상태, 실행 방법, 다음 할 일이 모두 여기 있습니다.

---

## 한 줄 요약

한국어 STT 3종(large-v3 / large-v3-turbo / whisper-medium-ko)을 동일 오디오·동일 지표로 비교해 **폐쇄망 배포에 적합한 최선 모델**을 정량적으로 선정하는 연구.

---

## 비교 대상 모델

| 키 | 모델 | 상태 |
|----|------|------|
| `faster_whisper_large_v3` | Whisper large-v3 | 실험 가능 (자동 다운로드) |
| `faster_whisper_large_v3_turbo` | Whisper large-v3-turbo | 실험 가능 (자동 다운로드) |
| `whisper_medium_ko` | seastar105/whisper-medium-ko-zeroth | 실험 가능 ✅ `models/whisper-medium-ko/` 에 CT2 변환 완료 |
| `kospeech` | KsponSpeech LAS checkpoint | checkpoint 없음 — 보류 |
| `clova` | CLOVA Speech API | API 키 없음 — 보류 |
| `kakao` | Kakao Speech API | API 키 없음 — 보류 |

> 현재는 faster-whisper 3종만 실험. `--skip-api` 플래그 필수.

---

## 현재 진행 상태 (2026-06-24 기준)

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | 파이프라인 구현 | ✅ 완료 |
| 2 | whisper-medium-ko CT2 변환 | ✅ 완료 (`models/whisper-medium-ko/`) |
| 3 | B 타입 smoke test (xsbdRlpLYhc, 72.9분) | ✅ 완료 — `results/raw/results.csv` |
| 4 | **A 타입 데이터 수집** | 🔲 진행 중 (데이터 구하는 중) |
| 5 | 전체 데이터(A×5 + B×5) 실험 실행 | 🔲 대기 |
| 6 | 통계 분석 (Wilcoxon) | 🔲 대기 |
| 7 | 시각화 (plot_generators.py) | 🔲 대기 |

---

## 현재까지 나온 결과 (B 타입 1개, 참고용)

파일: xsbdRlpLYhc (세바시 AI 강연, 72.9분, 대화·인터뷰)

| 모델 | CER | WER | RTF |
|------|-----|-----|-----|
| large-v3 | **12.31%** | **20.23%** | 0.107 |
| large-v3-turbo | 12.68% | 20.61% | **0.045** |
| whisper-medium-ko | 15.13% | 28.30% | 0.066 |

**해석 주의**: n=1, B 타입 단독 결과라 통계 검정 불가. 방향성 참고만.
- large-v3 vs turbo: CER 0.37%p 차이 — 사실상 동등, turbo가 2.4배 빠름
- whisper-medium-ko: KsponSpeech(낭독체) 파인튜닝 모델인데 대화체에서 불리함. **A 타입에서 역전 가능성 있음.**

---

## 데이터 구성 목표

### 왜 더 필요한가?
- Wilcoxon 검정에 최소 **파일 5쌍** 필요 (현재 1개)
- A 타입(낭독·강연)이 없으면 medium-ko 공정 평가 불가

### 목표 데이터셋

| 타입 | 의미 | 목표 수 | 현재 |
|------|------|---------|------|
| A | 낭독·강연 (또박또박 읽는 스타일) | **5개** | 0개 |
| B | 대화·인터뷰 (자연스러운 대화체) | **5개** | 1개 |

---

## A 타입 데이터 선정 체크리스트

YouTube 링크로 넣을 수 있음. 아래 기준 모두 충족해야 함.

### 필수
- [ ] **수동 자막(cc) 있음** — YouTube에서 자막 아이콘에 "cc" 표시 확인. 자동 생성 자막은 절대 안 됨
- [ ] **한국어 자막**
- [ ] **발화 스타일이 낭독·강연체** — 뉴스 앵커, 오디오북, 교육 강의, 다큐 내레이션 등
- [ ] **단독 화자** (여러 명이 대화하는 형식 X)
- [ ] **길이 10~30분** (5분 미만은 RTF 노이즈, 72분 이상은 반복 실험 부담)

### 음향
- [ ] 배경 음악 없거나 매우 작음
- [ ] 스튜디오 또는 조용한 실내 녹음
- [ ] 전화 음질(8kHz) 아님

### 다양성 (5개 채울 때)
- [ ] 남성 2~3개 + 여성 2~3개
- [ ] 도메인 섞기: 뉴스 / 교육 / 오디오북 / 다큐 / TED·세바시(강연만)
- [ ] 파일 간 화자 중복 없음

### 피할 것
- 자막이 수초 이상 밀리는 영상
- 중간광고로 내용이 잘리는 영상
- 박수·웃음 등 비발화 소음이 길게 이어지는 영상

---

## 실험 실행 방법

### 환경 변수 (매 세션마다 필수)
```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
```

### Step 1 — 오디오 + GT 준비 (YouTube 링크 1개당)
```bash
cd /home/piai/project-ai/stt_comparison_research

# 오디오 다운로드 → data/raw/{file_id}.wav
PYTHONPATH=$(pwd) python scripts/download_audio.py \
  --url "https://youtu.be/XXXX" --file-id A01

# 수동 자막 → data/ground_truth/{file_id}.txt
PYTHONPATH=$(pwd) python scripts/prepare_ground_truth.py \
  --url "https://youtu.be/XXXX" --file-id A01
```

> **주의**: `download_audio.py`가 ffmpeg 없어서 실패하면 PyAV로 직접 변환해야 함.
> 아래 방법 사용:
> ```bash
> conda run -n base python -c "
> import av, wave
> container = av.open('input.webm')
> resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
> pcm = []
> for frame in container.decode(audio=0):
>     for f in resampler.resample(frame): pcm.append(bytes(f.planes[0]))
> container.close()
> raw = b''.join(pcm)
> with wave.open('data/raw/A01.wav', 'wb') as wf:
>     wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
>     wf.writeframes(raw)
> "
> ```

### Step 2 — metadata.csv에 파일 추가

`data/metadata.csv` 에 행 추가:
```
file_id,utterance_type,duration_s,wav_path,url
A01,A,900.0,/home/piai/project-ai/stt_comparison_research/data/raw/A01.wav,https://youtu.be/XXXX
```

> `wav_path`는 **절대 경로** 사용. 상대 경로 쓰면 실험 스크립트가 못 찾음.

### Step 3 — 실험 실행

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

> `--models`와 `--gt-dir`과 경로들은 **모두 절대 경로**로 써야 함. 상대 경로 쓰면 깨짐.
> results.csv는 append 모드 — 이미 실험한 파일은 중복 추가되니 새 파일만 metadata에 넣고 실행할 것.

---

## 알려진 이슈

| 항목 | 내용 |
|------|------|
| ffmpeg 없음 | 시스템에 ffmpeg 미설치. WAV 변환은 PyAV로 대체 (위 방법 참고) |
| conda run + 상대경로 | conda run은 cwd를 바꾸지 않음 → 모든 경로 절대경로 필수 |
| LD_LIBRARY_PATH | CTranslate2가 libcublas.so.12 요구, 시스템엔 .so.13만 있음 → 매 세션마다 export 필요 |
| whisper-medium-ko 경로 | configs/experiment_config.yaml의 model_id가 로컬 절대경로로 설정됨: `/home/piai/project-ai/stt_comparison_research/models/whisper-medium-ko` |
| results.csv append | 실험 스크립트가 append 모드로 저장 → 같은 파일 두 번 돌리면 중복 행 생김 |

---

## 파일 구조

```
stt_comparison_research/
├── CLAUDE.md                              # 이 파일
├── configs/experiment_config.yaml         # 모든 파라미터 고정 (수정 금지)
├── data/
│   ├── metadata.csv                       # 실험 파일 목록 (file_id, type, duration, wav_path, url)
│   ├── raw/                               # WAV 파일 (git 제외, 절대경로로 참조)
│   └── ground_truth/{file_id}.txt         # 수동 자막 원문 텍스트
├── models/
│   └── whisper-medium-ko/                 # CT2 변환 완료 (734MB, git 제외)
│       ├── config.json
│       ├── model.bin
│       └── vocabulary.json
├── pipeline/stt/
│   ├── base.py                            # STTResult, BaseSTT ABC
│   ├── faster_whisper_runner.py           # large-v3 / turbo / medium-ko 공용
│   ├── kospeech_runner.py                 # 보류
│   └── api_runner.py                      # CLOVA / Kakao — 보류
├── evaluation/
│   ├── normalizer.py                      # 정규화 규칙 (공정 비교 핵심)
│   └── metrics.py                         # CER, WER
├── experiments/run_all_models.py          # 3종 RTF 측정 + 평가
├── analysis/
│   ├── statistical_tests.py              # Wilcoxon + gap 테이블
│   └── plot_generators.py                # 시각화 (데이터 모인 후 구현)
├── results/raw/results.csv               # 실험 결과 누적 (git 제외)
└── scripts/
    ├── download_audio.py                 # yt-dlp 오디오 다운로드
    ├── prepare_ground_truth.py           # 수동 자막 → GT txt
    └── run_experiment.py                 # 메인 실행 진입점
```

---

## VAD 연구와의 관계

`vad_stt_research/`와 별도 운영. 두 연구는 변수를 섞지 않음.
- STT 비교 연구에서 선정된 최선 모델이 추후 VAD 연구의 고정 백엔드로 투입될 예정.
- xsbdRlpLYhc WAV 파일과 ground truth는 두 연구가 공유 (VAD 연구 raw/ 경로 참조).
