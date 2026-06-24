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
| smoke test — xsbdRlpLYhc (세바시, 72.9분) | ✅ |
| **데이터 추가 수집 (4개 이상)** | 🔲 진행 중 |
| 전체 실험 · 통계 분석 · 시각화 | 🔲 |

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

## Smoke Test 결과 (n=1, 참고용)

파일: xsbdRlpLYhc (세바시 AI 강연, 72.9분, 대화체)

| 모델 | CER | WER | RTF |
|------|-----|-----|-----|
| large-v3 | **12.31%** | **20.23%** | 0.107 |
| large-v3-turbo | 12.68% | 20.61% | **0.045** |
| whisper-medium-ko | 15.13% | 28.30% | 0.066 |

> **주의**: 위 결과는 구버전 스키마로 기록됨 (S/D/I, cer_early, cs_wer 등 없음). 데이터 추가 수집 후 전체 실험 시 xsbdRlpLYhc도 포함해 재측정 필요.

---

## 데이터셋 조건

Wilcoxon 검정에 최소 **5개** 필요. 현재 1개 → **4개 이상 추가**.

### 필수 조건

| 항목 | 기준 |
|------|------|
| 자막 종류 | YouTube **수동 자막(cc)** — 자동 생성 절대 금지 |
| 자막 언어 | 한국어 |
| 자막 완전성 | 영상 내 **모든 발화**가 자막으로 처리됨 (일부만 달린 것 금지 → ins_rate 오염) |
| 자막 방식 | **축어** (요약·의역 금지 → del_rate 오염) |
| 비발화 주석 | `[웃음]` `(박수)` 등 없는 것 |
| 화자 수 | **1~2인**, 동시 발화 없음 |
| 길이 | **10~30분** |
| 음향 | 조용한 실내, 배경음악 없음 또는 매우 작음 |

### 권장 조건

- 파일 간 화자 중복 없음
- 뉴스 앵커·강의·오디오북·다큐 내레이션·단독 강연 등 선호
- 남녀 혼합

### 금지

- YouTube 자동 생성 자막 (`ko-ko`, `ko-orig` 등)
- 패널 토론·청중 Q&A (3인 이상, 동시 발화)
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
conda run -n base python -c "
import re, pathlib
vtt = pathlib.Path('/home/piai/project-ai/stt_comparison_research/data/raw/F02.ko.vtt').read_text(encoding='utf-8')
blocks = re.split(r'\n\n+', vtt.strip())
texts = []
for b in blocks:
    lines = b.strip().splitlines()
    if not any('-->' in l for l in lines): continue
    texts += [l for l in lines if '-->' not in l
              and not l.startswith(('WEBVTT','Kind:','Language:')) and l.strip()]
pathlib.Path('/home/piai/project-ai/stt_comparison_research/data/ground_truth/F02.txt').write_text(' '.join(texts), encoding='utf-8')
print('완료')
"
```

### Step 4 — metadata.csv에 행 추가

```
file_id,utterance_type,duration_s,wav_path,url
F02,-,1200.0,/home/piai/project-ai/stt_comparison_research/data/raw/F02.wav,https://youtu.be/XXXX
```

> `wav_path`는 절대 경로 필수.

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

```
[Stage 1 — 현재] STT 비교
  가장 CER 낮은 모델 선정 (예: large-v3-turbo)
               ↓ 선정 모델을 STT 백엔드로 고정
[Stage 2] VAD 연구 (vad_stt_research/)
  Phase 1: 단일 화자 롱폼 — Silero VAD 전처리 효과 정량화 (진행 중)
  Phase 2: 다중 화자 — PyAnnote Diarization → 화자별 구간 분리 → Stage 1 모델 전사
```

- `vad_stt_research/`와 코드 분리 운영. STT 연구 완료 후 최선 모델 확정 시 VAD 연구에 투입.
- PyAnnote는 VAD를 내부 포함 → Phase 2에서 Silero VAD 대신 PyAnnote가 VAD 역할까지 담당.
- xsbdRlpLYhc WAV는 `vad_stt_research/data/raw/`를 공유 참조.
