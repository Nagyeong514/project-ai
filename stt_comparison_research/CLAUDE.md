# CLAUDE.md — STT 엔진 비교 연구

새 세션을 시작하면 이 파일만 읽으면 됩니다.

---

## 한 줄 요약

한국어 STT 3종(large-v3 / large-v3-turbo / whisper-medium-ko)을 동일 오디오·동일 지표로 비교해 폐쇄망 배포에 적합한 최선 모델을 정량적으로 선정하는 연구.

---

## 현재 진행 상태 (2026-06-24 기준)

| 단계 | 내용 | 상태 |
|------|------|------|
| 파이프라인 구현 | 전체 코드 완성 | ✅ |
| whisper-medium-ko CT2 변환 | `models/whisper-medium-ko/` 로컬 저장 완료 | ✅ |
| 평가 지표 확장 | CER/WER + S/D/I + ins_rate/del_rate/length_ratio | ✅ |
| 데이터 1개 smoke test | xsbdRlpLYhc (세바시, 72.9분) — `results/raw/results.csv` | ✅ |
| **데이터 추가 수집** | 수동 자막 있는 한국어 영상 5개 이상 필요 | 🔲 |
| 전체 실험 실행 | 데이터 모인 후 | 🔲 |
| 통계 분석 + 시각화 | 데이터 모인 후 | 🔲 |

---

## 비교 모델 (현재 실험 가능한 3종)

| 키 | 모델 | 비고 |
|----|------|------|
| `faster_whisper_large_v3` | Whisper large-v3 | 자동 다운로드 |
| `faster_whisper_large_v3_turbo` | Whisper large-v3-turbo | 자동 다운로드 |
| `whisper_medium_ko` | whisper-medium-ko (파인튜닝) | `models/whisper-medium-ko/` CT2 변환 완료 |

> kospeech / CLOVA / Kakao 는 checkpoint·API키 없어 보류 중. `--skip-api` 필수.

---

## Smoke Test 결과 (참고용 — n=1, 통계 검정 불가)

| 모델 | CER | WER | RTF |
|------|-----|-----|-----|
| large-v3 | **12.31%** | **20.23%** | 0.107 |
| large-v3-turbo | 12.68% | 20.61% | **0.045** |
| whisper-medium-ko | 15.13% | 28.30% | 0.066 |

large-v3 vs turbo: CER 차이 0.37%p — 사실상 동등, turbo가 2.4배 빠름.
whisper-medium-ko: 대화체에서 불리. 낭독·강연 스타일 데이터 추가 시 재평가 필요.

---

## 다음 할 일 — 데이터 수집

### 몇 개?

Wilcoxon 검정에 최소 **5개** 필요. 현재 1개 보유 → 4개 이상 추가.

### 어떤 영상?

발화 스타일 구분 없이 아래 조건만 맞으면 됨.

**필수**
- 수동 자막(cc) 있음 — YouTube 자동 생성 자막 절대 금지
- 한국어 단일 언어
- 단독 또는 2인 이하 화자 (동시 발화 없는 것)
- 길이 10~30분
- 자막이 발화 전체를 커버 (일부만 자막 달린 것 금지 → ins_rate 오염)
- 축어 자막 (요약·의역 금지 → del_rate 오염)
- `[웃음]` `(박수)` 등 비발화 주석 없는 것

**권장**
- 조용한 실내 녹음, 배경음악 없음
- 파일 간 화자 중복 없음

YouTube 링크 가져오면 자막 샘플 확인 후 바로 다운로드·변환·실험 진행.

---

## 실험 실행 방법

### 환경 변수 (매 세션 필수)

```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
```

### Step 1 — 오디오 다운로드 (YouTube)

yt-dlp + PyAV 조합 사용. ffmpeg 미설치로 `download_audio.py` 직접 사용 불가.

```bash
# 1-a. yt-dlp로 webm 다운로드
conda run -n base yt-dlp \
  --no-write-auto-subs --write-subs --sub-lang ko --sub-format vtt \
  --output "data/raw/F02.%(ext)s" \
  "https://youtu.be/XXXX"

# 1-b. PyAV로 webm → 16kHz mono WAV 변환
conda run -n base python -c "
import av, wave
container = av.open('data/raw/F02.webm')
resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
pcm = []
for frame in container.decode(audio=0):
    for f in resampler.resample(frame): pcm.append(bytes(f.planes[0]))
container.close()
raw = b''.join(pcm)
with wave.open('data/raw/F02.wav', 'wb') as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
    wf.writeframes(raw)
print(f'완료: {len(raw)/2/16000:.1f}s')
"
```

### Step 2 — Ground Truth 생성

```bash
# VTT → TXT 변환 (vtt 파일이 data/raw/F02.ko.vtt 에 있을 때)
conda run -n base python -c "
import re, pathlib
vtt = pathlib.Path('data/raw/F02.ko.vtt').read_text(encoding='utf-8')
blocks = re.split(r'\n\n+', vtt.strip())
texts = []
for b in blocks:
    lines = b.strip().splitlines()
    tl = next((l for l in lines if '-->' in l), None)
    if not tl: continue
    texts += [l for l in lines if '-->' not in l
              and not l.startswith(('WEBVTT','Kind:','Language:')) and l.strip()]
pathlib.Path('data/ground_truth/F02.txt').write_text(' '.join(texts), encoding='utf-8')
print('완료')
"
```

### Step 3 — metadata.csv에 행 추가

```
file_id,utterance_type,duration_s,wav_path,url
F02,B,1200.0,/home/piai/project-ai/stt_comparison_research/data/raw/F02.wav,https://youtu.be/XXXX
```

> `wav_path`와 모든 경로는 **절대 경로** 필수. 상대 경로 쓰면 깨짐.
> `utterance_type` 컬럼은 현재 분석에서 사용 안 함 — 아무 값이나 넣어도 됨.

### Step 4 — 실험 실행

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

> results.csv는 **append 모드** — 이미 실험한 file_id는 metadata에서 빼고 새 파일만 넣어서 실행.

---

## 알려진 이슈

| 항목 | 내용 |
|------|------|
| ffmpeg 없음 | PyAV로 대체 (위 Step 1 방법 사용) |
| conda run + 상대경로 | cwd를 바꾸지 않음 → 모든 경로 절대경로 필수 |
| LD_LIBRARY_PATH | 매 세션마다 export 필요 (libcublas.so.12 문제) |
| whisper-medium-ko 경로 | configs/experiment_config.yaml의 model_id = `/home/piai/project-ai/stt_comparison_research/models/whisper-medium-ko` |
| results.csv append | 같은 파일 두 번 돌리면 중복 행 생김 |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| CER | 문자 오류율 (주지표) |
| WER | 단어 오류율 (보조) |
| ins_rate | insertions / ref_words — 환각 경향 |
| del_rate | deletions / ref_words — 누락 경향 |
| length_ratio | hyp_words / ref_words — 1보다 크면 과다 생성 |
| substitutions / deletions / insertions | 오류 유형 분해 |

---

## 파일 구조

```
stt_comparison_research/
├── CLAUDE.md
├── configs/experiment_config.yaml         # 파라미터 고정 (수정 금지)
├── data/
│   ├── metadata.csv                       # 실험 파일 목록
│   ├── raw/                               # WAV (git 제외)
│   └── ground_truth/{file_id}.txt         # 수동 자막 텍스트
├── models/whisper-medium-ko/              # CT2 변환 완료 (git 제외)
├── pipeline/stt/
│   ├── faster_whisper_runner.py
│   ├── kospeech_runner.py                 # 보류
│   └── api_runner.py                      # 보류
├── evaluation/
│   ├── normalizer.py
│   └── metrics.py                         # Metrics 데이터클래스 (S/D/I 포함)
├── experiments/run_all_models.py
├── analysis/
│   ├── statistical_tests.py
│   └── plot_generators.py                 # 데이터 모인 후 구현
├── results/raw/results.csv               # 누적 결과 (git 제외)
└── scripts/
    ├── download_audio.py                 # ffmpeg 필요 — 현재 미사용
    ├── prepare_ground_truth.py
    └── run_experiment.py
```

---

## VAD 연구와의 관계

`vad_stt_research/`와 별도 운영. 본 연구에서 선정된 최선 모델이 VAD 연구의 고정 백엔드로 투입될 예정. xsbdRlpLYhc WAV는 `vad_stt_research/data/raw/` 경로를 공유 참조.
