# VAD-STT Research

롱폼 오디오(1시간+)에 VAD 전처리를 붙이면 진짜 빨라지고 정확해지는지,  
그리고 빨라진 게 VAD 덕인지 배치 추론 덕인지까지 축을 갈라 정량화하는 실험.

---

## 실험 조건

| 조건 | 설명 | VAD | 배치 | 디코딩 파라미터 |
|------|------|-----|------|----------------|
| A | Vanilla (대조군) | X | X | Whisper 기본값 |
| A′ | 배치만 | X | O | 통일값 |
| B | VAD + 배치 (제안) | O (Silero) | O | 통일값 |

- A → A′ 차이 = 배치 추론의 순수 효과
- A′ → B 차이 = VAD(무음 제거)의 순수 효과
- A → B 차이 = 전체 파이프라인 효과

---

## 환경 설정

**요구사항**

```bash
pip install -r requirements.txt
pip install nvidia-cublas-cu12   # CUDA 12 라이브러리 (CTranslate2 의존성)
```

**실행 전 필수 환경변수**

```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
```

> 시스템에 `libcublas.so.13`만 설치되어 있고 faster-whisper(CTranslate2)는 `.so.12`를 요구하기 때문.

---

## 실행 순서

```bash
cd vad_stt_research

# 1. 데이터셋 무음 비율 사전 계산 → data/metadata.csv
PYTHONPATH=. python scripts/compute_silence_ratio.py data/raw/

# 2. 3-arm 실험 전체 실행 → results/raw/results.csv
PYTHONPATH=. python scripts/run_experiment.py \
  --metadata data/metadata.csv \
  --config configs/experiment_config.yaml \
  --output results/raw/results.csv \
  --repeats 3

# 3. VAD 파라미터 민감도 분석 (가설 3)
PYTHONPATH=. python scripts/sensitivity_analysis.py \
  --audio data/raw/{file}.wav \
  --gt data/ground_truth/{file}.json
```

---

## 데이터 준비

```
data/
├── raw/               # WAV 파일 (60분 이상, git 제외)
└── ground_truth/      # {file_id}.json
```

ground_truth JSON 형식:

```json
{
  "text": "전체 정답 텍스트",
  "segments": [
    {"start": 0.0, "end": 3.2},
    {"start": 5.1, "end": 8.7}
  ]
}
```

데이터 소스: [AI Hub 한국어 음성](https://aihub.or.kr) (연구 목적 무료 신청)

---

## 프로젝트 구조

```
vad_stt_research/
├── configs/
│   └── experiment_config.yaml      # 모든 실험 파라미터
├── pipeline/
│   ├── vad/                        # VAD 엔진 4종 (Silero / pyannote / WebRTC / Librosa)
│   ├── stt/                        # faster-whisper 래퍼
│   └── merge/                      # VAD 세그먼트 → 청크 WAV 분리
├── experiments/
│   ├── condition_a.py              # 조건 A
│   ├── condition_a_prime.py        # 조건 A′
│   └── condition_b.py              # 조건 B
├── evaluation/
│   ├── wer_cer.py                  # WER / CER
│   ├── hallucination.py            # 할루시네이션 감지
│   └── timestamp_eval.py           # 타임스탬프 드리프트
├── analysis/
│   ├── statistical_tests.py        # Wilcoxon signed-rank test
│   └── breakeven_analysis.py       # 손익분기점 계산
├── scripts/
│   ├── compute_silence_ratio.py    # 무음 비율 사전 계산
│   ├── run_experiment.py           # 메인 실험 실행
│   └── sensitivity_analysis.py     # VAD 파라미터 민감도 스윕
├── results/
│   ├── EXPERIMENT_SUMMARY.md       # 탐색 실험 통합 요약
│   ├── SMOKE_TEST_REPORT.md        # 토이셋 실험 상세
│   └── YT_TEST_REPORT.md           # YouTube 실험 상세
└── CLAUDE.md                       # Claude Code 세션용 컨텍스트
```

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| WER / CER | 정답 스크립트 대비 단어/문자 오류율 |
| 할루시네이션율 | 무음 구간 내 토큰 + n-gram 반복, 시간당 횟수 |
| Timestamp Drift (Δt) | 후반부(50분↑) 세그먼트 타임스탬프 절대 오차 평균 |
| RTF | 처리시간 / 오디오 길이 (낮을수록 빠름) |

통계 검정: Wilcoxon signed-rank test (paired, α = 0.05)

---

## 하드웨어

- GPU: NVIDIA RTX 2080 8GB × 2
- 모델: Whisper large-v3 (`int8_float16`, VRAM 8GB 대응)
