# VAD-STT 연구 진행 보고서

> 작성일: 2026-06-22  
> 작성자: AI Research Engineering Team

---

## 1. 연구 목적 한 줄 요약

롱폼 오디오(1시간+)에 VAD 전처리를 붙이면 **진짜 빨라지고 정확해지는지**, 그리고 빨라진 게 **VAD 덕인지 배치 추론 덕인지**까지 축을 갈라 정량화하는 실험.

---

## 2. 핵심 질문

| 비교 | 측정하는 것 |
|------|------------|
| A → A′ | 배치 추론의 순수 효과 |
| A′ → B | VAD(무음 제거)의 순수 효과 |
| A → B | 전체 파이프라인 효과 |

---

## 3. 실험 조건 (3-arm)

### 조건 A — Vanilla (대조군)
- faster-whisper를 **기본 설정 그대로** 전체 오디오에 순차 적용
- `condition_on_previous_text=True` → Whisper out-of-box 상태 재현
- 할루시네이션·타임스탬프 드리프트의 기준선 역할

### 조건 A′ — 배치만, VAD 없음 (배치 효과 분리)
- VAD 없이 faster-whisper로 전체 오디오 처리
- 통일 디코딩 파라미터 적용 (`condition_on_previous_text=False`)
- A와의 차이 = **배치·파라미터 조정의 순수 효과**

### 조건 B — VAD + 배치 (제안 파이프라인)
- Silero VAD로 무음 제거 → 발화 구간만 청크로 분리 → 배치 STT
- A′와의 차이 = **무음 제거(VAD)의 순수 효과**
- 파이프라인: `오디오 → Silero VAD → 청크 추출 → faster-whisper 배치 → 타임스탬프 재매핑`

---

## 4. 평가 지표

| 지표 | 측정 방법 | 검증 가설 |
|------|----------|----------|
| WER / CER | 정답 스크립트 대비 | 가설 1 (정확도) |
| 할루시네이션율 | 무음 구간 내 토큰 + n-gram 반복 검출, 시간당 횟수 | 가설 1 |
| Timestamp Drift (Δt) | 후반부(50분↑) 세그먼트 시작 시점 절대 오차 평균 | 가설 1 |
| RTF | 전체 처리시간 / 오디오 길이 (낮을수록 빠름) | 가설 2 (속도) |
| VAD 오버헤드 | 조건 B에서 VAD 연산 시간 별도 계측 | 가설 2 손익분기점 |
| VAD 파라미터 민감도 | threshold × speech_pad_ms 스윕 WER | 가설 3 (경계 클리핑) |

통계: Wilcoxon signed-rank test (paired, α = 0.05)

---

## 5. 하드웨어 및 모델 설정

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX 2080 8GB × 2 |
| STT 백엔드 | faster-whisper (CTranslate2) |
| 모델 | Whisper large-v3 |
| 연산 타입 | `int8_float16` (8GB VRAM OOM 방지) |
| 언어 | 한국어 (ko) |

---

## 6. 프로젝트 구조

```
vad_stt_research/
├── configs/
│   └── experiment_config.yaml     # 모든 실험 파라미터 고정값
├── data/
│   ├── raw/                       # 오디오 파일 (60분 이상 WAV)
│   └── ground_truth/              # 정답 JSON (텍스트 + 타임스탬프)
├── pipeline/
│   ├── vad/                       # VAD 엔진 4종 + 팩토리
│   │   ├── base.py                # 공통 인터페이스, padding·merge·split 로직
│   │   ├── silero_vad.py          # 메인 (조건 B)
│   │   ├── pyannote_vad.py        # WhisperX 계열 비교군
│   │   ├── webrtc_vad.py          # 경량 비교군
│   │   ├── librosa_vad.py         # 규칙 기반 하한선
│   │   └── __init__.py            # get_vad() 팩토리 진입점
│   ├── stt/
│   │   └── faster_whisper_runner.py  # 단건/배치 전사 + 타임스탬프 재매핑
│   └── merge/
│       └── chunk_extractor.py     # VAD 세그먼트 → 청크 WAV 분리
├── experiments/
│   ├── condition_a.py             # 조건 A 실행 함수
│   ├── condition_a_prime.py       # 조건 A′ 실행 함수
│   └── condition_b.py             # 조건 B 실행 함수
├── evaluation/
│   ├── wer_cer.py                 # WER/CER 계산 (정규화 포함)
│   ├── hallucination.py           # 할루시네이션 감지
│   └── timestamp_eval.py          # Δt 드리프트 (구간별 버킷 분석)
├── analysis/
│   ├── statistical_tests.py       # Wilcoxon + 3쌍 비교 자동화
│   ├── breakeven_analysis.py      # 손익분기점 계산
│   └── plot_generators.py         # [미구현] 5종 시각화 — 데이터 수집 후 작성
└── scripts/
    ├── compute_silence_ratio.py   # Step 1: metadata.csv 생성
    ├── run_experiment.py          # Step 2: 3-arm 실험 전체 실행
    └── sensitivity_analysis.py    # Step 3: 파라미터 민감도 스윕
```

---

## 7. 현재 구현 현황

| 항목 | 상태 | 비고 |
|------|------|------|
| 실험 설계 및 configs | ✅ 완료 | 모든 파라미터 고정 |
| pipeline/vad (4종 엔진) | ✅ 완료 | silero·pyannote·webrtc·librosa |
| pipeline/vad 팩토리 (`get_vad`) | ✅ 완료 | lazy import 적용 |
| pipeline/stt (faster-whisper) | ✅ 완료 | 청크 오프셋 재매핑 포함 |
| pipeline/merge (청크 추출) | ✅ 완료 | |
| experiments/condition_a | ✅ 완료 | |
| experiments/condition_a_prime | ✅ 완료 | |
| experiments/condition_b | ✅ 완료 | VAD 시간 별도 계측 포함 |
| evaluation/wer_cer | ✅ 완료 | 한국어 정규화 규칙 고정 |
| evaluation/hallucination | ✅ 완료 | 2가지 판정 기준 구현 |
| evaluation/timestamp_eval | ✅ 완료 | 10분 버킷 드리프트 분석 포함 |
| analysis/statistical_tests | ✅ 완료 | 3쌍 자동 비교 |
| analysis/breakeven_analysis | ✅ 완료 | 선형 회귀 기반 임계점 추정 |
| scripts/compute_silence_ratio | ✅ 완료 | |
| scripts/run_experiment | ✅ 완료 | |
| scripts/sensitivity_analysis | ✅ 완료 | |
| **analysis/plot_generators** | 🔲 **미구현** | 데이터 수집 후 작성 예정 |

---

## 8. 앞으로 남은 단계

```
Step 4  data/raw/ 에 오디오 파일 배치
        python scripts/compute_silence_ratio.py data/raw/
        → data/metadata.csv 생성 (무음 비율·그룹 분류)

Step 5  python scripts/run_experiment.py
        → results/raw/results.csv 생성 (조건별 WER/RTF/할루시네이션/Δt)

Step 6  statistical_tests.py + breakeven_analysis.py 실행
        → 가설 1·2·3 검정, 손익분기점 수치 도출

Step 7  실제 결과 데이터 기반으로 plot_generators.py 구현
        → 5종 시각화 자동 산출
```

---

## 9. 산출 예정 시각화 (Step 7)

| 그래프 | 검증 가설 | 내용 |
|--------|----------|------|
| Grouped Bar | 가설 1 | 조건별 WER·할루시네이션 정면 비교 |
| Waterfall | 가설 2 | 속도 향상 요인 분해 (배치 기여분 vs VAD 기여분) |
| Scatter + 회귀선 | 가설 2 | 무음 비율 vs RTF 이득 → 손익분기점 |
| Multi-line | 가설 3 | VAD 파라미터(threshold·padding) 민감도 WER 곡선 |
| Timeline | 가설 1 | 조건별 타임스탬프 드리프트 누적 추이 |
