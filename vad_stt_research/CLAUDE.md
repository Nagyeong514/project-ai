# CLAUDE.md — VAD STT 연구

---

## 전체 연구 로드맵

```
[Stage 1] stt_comparison_research/ — STT 모델 비교 → 최선 모델 선정
               ↓
[Stage 2] vad_stt_research/ — 이 연구
  Phase 1 (현재): 단일 화자 롱폼에서 Silero VAD 전처리 효과 정량화
  Phase 2 (예정): 다중 화자 — PyAnnote Diarization + Stage 1 선정 모델로 화자별 전사
```

---

## Phase 1 한 줄 요약

롱폼 오디오(1시간+)에 VAD 전처리를 붙이면 빨라지고 정확해지는지, 그리고 빨라진 게 VAD 덕인지 배치 추론 덕인지 축을 갈라 정량화하는 실험.

---

## 실험 조건 3가지

- **A**: Vanilla — faster-whisper 기본값, VAD 없음 (대조군)
- **A′**: 배치만 — 통일 디코딩 파라미터 적용, VAD 없음 (배치 효과 분리)
- **B**: VAD + 배치 — Silero VAD → 청크 분리 → faster-whisper (제안 파이프라인)

비교 축: A→A′ = 배치 효과 / A′→B = VAD 순수 효과 / A→B = 전체 효과

---

## 현재 진행 상태 (2026-06-24 기준)

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | configs 세팅 | ✅ |
| 2 | 버그 수정 | ✅ |
| 3 | 파이프라인 전체 구현 | ✅ |
| 4 | compute_silence_ratio → metadata.csv | ✅ 토이셋 2개 (GT 없음, 60분 미만) |
| 5 | run_experiment → results.csv | ✅ smoke test + YouTube 실험 완료 |
| 6 | **Phase 1 정식 데이터 수집** | 🔲 AI Hub 승인 대기 |
| 7 | statistical_tests + breakeven_analysis | 🔲 데이터 부족 |
| 8 | plot_generators.py (5종 시각화) | 🔲 |

실험 결과 보고서:
- `results/SMOKE_TEST_REPORT.md` — 토이셋 (GT 없음, RTF·무음비율만 측정)
- `results/YT_TEST_REPORT.md` — xsbdRlpLYhc (세바시 72.9분, GT 있음, WER/CER 측정)

---

## 환경 설정

**실행 환경**: Python 3.13, Anaconda, RTX 2080 8GB × 2

**매 세션 필수**:
```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
```
이유: 시스템에 `libcublas.so.13`만 있고 CTranslate2는 `.so.12`를 요구. `nvidia-cublas-cu12`로 해결.

**실험 실행 명령어**:
```bash
cd /home/piai/project-ai/vad_stt_research
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
PYTHONPATH=/home/piai/project-ai/vad_stt_research python scripts/run_experiment.py \
  --metadata data/metadata.csv \
  --config configs/experiment_config.yaml \
  --output results/raw/results.csv \
  --repeats 3
```

---

## 다음 할 일

### Phase 1 (단일 화자 롱폼) — 데이터 대기 중

**데이터 조건** (`data/metadata.csv` 기준):
- 각 60분 이상 WAV, ground_truth JSON 필수
- ground_truth 형식: `{"text": "전체 텍스트", "segments": [{"start": 0.0, "end": 2.5}]}`
- 목표: low_silence 10개 + high_silence 10개 (silence_ratio 기준: <0.20 / ≥0.50)

**현재 metadata.csv 상태** — 토이셋 2개만 등록 (60분 미만, GT 없음, 정식 실험 불가):
```
ycsEx4d3Ri4  15.4분  silence_ratio=0.19  low_silence
wgP9ARwAbNw   8.9분  silence_ratio=0.13  low_silence
```

**진행 순서**:
1. AI Hub 데이터 승인 후 `data/raw/`에 WAV 배치
2. `python scripts/compute_silence_ratio.py data/raw/ --output data/metadata.csv`
3. `python scripts/run_experiment.py --repeats 3`
4. `analysis/statistical_tests.py` → Wilcoxon 검정
5. `analysis/breakeven_analysis.py` → 무음 비율 손익분기 계산
6. `plot_generators.py` 구현 (데이터 수집 후 착수)

### Phase 2 (다중 화자 Diarization) — STT Stage 1 완료 후 착수

- `pipeline/vad/pyannote_vad.py` 구현 완료 (HF_TOKEN 환경변수 필요)
- 단, 화자 분리(Diarization)는 `pyannote/speaker-diarization` 모델 별도 필요
- Stage 1에서 최선 STT 모델 확정 후 투입

---

## 알려진 버그 및 수정 이력

| 항목 | 내용 |
|------|------|
| `temperature_increment_on_fallback` | faster-whisper 미지원. `temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` 리스트로 수정 완료 |
| `libcublas.so.12 not found` | `nvidia-cublas-cu12` 설치 + LD_LIBRARY_PATH 설정. 매 세션 export 필요 |
| `pipeline/vad/__init__.py` eager import | lazy import (`importlib`)로 수정. SpeechSegment, BaseVAD 패키지 레벨 노출 |
| PYTHONPATH 미설정 | 실행 시 `PYTHONPATH=/home/piai/project-ai/vad_stt_research` 명시 필요 |

---

## 파일 구조

```
vad_stt_research/
├── CLAUDE.md
├── README.md
├── configs/
│   └── experiment_config.yaml          # 수정 금지 (결과 재현성)
├── data/
│   ├── metadata.csv                    # 정식 실험용 (현재: 토이셋 2개)
│   ├── metadata_myn.csv                # MYN_07602 수동 실험용
│   ├── metadata_yt.csv                 # YouTube 수동 실험용
│   ├── raw/                            # WAV (git 제외)
│   └── ground_truth/                   # {file_id}.json
├── experiments/
│   ├── condition_a.py                  # Vanilla 조건
│   ├── condition_a_prime.py            # 배치만 조건 (DECODING_PARAMS_UNIFIED 정의)
│   └── condition_b.py                  # VAD + 배치 조건
├── pipeline/
│   ├── merge/chunk_extractor.py
│   ├── stt/faster_whisper_runner.py
│   └── vad/
│       ├── __init__.py                 # get_vad() 팩토리
│       ├── base.py                     # BaseVAD, SpeechSegment
│       ├── silero_vad.py               # Phase 1 사용 엔진
│       ├── pyannote_vad.py             # Phase 2 예정 (구현 완료, HF_TOKEN 필요)
│       ├── webrtc_vad.py
│       └── librosa_vad.py
├── evaluation/
│   ├── wer_cer.py
│   ├── hallucination.py
│   └── timestamp_eval.py
├── analysis/
│   ├── statistical_tests.py            # Wilcoxon 검정 (구현 완료)
│   └── breakeven_analysis.py           # 무음 비율 손익분기 (구현 완료)
├── results/
│   ├── EXPERIMENT_SUMMARY.md           # 탐색 실험 통합 요약
│   ├── SMOKE_TEST_REPORT.md
│   ├── YT_TEST_REPORT.md
│   └── raw/
│       ├── drift_by_time/              # 조건별 타임스탬프 드리프트 JSON
│       ├── results.csv                 # 전체 (git 제외)
│       ├── results_myn.csv             # MYN_07602 실험 결과
│       └── results_yt.csv              # YouTube 실험 결과
└── scripts/
    ├── run_experiment.py               # 메인 실행 진입점
    ├── compute_silence_ratio.py        # 무음 비율 계산 → metadata.csv
    └── sensitivity_analysis.py         # VAD 파라미터 민감도 분석

```

---

## 협업 규칙

1. 한 번에 하나의 파일 또는 태스크만 진행
2. 전체 코드를 한 번에 쏟아내지 않음
3. 코드 주석에 이모티콘 사용 금지
4. 테스트 성공 확인 후에만 다음 단계로 진행
