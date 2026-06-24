# CLAUDE.md — VAD STT 연구

---

## 전체 연구 로드맵

```
[Stage 1] stt_comparison_research/ — STT 모델 비교 → 최선 모델 선정 (완료)
               ↓ faster_whisper_large_v3_turbo 확정
[Stage 2] vad_stt_research/ — 이 연구
  Phase 1 (현재): 단일 화자 롱폼에서 Silero VAD 전처리 효과 정량화
  Phase 2 (예정): 다중 화자 — PyAnnote Diarization + turbo로 화자별 전사
```

---

## 연구 질문

**"VAD를 붙이면 STT가 빨라지고 정확해지는가?"**

VAD 모델 비교가 아님. VAD는 **Silero 단일 고정**, 전처리 방식(A/A′/B)만 비교.

---

## 실험 조건 3가지

- **A**: Vanilla — faster-whisper 기본값, VAD 없음 (대조군)
- **A′**: 파라미터만 — `condition_on_previous_text=False` 통일, VAD 없음 (파라미터 효과 분리)
- **B**: VAD + 파라미터 — Silero VAD → 청크 분리 → faster-whisper (제안 파이프라인)

비교 축: A→A′ = 파라미터 효과 / A′→B = VAD 순수 효과 / A→B = 전체 효과

---

## 현재 진행 상태 (2026-06-24 기준)

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | configs 세팅 | ✅ |
| 2 | 버그 수정 | ✅ |
| 3 | 파이프라인 전체 구현 | ✅ |
| 4 | 탐색 실험 (토이셋 + YouTube, large-v3) | ✅ `results/EXPERIMENT_SUMMARY.md` 참조 |
| 5 | **Phase 1 정식 데이터 수집** | 🔲 AI Hub 승인 대기 |
| 6 | 정식 실험 (turbo 고정, repeats=3) | 🔲 |
| 7 | statistical_tests + breakeven_analysis | 🔲 |
| 8 | plot_generators.py (5종 시각화) | 🔲 |

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
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
PYTHONPATH=/home/piai/project-ai/vad_stt_research python scripts/run_experiment.py \
  --metadata /home/piai/project-ai/vad_stt_research/data/metadata.csv \
  --config /home/piai/project-ai/vad_stt_research/configs/experiment_config.yaml \
  --output /home/piai/project-ai/vad_stt_research/results/raw/results.csv \
  --repeats 3
```

---

## 다음 할 일

### Phase 1 — 데이터 대기 중

**데이터 조건**:
- 각 60분 이상 WAV + ground_truth JSON
- ground_truth 형식: `{"text": "전체 텍스트", "segments": [{"start": 0.0, "end": 2.5}]}`
- 목표: low_silence(<0.20) 10개 + high_silence(≥0.50) 10개

**진행 순서**:
1. AI Hub 데이터 승인 후 `data/raw/`에 WAV 배치
2. `python scripts/compute_silence_ratio.py data/raw/ --output data/metadata.csv`
3. `python scripts/run_experiment.py --repeats 3`
4. `analysis/statistical_tests.py` → Wilcoxon 검정
5. `analysis/breakeven_analysis.py` → 무음 비율 손익분기 계산
6. `plot_generators.py` 구현 (데이터 수집 후 착수)

### Phase 2 — Phase 1 완료 후

- **STT 백엔드**: `faster_whisper_large_v3_turbo` (확정)
- PyAnnote speaker-diarization 모델 필요 (별도 구현, HF_TOKEN 필요)
- 폐쇄망 환경: 모델 파일 수동 다운로드 후 로컬 경로로 변경

---

## 알려진 버그 및 수정 이력

| 항목 | 내용 |
|------|------|
| `temperature_increment_on_fallback` | faster-whisper 미지원. `temperature` 리스트 방식으로 수정 완료 |
| `libcublas.so.12 not found` | `nvidia-cublas-cu12` 설치 + LD_LIBRARY_PATH 설정. 매 세션 export 필요 |
| PYTHONPATH 미설정 | 실행 시 `PYTHONPATH=/home/piai/project-ai/vad_stt_research` 명시 필요 |

---

## 파일 구조

```
vad_stt_research/
├── CLAUDE.md
├── README.md
├── configs/
│   └── experiment_config.yaml
├── data/
│   ├── metadata.csv                    # 정식 실험용 (현재: 토이셋 2개)
│   ├── metadata_myn.csv                # 수동 탐색 실험용
│   ├── metadata_yt.csv                 # YouTube 탐색 실험용
│   ├── raw/                            # WAV (git 제외)
│   └── ground_truth/                   # {file_id}.json
├── experiments/
│   ├── condition_a.py
│   ├── condition_a_prime.py
│   └── condition_b.py
├── pipeline/
│   ├── merge/chunk_extractor.py
│   ├── stt/faster_whisper_runner.py
│   └── vad/
│       ├── __init__.py                 # get_vad() — silero 단일 지원
│       ├── base.py
│       └── silero_vad.py               # VAD 고정 엔진
├── evaluation/
│   ├── wer_cer.py
│   ├── hallucination.py
│   └── timestamp_eval.py
├── analysis/
│   ├── statistical_tests.py
│   └── breakeven_analysis.py
├── results/
│   ├── EXPERIMENT_SUMMARY.md           # 탐색 실험 요약 (large-v3 기준, 참고용)
│   └── raw/
│       ├── drift_by_time/
│       ├── results_myn.csv
│       └── results_yt.csv
└── scripts/
    ├── run_experiment.py
    ├── compute_silence_ratio.py
    └── sensitivity_analysis.py         # Silero 파라미터 민감도 스윕

```

---

## 협업 규칙

1. 한 번에 하나의 파일 또는 태스크만 진행
2. 전체 코드를 한 번에 쏟아내지 않음
3. 코드 주석에 이모티콘 사용 금지
4. 테스트 성공 확인 후에만 다음 단계로 진행
