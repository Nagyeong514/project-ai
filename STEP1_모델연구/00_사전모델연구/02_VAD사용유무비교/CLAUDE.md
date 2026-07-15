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

## 현재 진행 상태 (2026-06-25 기준)

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | configs 세팅 | ✅ |
| 2 | 버그 수정 | ✅ |
| 3 | 파이프라인 전체 구현 | ✅ |
| 4 | 탐색 실험 (토이셋 + YouTube, large-v3) | ✅ (탐색 산출물은 정리 시 삭제됨) |
| 5 | Phase 1 데이터 수집 (YouTube 수동자막) | 🔄 7개 확정 + 3개 수집 중 |
| 6 | 정식 실험 (turbo 고정, repeats=3) | 🔄 3차(최종) 실험 진행 중 → `results/raw/results_7.csv` |
| 7 | statistical_tests + breakeven_analysis | ✅ CLI/main 추가, 실데이터 동작 확인 |
| 8 | plot_generators.py (5종 시각화) | ✅ 4종 동작, ④민감도는 sweep 데이터 생성 후 |
| 9 | 가설3 민감도 스윕 (sensitivity_analysis.py) | 🔲 3-arm 완료 후 H03+L03 대상 실행 예정 |

**실험 회차 기록:**
- 1차 (CPU VAD, 10파일): `results/PHASE1_EXPERIMENT_REPORT_CPU.md` — CPU VAD 버그, 참고용
- 2차 (GPU VAD, 8파일): `results/PHASE1_EXPERIMENT_REPORT_GPU.md` — H02/L02 제외
- 3차 (GPU, 7파일 + 3 예정): **진행 중** — 완료 후 최종 보고서 작성 예정

---

## 환경 설정

**실행 환경**: Python 3.13, Anaconda, RTX 2080 8GB × 2

**python 경로**: `python`이 PATH에 없음 → **`/home/piai/anaconda3/bin/python`** 사용 (또는 `conda run -n base`). yt-dlp도 PATH에 없고 `/home/piai/anaconda3/bin/yt-dlp`에 있음.

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

### Phase 1 — 실험 진행 중

**데이터 조건**:
- low_silence: 60분 이상 WAV / high_silence: 30분 이상 WAV + ground_truth JSON
- ground_truth 형식: `{"text": "전체 텍스트", "segments": [{"start": 0.0, "end": 2.5}]}`
- 목표: low_silence(<0.20) 5개 + high_silence(≥0.50) 5개 → **현재 7개 확정(`metadata_7.csv`) + 3개(high 1, low 2) 링크 탐색 중**

**신규 데이터 수집 파이프라인 (3개 추가용)**:
1. 후보 URL → `scripts/screen_candidates.py --urls urls.txt` (VTT만 받아 무음비율·길이 사전판별, 오디오 X)
2. 통과분 → `scripts/download_data.py` (WAV 16kHz 변환 + 3단계 검증: 기계번역/단어밀도/Silero 5분 발화율)
3. `scripts/prepare_ground_truth.py`로 VTT→GT JSON 생성
4. metadata에 행 추가 후 `run_experiment.py`로 3개만 추가 실행 → 결과 CSV concat → 최종 보고서

**분석 순서 (실험 완료 후)**:
1. `analysis/statistical_tests.py --results <csv>` → Wilcoxon 검정 (CLI 완성)
2. `analysis/breakeven_analysis.py --results <csv> --metadata <csv>` → 손익분기 (실질 RTF, CLI 완성)
3. `analysis/plot_generators.py --results <csv> --metadata <csv>` → 시각화 5종 (④는 sweep 데이터 필요)
4. **가설3 민감도** (3-arm 완료 후): `scripts/sensitivity_analysis.py --metadata data/metadata_10.csv --file_ids H03 L03 --output results/figures/sensitivity_wer.csv` → 그래프④ 생성. 전체 10개는 비용 과다(~6h)라 대표 2개(H03 high + L03 low)로 한정.

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
│   ├── metadata_7.csv                  # 현재 3차 실험용 (7파일) ← 사용 중
│   ├── metadata_8.csv                  # 2차(GPU 8파일) 기록
│   ├── metadata.csv                    # 구버전 10파일 기록
│   ├── raw/                            # WAV (git 제외) — H01~H05, L01~L05
│   └── ground_truth/                   # {file_id}.json + {file_id}.ko.vtt
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
│   ├── _plot_style.py                  # matplotlib Agg + 한글폰트(Noto CJK) 공용
│   ├── statistical_tests.py            # Wilcoxon (CLI: --results)
│   ├── breakeven_analysis.py           # 손익분기 (CLI: --results --metadata)
│   └── plot_generators.py              # 시각화 5종 (CLI: --results --metadata)
├── results/
│   ├── PHASE1_EXPERIMENT_REPORT_CPU.md # 1차 실험 (CPU VAD)
│   ├── PHASE1_EXPERIMENT_REPORT_GPU.md # 2차 실험 (GPU VAD)
│   ├── figures/                        # plot_generators 출력 (생성됨)
│   └── raw/
│       ├── drift_by_time/              # {file_id}_{cond}.json (그래프⑤용)
│       ├── results_7.csv               # 현재 3차 결과 ← 기록 중
│       └── results_8.csv               # 2차 결과
└── scripts/
    ├── run_experiment.py
    ├── screen_candidates.py            # 후보 URL VTT 사전 스크리닝
    ├── download_data.py                # WAV 변환 + 3단계 검증
    ├── prepare_ground_truth.py         # VTT → GT JSON
    ├── compute_silence_ratio.py
    └── sensitivity_analysis.py         # Silero 파라미터 민감도 스윕 (그래프④용)

```

---

## 협업 규칙

1. 한 번에 하나의 파일 또는 태스크만 진행
2. 전체 코드를 한 번에 쏟아내지 않음
3. 코드 주석에 이모티콘 사용 금지
4. 테스트 성공 확인 후에만 다음 단계로 진행
