# VAD STT 연구 — 실험 요약 보고서

> 작성일: 2026-06-24  
> 상태: 탐색 실험 완료 / 정식 실험 대기 중

---

## 실험 개요

한국어 롱폼 오디오에 Silero VAD 전처리를 붙였을 때 STT 정확도·속도가 개선되는지, 그리고 그 효과가 VAD 덕인지 디코딩 파라미터 덕인지 분리해 정량화하는 실험.

**탐색 실험 모델**: faster-whisper large-v3 (GPU: RTX 2080 8GB × 2, int8_float16)

> **정식 실험(Phase 1)부터는 `faster_whisper_large_v3_turbo` 고정** — STT 비교 연구(Stage 1) 선정 결과. `configs/experiment_config.yaml` 반영 완료.

**실험 조건 3가지**:

| 조건 | 내용 |
|------|------|
| A (Vanilla) | faster-whisper 기본값 그대로 (`condition_on_previous_text=True`) |
| A′ (배치만) | 디코딩 파라미터 통일 (`condition_on_previous_text=False` 등), VAD 없음 |
| B (VAD+배치) | Silero VAD → 청크 분리 → A′ 파라미터로 전사 |

**측정 컬럼**:

| 컬럼 | 설명 | 정식 평가 사용 |
|------|------|--------------|
| `wer` | 단어 오류율 | ✅ |
| `cer` | 문자 오류율 | ✅ 주지표 |
| `rtf_mean` | 실시간 처리 배율 | ✅ |
| `hallucination_per_hour` | 시간당 할루시네이션 수 | 참고용만 (GT 없이 역산 한계) |
| `timestamp_drift_late_s` | 후반부 타임스탬프 드리프트 | 참고용만 |
| `vad_time_s` | VAD 처리 시간 | 참고용 (오버헤드 파악) |
| `n_chunks` | VAD가 분리한 청크 수 | 참고용 |

---

## 실험 1 — 토이셋 Smoke Test (2026-06-22)

파일 2개, GT 없음 → RTF와 파이프라인 동작 확인 목적.

| file_id | 길이 | 무음 비율 | RTF(A) | RTF(A′) | RTF(B) |
|---------|------|-----------|--------|---------|--------|
| ycsEx4d3Ri4 | 15.4분 | 18.8% | 0.117 | **0.075** | 0.093 |
| wgP9ARwAbNw | 8.9분 | 12.7% | 0.172 | **0.118** | 0.130 |

파이프라인 전체 에러 없이 통과. 두 파일 모두 무음 비율이 낮아 VAD(B) 오버헤드 > 이득.

---

## 실험 2 — YouTube 영상 (2026-06-24)

파일 1개, 수동 자막 GT 있음 → WER/CER 포함 전체 지표 최초 측정.

**파일**: xsbdRlpLYhc (세바시 AI 강연, 72.9분, 무음 비율 4.8%)

| 지표 | A (Vanilla) | A′ (배치만) | B (VAD+배치) |
|------|------------|------------|-------------|
| WER | 23.5% | **17.7%** | 18.8% |
| CER | 15.9% | **9.3%** | 10.1% |
| RTF | 0.127 | **0.074** | 0.086 |
| 할루시네이션/시간 | 231.3 | **128.4** | 172.0 |
| 타임스탬프 드리프트 | **1.21s** | 1.40s | 1.38s |

**비교 축별 요약**:

- **A → A′**: `condition_on_previous_text=False` 단 하나의 파라미터 변경이 WER −25%, CER −41%, RTF −42%, 할루시네이션 −44%. 가장 큰 단일 변수.
- **A′ → B**: 무음 4.8% 환경에서 VAD는 전 지표 악화. VAD 오버헤드 168.9초가 무음 제거 이득(≈3.5분)을 거의 상쇄.

---

## 핵심 발견

1. **이 데이터에서 최선 조건은 A′** — 별도 VAD 없이 파라미터만 조정한 A′이 WER·CER·RTF·할루시네이션 모두 우세.
2. **VAD 손익분기 미달** — 무음 비율이 낮은(4.8~18%) 데이터에서는 VAD 오버헤드가 이득을 초과. 연구 설계 예측(≥50%에서 이득)과 일치.
3. **파라미터 효과 > VAD 효과** — 배치 파라미터 최적화가 VAD 전처리보다 실질적으로 더 큰 개선을 만들었음.

---

## 현재 한계 및 정식 실험 필요성

- **n=3 (토이 2개 + YouTube 1개), 모두 low_silence** → Wilcoxon 검정 불가, high_silence 데이터 없음
- **반복 측정 1회** → RTF 분산 미확보
- VAD 이득이 나타날 것으로 예측되는 high_silence(≥50%) 데이터 미수집

---

## 향후 계획

| 단계 | 내용 | 상태 |
|------|------|------|
| Phase 1 데이터 수집 | AI Hub 한국어 음성 — 60분+, low 10개 + high 10개 | 승인 대기 |
| 정식 실험 | repeats=3, 전체 지표 측정 | 데이터 후 착수 |
| 통계 분석 | Wilcoxon signed-rank, breakeven_analysis | 코드 구현 완료 |
| 시각화 | plot_generators.py 5종 | 데이터 후 구현 |
| Phase 2 | PyAnnote Diarization + `faster_whisper_large_v3_turbo` | STT 연구 완료 → **모델 확정** |

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `results/SMOKE_TEST_REPORT.md` | 토이셋 실험 상세 |
| `results/YT_TEST_REPORT.md` | YouTube 실험 상세 |
| `results/raw/results_yt.csv` | 원본 수치 |
| `configs/experiment_config.yaml` | 실험 파라미터 전체 |
