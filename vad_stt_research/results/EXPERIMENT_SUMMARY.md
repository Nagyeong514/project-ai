# VAD STT 연구 — 탐색 실험 요약

> 작성일: 2026-06-24 | 상태: 탐색 실험 완료 / 정식 실험 대기 중

---

## 연구 질문

**"VAD를 붙이면 STT가 빨라지고 정확해지는가?"**

VAD 모델 비교가 아님. Silero VAD 단일 고정, 전처리 방식(A/A′/B)만 비교.

---

## 실험 조건

| 조건 | 내용 |
|------|------|
| A (Vanilla) | faster-whisper 기본값, `condition_on_previous_text=True` |
| A′ (파라미터만) | `condition_on_previous_text=False` 통일, VAD 없음 |
| B (VAD+파라미터) | Silero VAD → 청크 분리 → A′ 파라미터 |

비교 축: A→A′ = 파라미터 효과 / A′→B = VAD 순수 효과 / A→B = 전체 효과

> **탐색 실험 STT 모델**: faster-whisper large-v3 (turbo 선정 이전 실험)  
> **정식 실험(Phase 1)부터**: `faster_whisper_large_v3_turbo` 고정 (`configs/experiment_config.yaml` 반영 완료)

---

## 탐색 실험 1 — 토이셋 (2026-06-22, large-v3)

파일 2개, GT 없음 → RTF·파이프라인 동작 확인만.

| file_id | 길이 | 무음 비율 | RTF(A) | RTF(A′) | RTF(B) |
|---------|------|-----------|--------|---------|--------|
| ycsEx4d3Ri4 | 15.4분 | 18.8% | 0.117 | **0.075** | 0.093 |
| wgP9ARwAbNw | 8.9분 | 12.7% | 0.172 | **0.118** | 0.130 |

파이프라인 전체 에러 없이 통과. 무음 낮아 VAD(B) 오버헤드 > 이득.

---

## 탐색 실험 2 — YouTube (2026-06-24, large-v3)

파일 1개, GT 있음 → WER/CER 포함 전체 지표 최초 측정.

**파일**: xsbdRlpLYhc (72.9분, 무음 4.8%)

| 지표 | A | A′ | B |
|------|---|----|---|
| WER | 23.5% | **17.7%** | 18.8% |
| CER | 15.9% | **9.3%** | 10.1% |
| RTF | 0.127 | **0.074** | 0.086 |
| 할루시네이션/시간 | 231.3 | **128.4** | 172.0 |

---

## 핵심 발견

1. **파라미터 효과 > VAD 효과** — `condition_on_previous_text=False` 한 줄이 WER −25%, CER −41%
2. **VAD 손익분기 미달** — 무음 4.8~18% 환경에서 VAD 오버헤드 > 이득 (예측과 일치)
3. **high_silence 데이터 없이는 결론 못 냄** — VAD 이득은 무음 ≥50% 구간에서 기대

> ⚠️ 위 결과는 탐색 실험(large-v3, n=3)으로 참고용. 정식 실험은 turbo + AI Hub 데이터(60분+, n=20)로 재실행.

---

## 향후 계획

| 단계 | 내용 | 상태 |
|------|------|------|
| Phase 1 데이터 수집 | AI Hub 한국어 음성 — 60분+, low 10개 + high 10개 | 승인 대기 |
| 정식 실험 | turbo 고정, repeats=3 | 데이터 후 착수 |
| 통계 분석 | Wilcoxon signed-rank, breakeven_analysis | 코드 구현 완료 |
| 시각화 | plot_generators.py 5종 | 데이터 후 구현 |
| Phase 2 | PyAnnote Diarization + `faster_whisper_large_v3_turbo` | Phase 1 완료 후 |
