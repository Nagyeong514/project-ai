# VAD STT 연구 — 진행 현황 / 전달 사항

> 갱신: 2026-06-26 | Phase 1 완료 + 후속(엔진비교) 완료. 전부 커밋·푸시됨.

---

## 1. 한 줄 요약

롱폼 한국어 STT에서 **VAD 전처리는 large-v3-turbo 환경에서 속도 순손해 + 할루시네이션 유발**, 개선은 **파라미터 통일(A′)이 전담**. → **A′ 채택, VAD 비권장.** 굳이 VAD 엔진을 고르면 WebRTC(Silero보다 116배 저렴).

---

## 2. 세 연구 상태 (전부 완료)

| 연구 | 위치 | 상태 |
|------|------|------|
| **STT 모델 비교** (Stage 1) | `stt_comparison_research/` | ✅ 완료 — turbo 선정 |
| **VAD 쓸까말까** (Phase 1) | `vad_stt_research/` | ✅ 완료 — 3가설 전부 |
| **VAD 엔진비교** (후속) | `vad_model_comparison/` | ✅ 완료 — Silero vs WebRTC |

---

## 3. Phase 1 최종 결과 (10파일, Wilcoxon n=10)

| 지표 | A→A′ (파라미터) | A′→B (VAD) |
|------|----------------|-----------|
| WER | 0.299→0.225 (p=.020 ✔) | +악화 (ns) |
| CER | 0.172→0.094 (p=.002 ✔) | +악화 (ns) |
| 할루시네이션/h | 273→100 (p=.002 ✔) | 100→119 (p=.027 ✔ 악화) |
| RTF | 0.034→0.021 (p=.002 ✔) | +84% (p=.002 ✔ 느려짐) |

- 가설1 **기각**, 가설2 **기각**(손익분기 없음), 가설3 **지지**(padding 0→400 WER 감소, 기본값 400 최적).
- 후반 Δt는 측정 한계로 **한계 섹션에 강등**(결론서 제외).
- 데이터: 깨끗한 10파일(5 high + 5 low). 사전검사 2단계(VTT 스크리닝 + Silero 실측·STT 샘플).
- 보고서: `results/PHASE1_EXPERIMENT_REPORT.md`, 그래프 5종 `results/figures/`.

---

## 4. VAD 엔진비교 결과 (`vad_model_comparison/`)

- WebRTC가 **116배 빠름**(p=.002), F1도 우위(recall 견인, p=.004). 단 두 엔진 다 참조(VAD없음)보다 WER·할루시 악화.
- → "VAD 쓴다면 WebRTC, 근본적으론 turbo서 VAD 비권장"(Phase 1 재확인).
- 보고서: `vad_model_comparison/COMPARISON_REPORT.md`, 그래프 4종.

---

## 5. 남은 것 (전부 선택 사항)

| 항목 | 비고 |
|------|------|
| STT Wilcoxon 통계 | n=5라 검정력 약함 — 선택적 (CLAUDE.md에 미구현 명시) |
| STT 데이터 보강 | xsbdRlpLYhc(2인 대화 레거시, wav도 삭제됨)을 1인 파일로 교체하면 더 엄밀. 단 현 결론(turbo)은 F02~F05 4개로도 유효 |
| 엔진비교 메모리(peak) 지표 | GPU/CPU 공정 측정 이슈로 보류 |
| 엔진비교 noisy 데이터 | 잡음 환경 가설2 재검증용 (추가 데이터 필요) |
| Phase 2 | 다중화자 — PyAnnote Diarization + turbo (미착수) |

> **주의:** STT의 xsbdRlpLYhc.wav는 정리 때 삭제됨. STT 재현 시 재다운로드 필요(GT txt·results는 남음).

---

## 6. 핵심 결정 기록

1. **모델 large-v3-turbo 고정** (Stage 1 선정작, config 수정 금지).
2. **후반 Δt 강등** — 세그먼트 수 교란 + 표본 부족(n=6). 정밀화는 단어 단위 forced alignment(Phase 2).
3. **백그라운드 체인 금지 교훈:** `pgrep -f "...metadata_run"` 패턴이 체인 자기 명령줄을 매칭해 무한루프. 실험은 체인 말고 직접 실행, 정리는 PID로.
