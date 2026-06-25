# VAD STT 연구 — 진행 현황 / 내일 전달 사항

> 작성: 2026-06-25 야간 | Phase 1 최종 실험 마무리 단계

---

## 1. 한 줄 요약

롱폼 한국어 STT에서 **VAD 전처리(B)는 large-v3-turbo 환경에서 속도·정확도 모두 순손해**, 개선은 사실상 **파라미터 통일(A′)이 전담**. → **A′ 채택, VAD 비권장.** (1·2·3차 일관)

---

## 2. 데이터셋 — 깨끗한 10파일 확정 (5 high + 5 low)

| 그룹 | 파일 | 비고 |
|------|------|------|
| high (무음≥50%) | H01, H02, H03, H04, H05 | H03=RviNZGjJBpI(교체본) |
| low (무음<20%) | L01, L02, L03, L04, L05 | L02=IS6PybD5t3M, L04=1ZtxBPRSMZA(교체본) |

- 사전검사 강화: VTT 스크리닝만으론 부족(자막 밖 인트로/아웃트로 미반영) → **Silero 실측 무음 + STT 샘플**까지 봐야 함을 확인. 이 과정에서 q-XmkvrNvis(음악성 콘텐츠), NGLOS(실측 무음 43%=mid) 등 탈락 처리.
- L04 GT는 롤링자막이라 cue 단위로 재생성(병합 안 함).

---

## 3. 현재 결과 (7파일 기준, Wilcoxon n=7 — 10파일로 갱신 진행 중)

| 지표 | A→A′ (파라미터) | A′→B (VAD) | 신뢰도 |
|------|----------------|-----------|--------|
| WER | **−8.7%p (p=0.031 ✔)** | +5.3%p 악화 (무의) | 높음 |
| CER | **−9.1%p (p=0.016 ✔)** | +5.3%p 악화 (무의) | 높음 |
| 할루시네이션/h | **−63% (p=0.016 ✔)** | +27 증가 (p=0.016 ✔, 악화) | 높음 |
| RTF | **−42% (p=0.016 ✔)** | +90% 느려짐 (p=0.016 ✔) | **가장 높음** |
| 후반 Δt | 무의 (n=4) | 무의 | **낮음 → 강등** |

- **손익분기점 없음:** 무음 70%여도 B가 A′보다 느림 (turbo가 너무 빨라 VAD 오버헤드 회수 불가).
- **RTF가 가장 단단한 근거** — 측정 규약 엄격(warmup1+3회), 교란 없음.

---

## 4. 주요 의사결정 (확정)

1. **모델: large-v3-turbo 고정** (Stage 1 STT 연구 선정작, config "수정 금지").
2. **후반 Δt 지표 = 강등**(삭제 아님). 별도 "한계 및 향후과제" 섹션으로 분리. 사유 3가지: ①세그먼트 수 교란(A가 할루시로 많이 뱉어 Δt 인위적 저하) ②high 그룹이 53~59분으로 짧아 후반부 표본 부족 ③세그먼트 레벨 정밀도 한계. 정밀 측정은 단어 단위 forced alignment(향후).
3. **가설 3(padding 민감도) = 진행 예정.** 비용 문제로 대표 2개(H03 high + L03 low)만 9조합 스윕.

---

## 5. ⚠️ 재개 시 가장 먼저 — L04 실험 (아직 안 돌아감!)

**L04는 실행되지 않았다.** 자동 체인이 버그로 멈춤: 대기 조건 `pgrep -f "run_experiment.py.*metadata_run"`가 **체인 자신의 명령줄을 매칭**해 무한 대기에 빠짐(L04를 끝내 못 띄움). H03+L02는 정상 완료(`results_run2.csv` 6행 안전).

**→ 재개하려면 L04를 직접 실행 (체인 쓰지 말 것):**
```bash
cd /home/piai/project-ai/vad_stt_research
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
PYTHONPATH=. /home/piai/anaconda3/bin/python scripts/run_experiment.py \
  --metadata data/metadata_L04.csv --config configs/experiment_config.yaml \
  --gt_dir data/ground_truth --output results/raw/results_run3.csv --repeats 3
```
소요 ~45~60분. (멈춘 체인 bash가 아직 떠있으면 `pkill -f "while pgrep.*metadata_run"`로 정리)

## 5-1. L04 끝난 뒤 (Phase 1 마무리)
1. `results_7.csv` + `results_run2.csv` + `results_run3.csv` → **results_10.csv** concat
2. 분석 3종(n=10): `statistical_tests.py` / `breakeven_analysis.py` / `plot_generators.py` (`--results results/raw/results_10.csv --metadata data/metadata_10.csv`)
3. **가설3 민감도** (~1h): `scripts/sensitivity_analysis.py --metadata data/metadata_10.csv --file_ids H03 L03 --output results/figures/sensitivity_wer.csv` → 그래프④
4. `results/PHASE1_EXPERIMENT_REPORT.md` 10파일+가설3로 갱신 (후반 Δt는 한계 섹션)
- 산출물 경로: 결과 `results/raw/`, 그래프/통계 `results/figures/`

## 5-2. 다음 연구 (Phase 1 후속) — VAD 모델 비교
계획서: `/home/piai/project-ai/VAD모델비교계획서.md` (**Silero vs WebRTC**, 실행비용·검출F1·downstream 3축)
**시작 전 구현 필요 2가지:**
- **WebRTC VAD 엔진** 신규 구현 — 현재 `pipeline/vad/`는 Silero만, `get_vad()`가 타 엔진 거부. `BaseVAD` 상속해 WebRTC(`webrtcvad`) 추가 + `get_vad()`에 등록
- **검출 정확도(F1) 평가** 신규 — speech/non-speech 프레임 단위로 VTT GT와 비교 (`evaluation/`에 추가)
- 2-arm(S/W) + 참조(A′), 데이터는 Phase 1 동일 풀, noisy 일부 포함 권장

---

## 6. 내일 공유 포인트 (요약)

- "깨끗한 10파일(5+5) 확보 완료, 사전검사 2단계(VTT+오디오실측)로 음악/인트로 함정 걸러냄."
- "핵심 결론은 RTF·WER·할루시네이션 3종으로 통계적으로 확정됨 → **A′ 채택, VAD 비권장**."
- "후반 타임스탬프 지표는 측정 한계로 결론에서 제외(한계 섹션으로), 향후 forced alignment로 정밀화."
- "가설 3(padding 민감도)은 대표 2파일로 곧 검증 → 그래프 추가 예정."
- 열린 질문: high 그룹을 60분+로 맞추면 후반 Δt도 살릴 수 있는데, 데이터 확보 난이도 vs 가치 판단 필요.
