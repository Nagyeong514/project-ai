# 4-arm 최종 비교 라운드 (Whisper turbo / Seamless / Wav2Vec2)

> `06_VAD재검증_v2/results/final_comparison/` — 각 모델의 지금까지 확정된 "최종 설정"으로
> CLIP1~4를 재전사하고 CER/WER을 3회 반복 측정 후 평균으로 계산했다. GT는 신규 축어대본이
> 준비되지 않아 기존 `data/ground_truth/CLIP{1..4}.txt`를 그대로 사용했다(사용자 확인).
> Whisper large-v3(non-turbo)는 STT_v2_EXPERIMENT_REPORT.md 5장(한계) 4번 항목에서 이미
> "정확도 우위 없이 속도만 느림 — 탈락 검토 대상"으로 명시돼 있어 이번 라운드에서 제외했다.
> 날짜: 2026-07-02

---

## 0. 요약

- **Whisper large-v3-turbo**: 이번 라운드에서 아무 조건도 바뀌지 않았다(디코딩파라미터는
  Stage2에서 0% 효과 확인돼 재검증 스킵, initial_prompt는 도메인 용어집이 아직 없어 스킵).
  CER은 Stage1과 완전히 동일 — 새로울 게 없다는 것 자체가 "이미 최종 설정에 도달했었다"는
  확인.
- **Seamless-M4T-v2-Large**: `num_beams=5, repetition_penalty=1.3`를 필수 적용한 결과,
  **CLIP2·CLIP4는 크게 개선(0.9604→0.6784, 0.9187→0.5447)**됐지만 **CLIP1은 오히려
  악화(0.3333→0.4296, +29%)**했다. CLIP3는 변화 없음(0.2427→0.2427). 즉 이 설정은
  "만능 개선"이 아니라 클립 의존적이다 — 표1·2절 참고.
- **RTF 게이트에서 Seamless는 4클립 전부 탈락**(0.21~0.30, v1 기준 0.10 초과) — beam=5가
  속도를 추가로 깎아먹었다. 그 결과 **v1 선정룰(RTF≤0.10 통과 후 CER 최소) 적용 시 4개
  파일 전부 Whisper large-v3-turbo가 최종 승자**로 확정됐다(표3).
- **Wav2Vec2-XLSR-Korean**: 재전사 없이 v2 원본 필터후 텍스트를 그대로 가져왔다. GT가
  안 바뀌었으니 CER도 그대로.
- CER 3회 반복 측정 결과 **표준편차 0** — 3개 모델 다 디코딩이 결정적(그리디/빔서치,
  temperature 샘플링 없음)이라 반복해도 텍스트가 완전히 동일했다. "반복측정 부족" 한계는
  이번 라운드로 해소됐다(=텍스트 재현성 자체는 확인됨). RTF는 실측 wall-clock이라 반복마다
  약간씩 다르며, 그 표준편차는 표1 각주와 `results.csv`에 그대로 남겼다.

---

## 표 1. 모델별 CLIP1~4 CER — v2(Stage1 필터후) 원 결과 vs 이번 최종 결과

| 클립 | Whisper large-v3-turbo (원 → 최종) | Seamless-M4T v2-Large (원 → 최종) | Wav2Vec2-XLSR-Korean (원 → 최종) |
|---|---|---|---|
| CLIP1 | 0.2148 → **0.2148** (변화없음) | 0.3333 → **0.4296** (+28.9%, 악화) | 0.4370 → **0.4370** (재사용) |
| CLIP2 | 0.5859 → **0.5859** (변화없음) | 0.9604 → **0.6784** (-29.4%, 개선) | 0.6696 → **0.6696** (재사용) |
| CLIP3 | 0.1359 → **0.1359** (변화없음) | 0.2427 → **0.2427** (변화없음) | 0.3010 → **0.3010** (재사용) |
| CLIP4 | 0.0081 → **0.0081** (변화없음) | 0.9187 → **0.5447** (-40.7%, 개선) | 0.2520 → **0.2520** (재사용) |

> "원 결과"는 `results/stage1_vad_only`(Whisper/Wav2Vec2) 및 VAD_비교보고서.md 2.1절
> (Seamless, Fix 적용 전 Stage1 값). Whisper/Wav2Vec2는 이번 라운드에 조건 변경이 없어
> 원=최종. CER 3회 반복 표준편차는 세 모델 전부 0.0000(완전 재현). RTF(3회 평균±표준편차):
> Whisper turbo 0.042~0.052(±0.001~0.008), Seamless 0.21~0.30(±0.003~0.035),
> Wav2Vec2 0.006 내외(stage1 재사용, 이번 라운드 재측정 없음).

---

## 표 2. 모델별 적용 스택 요약

| 모델 | VAD | 디코딩파라미터 | initial_prompt |
|---|---|---|---|
| Whisper large-v3-turbo | 적용 (Stage1 VAD-trim 그대로) | 스킵 (Stage2에서 0% 효과 확인, 재검증 불필요) | 스킵 (도메인 용어집 미작성 — 이번 라운드는 initial_prompt 없이 진행, 후속 작업으로 분리) |
| Seamless-M4T-v2-Large | 적용 (Stage1 VAD-trim 그대로, 재트리밍 금지 — CLIP2에서 A+B 병행 시 역효과/청크경계 밀림 확인됨) | 적용 (num_beams=5, repetition_penalty=1.3 — 기본값으로 승격, 필수) | 해당없음 (모델이 미지원) |
| Wav2Vec2-XLSR-Korean | 해당없음 (v2 원본 그대로, 재전사 없음) | 해당없음 (대응 파라미터 없음) | 해당없음 (모델이 미지원) |

---

## 표 3. 파일별 최종 승자 (v1 룰: RTF≤0.10 통과 후 CER 최소)

| 파일 | RTF≤0.10 통과 모델 (RTF) | 승자 | CER |
|---|---|---|---|
| CLIP1 | Whisper turbo(0.0519), Wav2Vec2(0.0058) — Seamless 탈락(0.2969) | **Whisper large-v3-turbo** | 0.2148 |
| CLIP2 | Whisper turbo(0.0422), Wav2Vec2(0.0063) — Seamless 탈락(0.2463) | **Whisper large-v3-turbo** | 0.5859 |
| CLIP3 | Whisper turbo(0.0524), Wav2Vec2(0.0059) — Seamless 탈락(0.2947) | **Whisper large-v3-turbo** | 0.1359 |
| CLIP4 | Whisper turbo(0.0504), Wav2Vec2(0.0058) — Seamless 탈락(0.2081) | **Whisper large-v3-turbo** | 0.0081 |

**4개 파일 전부 Whisper large-v3-turbo 승리.** Seamless는 Fix 적용 후 CLIP2·CLIP4의 CER
자체는 Whisper turbo보다 여전히 나쁘고(0.6784/0.5447 vs 0.5859/0.0081), 설령 CER이 더
좋았더라도 RTF 게이트에서 이미 탈락이라 승자가 될 수 없는 구조. beam=5가 CER은 개선해도
RTF를 함께 깎아먹어 v1 선정룰상 실질적으로 배포 후보에서 배제된다는 점이 이번 라운드의
핵심 결론.

---

## 부록 — 산출물

| 경로 | 내용 |
|---|---|
| `scripts/run_final_comparison.py` | 이번 라운드 실행 스크립트 (3모델×4클립×3반복) |
| `results/final_comparison/results.csv` | 모델×클립별 CER/WER/RTF 평균·표준편차, 오류 분해 |
| `results/final_comparison/transcripts/raw/{모델}/{CLIP}.txt` | 환각필터 적용 전 원본 전사 (3회 중 마지막 1회분) |
| `results/final_comparison/transcripts/filtered/{모델}/{CLIP}.txt` | 환각필터 적용 후 최종 전사 |
| `results/final_comparison/transcripts/diff/{모델}/{CLIP}.txt` | GT 대비 jiwer 단어 정렬 기반 diff ([유지]/[치환]/[삭제]/[삽입]) |

## 남은 과제

- Whisper용 도메인 용어집(initial_prompt)이 아직 없음 — 작성 후 재검증 필요(이번 라운드는
  용어집 없이 진행).
- Seamless의 CLIP1 악화 원인 미분석 — CLIP2/4와 달리 CLIP1은 jump-cut이 적은 편인데도
  beam=5/repetition_penalty=1.3이 역효과를 낸 이유를 다음 잔차분석에서 다뤄야 함.
- GT를 축어 전사로 교체하는 작업은 이번 라운드에 반영되지 않음(신규 대본 미확보) — 확보되면
  재계산 필요.
