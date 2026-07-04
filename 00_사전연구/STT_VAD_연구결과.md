# STT·VAD 연구 결과 — 06_VAD재검증_v2 종합 (2026-07-02)

> 이 문서는 `06_VAD재검증_v2/`에서 진행한 STT 4-arm 최종비교 + VAD 효과 재검증을 한 곳에
> 모은 것이다. **`종합연구보고서.md`의 ②③(VAD 비권장) 결론과 정면으로 배치**된다 —
> 이유와 관계는 6절 참고. 그 보고서를 대체하는 게 아니라, **다른 데이터셋(무음비율
> 78~93%의 실습형 영상)에 대해서는 정반대 결론이 나온다는 걸 별도로 기록**하는 문서다.

---

## 0. 요약

- **STT 백엔드**: Whisper large-v3-turbo (01_STT선정에서 선정, 이번 4-arm에서도 재확인 —
  v1 선정룰상 CLIP1~4 전부 최종 승자).
- **이 데이터셋에서 VAD는 필수**: Whisper 기준 CER **-54.9%**(0.5241→0.2362). "VAD를
  켜는가/안 켜는가"가 압도적으로 중요하고, threshold를 0.5→0.2로 낮추는 건 거의 무의미
  (-2.1%, 클립별 부호도 갈림).
- **모델별로 VAD를 다루는 방식은 다르다**: Whisper·Wav2Vec2는 VAD 그대로 켜면 끝. Seamless는
  VAD를 그대로 켜되(재트리밍 금지 — 역효과 확인됨) 디코딩 파라미터(num_beams=5,
  repetition_penalty=1.3)로 아키텍처 취약점을 보완해야 함.
- **v1 선정룰(RTF≤0.10 통과 후 CER 최소) 적용 시 CLIP1~4 전부 Whisper large-v3-turbo 승리**
  — Seamless는 CER이 개선돼도 beam=5 때문에 RTF가 0.21~0.30으로 게이트 자체를 못 넘음.

---

## 1. 데이터셋 특성 — 왜 결론이 갈리는가

| 클립 | 원본 길이 | VAD 트리밍 후 | 발화 비율 |
|---|---|---|---|
| CLIP1 | 198.5s | 16.4s | 8.2% |
| CLIP2 | 132.6s | 29.7s | 22.4% |
| CLIP3 | 110.9s | 13.9s | 12.5% |
| CLIP4 | 229.3s | 16.0s | 7.0% |

CLIP1~4는 **무음/비발화 구간이 78~93%**인 실습형 영상(모더보드·슬롯 등 PC 조립 용어가
등장 — 손 움직이며 작업하는 시간이 훨씬 길고 말은 간헐적). `02_VAD효과`/`03_VAD엔진비교`가
쓴 YouTube 낭독·대화 데이터는 high_silence로 분류돼도 무음비율이 ~50% 수준이었다 — 애초에
"침묵이 압도적인 콘텐츠"가 아니었다. **VAD의 손익은 콘텐츠의 무음 비율에 강하게 의존**하는
것으로 보인다: 무음이 조금이면 VAD 오버헤드가 이득을 못 이기고(②③의 결론), 무음이
압도적이면 그 자체가 환각의 원인이라 제거 효과가 오버헤드를 압도한다(이 문서의 결론).

---

## 2. 모델별 최종 확정 스택

| 모델 | VAD | 디코딩파라미터 | initial_prompt |
|---|---|---|---|
| Whisper large-v3-turbo | 적용 (Silero 기본값 threshold=0.5) | 스킵 (no_speech_threshold/temperature 조정 — Stage2에서 0% 효과 확인) | 스킵 (도메인 용어집 미작성, 후속 작업) |
| Seamless-M4T-v2-Large | 적용 (기본값 그대로, **재트리밍 금지** — threshold 낮추면 청크경계 밀림→반복루프 재발) | 적용 (num_beams=5, repetition_penalty=1.3 — 필수, 기본값으로 승격) | 해당없음 (미지원) |
| Wav2Vec2-XLSR-Korean | 적용 (기본값, 튜닝 불필요) | 해당없음 (대응 파라미터 없음) | 해당없음 (미지원) |

---

## 3. 4-arm 최종비교 (모델 간)

### 3.1 CLIP별 CER — v2(Stage1 필터후) 원 결과 vs 최종 결과

| 클립 | Whisper turbo (원→최종) | Seamless (원→최종) | Wav2Vec2 (원→최종) |
|---|---|---|---|
| CLIP1 | 0.2148 → 0.2148 | 0.3333 → **0.4296**(악화) | 0.4370 → 0.4370 |
| CLIP2 | 0.5859 → 0.5859 | 0.9604 → **0.6784**(개선) | 0.6696 → 0.6696 |
| CLIP3 | 0.1359 → 0.1359 | 0.2427 → 0.2427 | 0.3010 → 0.3010 |
| CLIP4 | 0.0081 → 0.0081 | 0.9187 → **0.5447**(개선) | 0.2520 → 0.2520 |

Seamless의 num_beams=5/repetition_penalty=1.3은 CLIP2·4는 크게 개선하지만 **CLIP1은
오히려 악화**시킨다 — "만능 개선"이 아니라 클립(콘텐츠) 의존적이라는 걸 숨기지 않고 기록.

### 3.2 파일별 최종 승자 (v1 룰: RTF≤0.10 통과 후 CER 최소)

| 파일 | RTF≤0.10 통과 | 승자 | CER |
|---|---|---|---|
| CLIP1 | turbo(0.0519), Wav2Vec2(0.0058) — Seamless 탈락(0.2969) | **Whisper turbo** | 0.2148 |
| CLIP2 | turbo(0.0422), Wav2Vec2(0.0063) — Seamless 탈락(0.2463) | **Whisper turbo** | 0.5859 |
| CLIP3 | turbo(0.0524), Wav2Vec2(0.0059) — Seamless 탈락(0.2947) | **Whisper turbo** | 0.1359 |
| CLIP4 | turbo(0.0504), Wav2Vec2(0.0058) — Seamless 탈락(0.2081) | **Whisper turbo** | 0.0081 |

CER 3회 반복 측정 결과 표준편차는 세 모델 다 0.0000 — 디코딩이 결정적(그리디/빔서치,
temperature 샘플링 없음)이라 완전히 재현됨.

> 상세: `06_VAD재검증_v2/results/final_comparison/FINAL_비교.md`,
> 전사 원문/필터후/diff는 `results/final_comparison/transcripts/`.

---

## 4. VAD 사용 정책 — 모델별로 다르게 가는 이유

- **Whisper·Wav2Vec2**: VAD 기본값을 켜는 것만으로 끝. Stage2에서 디코딩파라미터 조정은
  0% 효과였고, threshold를 더 낮춰도(0.2) 거의 차이 없음(5절 참고) — 튜닝할 필요 자체가 없음.
- **Seamless**: VAD 기본 트리밍이 "이어붙은 오디오에 인위적 jump-cut"을 만드는데, 이 모델의
  음성 인코더(청크 어텐션 + 상대적 위치 임베딩)가 여기 취약해 생성을 포기해버림
  (`stage_seamless_fix`, `seamless_누락진단/SEAMLESS_누락원인.md`). VAD threshold를
  낮춰(0.2) "더 자연스럽게" 트리밍하는 것도 시도했지만 CLIP2에서 디코딩 수정과 같이 쓰면
  청크 경계가 오히려 밀려 반복루프가 재발해 악화(0.6784→0.8370). **결론: VAD는 그대로 두고,
  문제는 디코딩(num_beams/repetition_penalty)에서 흡수**하는 게 유일하게 먹힌 조합.

### VAD 조건 정리 (헷갈리기 쉬운 부분)

VAD 세팅 자체는 이 프로젝트 전체에서 **2가지뿐**(default threshold=0.5 / lowthresh=0.2).
`stage_seamless_fix`의 "4개 조건" 표는 이 2가지 VAD × 디코딩(Fix A 유무) 2가지의 **조합**이
4개라는 뜻이지, VAD 세팅이 4종류라는 뜻이 아님(CLIP2 한정 실험).

---

## 5. VAD 효과 재검증 — Whisper 고정, 3-arm

Whisper large-v3-turbo(최종 확정 설정)를 고정하고 VAD 축만 3가지로 비교:

| 클립 | ① No-VAD | ② VAD-default(0.5) | ③ VAD-lowthresh(0.2) | ①→② | ②→③ |
|---|---|---|---|---|---|
| CLIP1 | 0.5630 | 0.2148 | 0.2000 | -61.8% | -6.9% |
| CLIP2 | 1.0925 | 0.5859 | 0.5903 | -46.4% | +0.8% |
| CLIP3 | 0.3107 | 0.1359 | 0.1262 | -56.3% | -7.1% |
| CLIP4 | 0.1301 | 0.0081 | 0.0081 | -93.8% | 0.0% |
| **평균** | 0.5241 | **0.2362** | 0.2312 | **-54.9%** | **-2.1%** |

- **①→② (VAD 자체 효과)**: 압도적 개선(-54.9%) — 무음 구간의 상투구 환각이 사라짐.
- **②→③ (threshold 민감도, 신규 검증)**: 거의 무의미(-2.1%, 클립별 부호도 갈림) —
  Seamless와 달리 Whisper는 VAD threshold에 민감하지 않음. **기본값(0.5) 유지가 맞고
  더 낮출 이유 없음.**

> 상세: `06_VAD재검증_v2/results/vad_effect_whisper/VAD효과_Whisper.md`,
> 전사는 `results/vad_effect_whisper/transcripts/`.

---

## 5.5 프로덕션(STEP3_전처리) 반영 — 병합식 VAD 대신 faster-whisper 내장 vad_filter

5절까지의 VAD는 전부 Silero로 발화 구간을 **잘라 이어붙이는** 방식(`collect_chunks`)이다.
이 방식은 결과 오디오가 원본보다 훨씬 짧아져 Whisper가 내놓는 `seg.start/end`가 원본 영상
타임라인과 어긋난다. `STEP3_전처리/tacit_pipeline/components/aligner.py`(`WindowAligner`)는
발화 타임스탬프를 VLM 액션 타임스탬프와 **±윈도우로 직접 매칭**하므로, 이 어긋남은 그대로
정렬 오류가 된다 — 실제로 `_내일_서버전환_가이드.md`에 "VAD는 병합 부작용 → OFF 유지"로
이미 한 번 기록돼 있었다(암묵지 추출은 "명장이 말은 하고 행동은 안 하거나, 행동만 하거나,
둘 다 하는" 비동기 이벤트를 타임스탬프로 매칭해야 하므로 이 오차가 특히 치명적).

faster-whisper에 내장된 `vad_filter=True`는 내부적으로 Silero를 쓰지만 오디오를 자르지
않고 무음 구간만 스킵하면서 세그먼트 타임스탬프를 **원본 기준으로 그대로 리포트**한다.
원본(비트리밍) CLIP1~4에 직접 적용해 실측:

| 클립 | No-VAD | VAD-병합(5절 방식) | vad_filter=True(내장) | 타임스탬프 |
|---|---|---|---|---|
| CLIP1 | 0.5630 | 0.2148 | **0.1778** | 원본기준유지 (구간 15.7~113.7s / 원본 198.5s) |
| CLIP2 | 1.0925 | 0.5859 | 0.6035 | 원본기준유지 (구간 0.7~114.1s / 원본 132.6s) |
| CLIP3 | 0.3107 | 0.1359 | 0.1359 | 원본기준유지 (구간 0.6~99.6s / 원본 110.9s) |
| CLIP4 | 0.1301 | 0.0081 | 0.0081 | 원본기준유지 (구간 22.4~173.5s / 원본 229.3s) |
| **평균** | 0.5241 | 0.2362 | **0.1813** | — |

CER은 병합 방식과 동급 이상, RTF는 더 낮음(0.009~0.014 vs 0.042~0.052), **세그먼트가
원본 오디오 전체 구간에 걸쳐 분포**(트리밍후처럼 짧은 구간에 쏠리지 않음) — 세 축 다
병합 방식보다 낫거나 같다. **`_내일_서버전환_가이드.md`의 "VAD OFF 유지" 결정을 뒤집고,
`config.yaml`/`config.example.yaml`의 `stt.params`에 `vad_filter: true`,
`vad_parameters.threshold: 0.5`로 반영함**(2026-07-02). `stt_whisper.py`도 이 두 파라미터를
받아 `WhisperModel.transcribe()`에 전달하도록 수정.

> 상세: `06_VAD재검증_v2/results/vad_filter_builtin/results.csv`,
> 전사는 `results/vad_filter_builtin/transcripts/`.

---

## 6. `종합연구보고서.md`와의 관계

`종합연구보고서.md`의 ②`02_VAD효과`·③`03_VAD엔진비교`는 **"VAD 비권장"**으로 결론 났다
(YouTube 낭독·대화, 무음비율 최대 ~50%대 데이터). 이 문서의 결론은 그 반대다(무음비율
78~93%의 실습형 영상 데이터). **두 결론 다 각자의 데이터셋 안에서는 맞다** — 무음 비율이
낮으면 VAD 오버헤드가 손해, 무음 비율이 압도적으로 높으면 VAD의 환각 제거 효과가 오버헤드를
압도한다는 게 지금까지 나온 데이터를 관통하는 설명이다.

**정리하면 VAD 사용 여부는 "STT 백엔드가 뭐냐"보다 "콘텐츠의 무음 비율이 얼마냐"에
더 좌우된다** — 본실험에서 다룰 제철/현장 시나리오 영상의 실제 무음 비율을 먼저 재보고
그 값이 낮은 쪽(②③ 데이터)에 가까운지 높은 쪽(이 문서의 CLIP1~4)에 가까운지로 VAD
채택 여부를 결정해야 한다.

---

## 남은 과제

- Whisper용 도메인 용어집(initial_prompt) 아직 없음 — 작성 후 재검증 필요.
- Seamless의 CLIP1 악화 원인 미분석 — 다음 잔차분석에서 다뤄야 함.
- GT를 축어 전사로 교체하는 작업 미반영(신규 대본 미확보).
- 본실험 영상의 실제 무음 비율 측정 → 이 문서(고무음)와 ②③(저무음) 중 어느 쪽 결론을
  적용할지 결정.
- ~~STEP3_전처리 VAD 반영~~ — 완료(5.5절). Seamless/Wav2Vec2는 STEP3에 아예 안 들어가므로
  이 결정은 Whisper 단독 기준.
- STEP3_전처리는 아직 end-to-end 서버 실행 전(RTX2080 검증만 완료) — `vad_filter=True`가
  실제 영상(짧은 CLIP이 아니라 시나리오 롱폼)에서도 같은 효과인지는 서버 이관 후 재확인 필요.

## 부록 — 산출물

| 경로 | 내용 |
|---|---|
| `06_VAD재검증_v2/results/final_comparison/` | 4-arm 최종비교(모델 간) 결과·전사 |
| `06_VAD재검증_v2/results/vad_effect_whisper/` | VAD 효과 3-arm(Whisper 고정) 결과·전사 |
| `06_VAD재검증_v2/results/vad_filter_builtin/` | faster-whisper 내장 vad_filter 검증(CER+타임스탬프 실측) |
| `06_VAD재검증_v2/results/stage_seamless_fix/`, `seamless_누락진단/` | Seamless 디코딩/VAD 수정 진단 원자료 |
| `06_VAD재검증_v2/results/VAD_비교보고서.md` | VAD 트리밍 통계·초기 필터전후 비교 |
| `STEP3_전처리/tacit_pipeline/components/stt_whisper.py` | vad_filter/vad_parameters 반영(2026-07-02) |
| `STEP3_전처리/config.yaml`, `config.example.yaml` | `stt.params.vad_filter: true` 반영 |
