# VAD 효과 3-arm 비교 — Whisper large-v3-turbo 고정

> `06_VAD재검증_v2/results/vad_effect_whisper/` — 최종 확정 STT(Whisper large-v3-turbo,
> 디코딩파라미터/initial_prompt 스킵)를 고정하고 VAD 축만 3가지로 비교했다.
> ①→② = VAD 자체의 순수 효과, ②→③ = VAD threshold 민감도(지금까지 미검증). 날짜: 2026-07-02

---

## 표. 클립별 CER 3-arm 비교

| 클립 | ① No-VAD | ② VAD-default(0.5) | ③ VAD-lowthresh(0.2) | ①→② | ②→③ |
|---|---|---|---|---|---|
| CLIP1 | 0.5630 | **0.2148** | 0.2000 | -61.8% | -6.9% |
| CLIP2 | 1.0925 | **0.5859** | 0.5903 | -46.4% | +0.8% |
| CLIP3 | 0.3107 | **0.1359** | 0.1262 | -56.3% | -7.1% |
| CLIP4 | 0.1301 | **0.0081** | 0.0081 | -93.8% | 0.0% |
| **평균** | 0.5241 | **0.2362** | 0.2312 | **-54.9%** | **-2.1%** |

RTF (참고): ② VAD-default 0.042~0.052, ③ VAD-lowthresh 0.047~0.049 — 둘 다 v1 기준(≤0.10)
통과, 서로 차이 없음. ① No-VAD는 이번 라운드에서 재전사하지 않아 RTF 미측정(01_STT선정 v2
당시 원본 오디오 길이가 CLIP당 111~229초로 VAD 후(14~30초)보다 훨씬 길어 절대 처리시간은
더 걸렸을 것으로 추정되나, RTF 수치 자체는 이번 산출물에 없음).

---

## 해석

1. **①→② (VAD 자체 효과, 재확인)**: 평균 CER -54.9% — VAD_비교보고서.md에서 이미 확인된
   값과 동일(재전사 없이 기존 결과 재사용). Whisper 계열은 무음/침묵 구간에서 상투구를
   지어내는 환각이 주된 오류 원인이었는데, VAD로 그 구간 자체를 없애니 지어낼 대상이
   없어져서 개선폭이 크다.
2. **②→③ (VAD threshold 민감도, 신규)**: 평균 -2.1%, **클립별로도 절대값이 다 ±0.1%p
   이내**(CLIP1 -1.48%p, CLIP2 +0.44%p, CLIP3 -0.97%p, CLIP4 0%p) — **Whisper는 VAD
   threshold를 0.5→0.2로 낮춰도 거의 차이가 없다.** Seamless가 같은 축(VAD_비교보고서.md
   3절, stage_seamless_fix)에서 jump-cut에 취약해 크게 흔들렸던 것과 대조적 — Whisper의
   상투구 환각은 "무음이 얼마나 완벽히 제거됐는가"보다는 "무음 구간이 존재하는가 자체"에
   더 좌우되는 것으로 보인다.
3. VAD_비교보고서.md 4절 한계("VAD 파라미터는 튜닝하지 않았다")가 Whisper 기준으로는 이번
   실험으로 해소됐다 — **threshold는 0.5(기본값) 유지가 맞고, 더 낮출 이유가 없다**(정확도
   차이가 없는데 트리밍 후 speech_ratio만 살짝 늘어 처리량이 소폭 늘어남: CLIP1
   8.2%→8.75%, CLIP3 12.5%→13.5%, CLIP4 7.0%→7.5%).

**결론**: Whisper large-v3-turbo에게 VAD는 "켜는가/안 켜는가"가 압도적으로 중요하고
(-54.9%), 켜고 나서 threshold를 얼마로 잡는지는 거의 안 중요하다(-2.1%, 클립별 부호도
갈림 — 노이즈 수준). 지금 채택 중인 VAD-default(threshold=0.5) 설정을 바꿀 근거 없음.

---

## 부록 — 산출물

| 경로 | 내용 |
|---|---|
| `scripts/run_vad_effect_whisper.py` | 이번 3-arm 실행 스크립트 |
| `data/vad_trimmed_lowthresh/CLIP{1,3,4}_vad.wav` | 신규 생성 (threshold=0.2), CLIP2는 stage_seamless_fix에서 이미 생성된 것 재사용 |
| `results/vad_effect_whisper/results.csv` | 조건×클립별 CER/WER/RTF |
| `results/vad_effect_whisper/transcripts/{raw,filtered,diff}/{no_vad,vad_default,vad_lowthresh}/{CLIP}.txt` | 조건별 전사 전체 (①②는 기존 결과 재사용, ③만 신규 전사) |
