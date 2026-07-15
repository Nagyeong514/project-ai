# Stage2 — +VAD +디코딩파라미터

> Stage1(+VAD만) 대비 faster-whisper 계열에 `no_speech_threshold=0.8`, `temperature=0.0`
> (get_stt() 팩토리를 우회해 `FasterWhisperRunner(decoding_params=...)`로 직접 주입 —
> 01_STT선정 원본 파일은 수정하지 않음)을 적용했을 때 추가 개선폭.
> Seamless-M4T-v2/Wav2Vec2-XLSR-Korean은 이 두 파라미터에 대응하는 개념이 없어(Whisper
> 전용 디코딩 옵션) Stage1 값을 그대로 이어씀. 날짜: 2026-07-02

---

## 결과: Stage1(+VAD만) → Stage2(+VAD+디코딩파라미터)

| 파일 | Faster-Whisper large-v3 | Whisper large-v3-turbo | Seamless-M4T v2-Large | Wav2Vec2-XLSR-Korean |
|---|---|---|---|---|
| CLIP1 | 0.2222 → 0.2222 | 0.2148 → 0.2148 | 0.3333 (동일, 파라미터 없음) | 0.4370 (동일, 파라미터 없음) |
| CLIP2 | 0.5991 → 0.5991 | 0.5859 → 0.5859 | 0.9604 (동일) | 0.6696 (동일) |
| CLIP3 | 0.1456 → 0.1456 | 0.1359 → 0.1359 | 0.2427 (동일) | 0.3010 (동일) |
| CLIP4 | 0.0081 → 0.0081 | 0.0081 → 0.0081 | 0.9187 (동일) | 0.2520 (동일) |

**모델별 평균 CER 변화(Stage1 → Stage2)**

| 모델 | Stage1 CER | Stage2 CER | 변화 |
|---|---|---|---|
| Faster-Whisper large-v3 | 0.2438 | 0.2438 | **0.0%** |
| Whisper large-v3-turbo | 0.2362 | 0.2362 | **0.0%** |
| Seamless-M4T v2-Large | 0.6138 | 0.6138 | (해당없음, 값 이어씀) |
| Wav2Vec2-XLSR-Korean | 0.4149 | 0.4149 | (해당없음, 값 이어씀) |

## 왜 개선폭이 0%인가

원문(raw, 필터 적용 전) 전사를 Stage1·Stage2 사이에 직접 diff해봤다 — **4파일 × 2모델
전부 바이트 단위로 100% 동일**했다. 즉 파라미터를 바꿔도 모델이 실제로 다르게 행동하지
않았다.

원인 추정: `no_speech_threshold`는 "이 구간이 무음/비발화인지" 판정하는 임계값인데, VAD가
이미 무음 구간을 전부 잘라내 버려서 이 임계값이 작동할 대상 자체가 남아있지 않다.
`temperature=0.0`도 마찬가지로 — faster-whisper의 온도 폴백(기본값이 `[0.0, 0.2, 0.4, 0.6,
0.8, 1.0]` 리스트)은 **1차 디코딩(temperature=0.0)이 compression_ratio/logprob 기준을
못 넘길 때만** 다음 온도로 재시도하는 구조다. Stage1(파라미터 오버라이드 없음, 라이브러리
기본값)에서도 VAD 덕분에 애초에 재시도가 한 번도 발동하지 않았던 것으로 보인다(1차
디코딩이 이미 통과) — 그래서 명시적으로 `temperature=0.0`을 강제해도 달라질 게 없었다.

**결론**: 이 두 파라미터가 원래 잡으려던 문제(무음 구간에서의 환각·불필요한 재시도)는
**VAD 트리밍이 이미 해결**했다. 파라미터 튜닝의 "몫"이 아니라 VAD의 몫이었다는 뜻 —
어블레이션 순서상 중요한 발견(다음 단계에서 initial_prompt·LLM필터의 효과를 볼 때, 이미
VAD로 대부분 해결된 뒤의 "나머지" 오차에 대한 효과로 해석해야 함).

---

## 산출물

| 파일 | 내용 |
|---|---|
| `results.csv` | Stage2 원 채점 결과 |
| `transcripts/raw/{model}/{clip}.txt` | 필터 적용 전 원문 전사 |
| `transcripts/filtered/{model}/{clip}.txt` | 환각 필터 적용 후 전사 |
| (`../stage1_vad_only/`) | Stage1(+VAD만) 결과·전사 — 비교 기준 |
