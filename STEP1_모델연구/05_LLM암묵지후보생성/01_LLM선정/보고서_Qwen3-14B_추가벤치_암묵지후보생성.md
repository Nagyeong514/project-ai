# Qwen3-14B 추가 벤치 보고서 — 3단계 암묵지 후보 생성 (p.24 실험의 동일 조건 재실행)

- 실행: 2026-07-11, SLURM 잡 **2595** (n5, RTX6000×2), 로그 `run_logs/llmselect_llmsel_qwen3_14b_2595.log`
- 대상 모델: **Qwen/Qwen3-14B** (STEP5 융합 실전 채택 모델) — 단독 실행. 기존 3종(Qwen3-VL-8B / Llama-3.1-8B / Gemma-3-4B) 결과는 `results/`·`metrics.json`에 동결 보존, 본 결과는 `results/qwen3_14b/nf4/` + `metrics_qwen3_14b.json`.
- 조건 동일성(기존 실험과 완전 동일 — `bench_config.yaml` 통제변수):
  - 정밀도 nf4, transformers+sdpa, device_map=auto, temperature 0.2 / top_p 1.0 / max_new_tokens 2048
  - 프롬프트: STEP5 `llm_fusion_prompt.py` 2026-07-09 동결본
  - 입력: 실전 산출물 동결 윈도우 15개(`frozen_windows.json`) × 3회 반복 + 웜업 1회 = 집계 45레코드 (장표의 "18회"는 이 중 **케이스 A(행동+발화) 6윈도우 × 3회 = 18레코드** 기준)
  - Qwen3-14B 특이 처리 1건: hybrid thinking 차단(`run_bench.py`에 `no_think` 옵션 추가 — STEP5 실전 `llm_fusion._load`와 동일한 enable_thinking=False 주입. 기존 3종 코드 경로 불변)

## 1. 정량 결과 (45레코드 자동 집계, `aggregate.py` 동일 스크립트)

| 지표 | **Qwen3-14B (신규)** | Qwen3-VL-8B | Llama-3.1-8B | Gemma-3-4B |
|---|---|---|---|---|
| 평균 Latency | **57.14초 (±7.47)** | 57.79초 (±6.73) | 31.32초 (±3.29) | 49.74초 |
| Throughput | **4.1 tok/s** | 4.9 tok/s | 6.0 tok/s | 4.6 tok/s |
| VRAM Peak (생성 중 alloc) | **12.94 GiB** | 8.43 GiB | 7.39 GiB | 4.18 GiB |
| 로드 시간 / 정적 alloc | 15.3초 / 9.25 GiB(2장 합) | 50.4초 / 5.94 GiB | 39.8초 / 5.31 GiB | — |
| 실행 에러 | **0건 / 45** | 0 | 0 | 0 |
| 지시 이행률(스키마·형식) | **100% (45/45)** | 100% | 100% | 100% |
| 충실도(키워드 휴리스틱)* | 24.4% (케이스A 55.6%) | 26.7% (A 61.1%) | 51.1% (A 83.3%) | 17.8% (A 44.4%) |

\* 충실도는 kiwi 명사 겹침 기반 자동 근사치(원 실험과 동일 캐비어트) — 서술이 풍부할수록 불리하게 나오는 지표라 순위 해석 주의. Llama가 높은 것은 원 실험 정성평가에서 확인된 "발화를 그대로 반복"하는 성향의 부수효과다(입력 단어 재사용률이 높음).

요약: **Qwen3-14B는 8B급인 Qwen3-VL과 같은 Latency대(57초)를 유지하면서 45건 전건 스키마 통과.** VRAM peak 12.94 GiB로 기존 3종보다 크지만 RTX6000 2장(48GB) 환경에서 여유 충분. 14B로 커졌는데 Latency가 늘지 않은 것은 로드가 빠르고(15.3초) 생성 토큰이 콤팩트(평균 ~240tok)했기 때문.

## 2. 정성 결과 — 출력 문장 (실제 산출물 발췌)

케이스 A(행동+발화) 대표 윈도우 CLIP2_W01(LED 진단 발화)의 run1 실제 출력:

```json
"situation": "정비공이 컴퓨터의 전원 버튼을 눌러 LED 반응을 확인하고, 케이스 상단 덮개를 열어 내부를 점검한다.",
"tacit_insight": "전원 버튼을 누르고 LED 반응을 관찰하여 시스템 상태를 확인한다.",
"reasoning": "정비공이 '황색 4번에 백색 5번이면은 램 불량이네요'라고 말함 — LED 패턴을 통해 문제 원인을 추정하고, 연결 상태를 먼저 확인해야 한다고 강조한다.",
"reasoning_origin": "utterance"
```

- **행동(전원 버튼·LED 관찰)과 발화 근거(패턴→램 불량)를 연결해 서술** — 원 실험에서 Qwen3-VL이 채택된 바로 그 기준(Llama: 발화 단순 반복 / Gemma: 발화에 없는 정보 창작)을 Qwen3-14B도 충족한다. 발화 원문을 따옴표 인용하고 utterance 태깅하는 접지 습관은 STEP5 실전(접지 게이트 통과 이력)과 일치.
- 45건 중 금지 필드 출력(diagnostic_steps 등)·탈선 문자·한국어 이탈 **0건** (`rule_flags` 전무).

## 3. 결론

STEP5 실전 채택 모델 Qwen3-14B를 원 선정 벤치에 태운 결과, **후보 생성 역할에서 기존 채택 기준(스키마 100% + 행동·근거 연결 서술)을 동일하게 충족**했다. 비용은 VRAM peak +4.5 GiB. 원 실험의 최종 후보였던 Qwen3-VL 대비 Latency 동급, 스키마 안정성 동급 — "브랜드벤치(06)에서 Qwen3-14B가 Pass 2 응집까지 5/5로 이긴 것"과 합치면 융합 역할의 Qwen3-14B 채택을 이 벤치도 지지한다.

- 원자료: `results/qwen3_14b/nf4/run1~3/outputs.jsonl`, 집계 `metrics_qwen3_14b.json`, `summary_qwen3_14b.md`
- 주의: 기존 3종과 실행 시점이 다르다(2026-07-09 vs 07-11). 통제변수는 동일하나 동시 실행 비교는 아니며, temperature 0.2 샘플링·시드 미고정이라 문구 수준 재현은 안 된다(수치는 3회 평균).
