# 융합 LLM 선정 — 정량 집계 (자동)

- precision: **nf4** (전 모델 동일) / 윈도우 15개(A6/B5/C4) × 본 실행 3회
- ★ 통제실험이 아니라 **실사용 비교**: 프롬프트=STEP5 현행 동결본, 입력=실전 산출물.
- 최종 판정 없음 — 정성(블라인드) 채점 합산 전까지 보류.

- 환경: transformers 5.12.1 / torch 2.12.1+cu130 / CUDA 13.0 / Quadro RTX 6000

| 모델 | latency(s) ±σ | tok/s | VRAM 정적(GiB) | VRAM peak | 이행률 | 충실도 | 실행오류 |
|---|---|---|---|---|---|---|---|
| gemma3_12b (gemma-3-12b-it) | 77.19±9.85 | 3.1 | {'0': 3.17, '1': 4.09} | 7.39 | 100.0% | 37.8% | 0 |

## 케이스별 분해 (이행률% / 충실도%)

| 모델 | A(융합) | B(행동만·침묵) | C(발화만) |
|---|---|---|---|
| gemma3_12b | 100.0 / 88.9 | 100.0 / 6.7 | 100.0 / 0.0 |

## 환각/위반 플래그 유형별 건수

- **gemma3_12b**: {'keyword_grounding<0.5': 28}

## 플래그 상세(정성 재확인 대상)

### gemma3_12b
- run1 CLIP1_W01 (A): keyword_grounding<0.5(0.40)
- run1 CLIP1_W06 (B): keyword_grounding<0.5(0.48)
- run1 CLIP1_W09 (B): keyword_grounding<0.5(0.38)
- run1 CLIP3_W02 (B): keyword_grounding<0.5(0.36)
- run1 CLIP3_W03 (B): keyword_grounding<0.5(0.45)
- run1 CLIP4_W01 (B): keyword_grounding<0.5(0.33)
- run1 CLIP3_W01 (C): keyword_grounding<0.5(0.10)
- run1 CLIP3_W05 (C): keyword_grounding<0.5(0.20)
- run1 CLIP4_W03 (C): keyword_grounding<0.5(0.12)
- run1 CLIP4_W09 (C): keyword_grounding<0.5(0.10)
- run2 CLIP1_W01 (A): keyword_grounding<0.5(0.40)
- run2 CLIP1_W06 (B): keyword_grounding<0.5(0.43)
- run2 CLIP1_W09 (B): keyword_grounding<0.5(0.38)
- run2 CLIP3_W02 (B): keyword_grounding<0.5(0.38)
- run2 CLIP3_W03 (B): keyword_grounding<0.5(0.48)
- run2 CLIP4_W01 (B): keyword_grounding<0.5(0.20)
- run2 CLIP3_W01 (C): keyword_grounding<0.5(0.08)
- run2 CLIP3_W05 (C): keyword_grounding<0.5(0.23)
- run2 CLIP4_W03 (C): keyword_grounding<0.5(0.12)
- run2 CLIP4_W09 (C): keyword_grounding<0.5(0.08)
- run3 CLIP1_W09 (B): keyword_grounding<0.5(0.40)
- run3 CLIP3_W02 (B): keyword_grounding<0.5(0.41)
- run3 CLIP3_W03 (B): keyword_grounding<0.5(0.46)
- run3 CLIP4_W01 (B): keyword_grounding<0.5(0.30)
- run3 CLIP3_W01 (C): keyword_grounding<0.5(0.11)
- run3 CLIP3_W05 (C): keyword_grounding<0.5(0.23)
- run3 CLIP4_W03 (C): keyword_grounding<0.5(0.11)
- run3 CLIP4_W09 (C): keyword_grounding<0.5(0.07)
