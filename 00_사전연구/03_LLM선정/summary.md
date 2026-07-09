# 융합 LLM 선정 — 정량 집계 (자동)

- precision: **nf4** (전 모델 동일) / 윈도우 15개(A6/B5/C4) × 본 실행 3회
- ★ 통제실험이 아니라 **실사용 비교**: 프롬프트=STEP5 현행 동결본, 입력=실전 산출물.
- 최종 판정 없음 — 정성(블라인드) 채점 합산 전까지 보류.

- 환경: transformers 5.12.1 / torch 2.12.1+cu130 / CUDA 13.0 / Quadro RTX 6000

| 모델 | latency(s) ±σ | tok/s | VRAM 정적(GiB) | VRAM peak | 이행률 | 충실도 | 실행오류 |
|---|---|---|---|---|---|---|---|
| qwen3vl (Qwen3-VL-8B-Instruct) | 57.79±6.73 | 4.9 | {'0': 1.54, '1': 4.4} | 8.43 | 100.0% | 26.7% | 0 |
| llama31 (Meta-Llama-3.1-8B-Instruct) | 31.32±3.29 | 6.0 | {'0': 1.4, '1': 3.91} | 7.39 | 100.0% | 51.1% | 0 |
| gemma3 (gemma-3-4b-it) | 49.74±5.06 | 4.6 | {'0': 0.13, '1': 2.88} | 4.18 | 100.0% | 17.8% | 0 |

## 케이스별 분해 (이행률% / 충실도%)

| 모델 | A(융합) | B(행동만·침묵) | C(발화만) |
|---|---|---|---|
| qwen3vl | 100.0 / 61.1 | 100.0 / 6.7 | 100.0 / 0.0 |
| llama31 | 100.0 / 83.3 | 100.0 / 53.3 | 100.0 / 0.0 |
| gemma3 | 100.0 / 44.4 | 100.0 / 0.0 | 100.0 / 0.0 |

## 환각/위반 플래그 유형별 건수

- **qwen3vl**: {'keyword_grounding<0.5': 33}
- **llama31**: {'keyword_grounding<0.5': 21, 'case_B_utterance_origin': 2}
- **gemma3**: {'keyword_grounding<0.5': 36, 'case_B_utterance_origin': 14}

## 플래그 상세(정성 재확인 대상)

### qwen3vl
- run1 CLIP1_W01 (A): keyword_grounding<0.5(0.38)
- run1 CLIP1_W02 (A): keyword_grounding<0.5(0.46)
- run1 CLIP2_W01 (A): keyword_grounding<0.5(0.31)
- run1 CLIP1_W06 (B): keyword_grounding<0.5(0.32)
- run1 CLIP3_W02 (B): keyword_grounding<0.5(0.33)
- run1 CLIP3_W03 (B): keyword_grounding<0.5(0.39)
- run1 CLIP4_W01 (B): keyword_grounding<0.5(0.29)
- run1 CLIP3_W01 (C): keyword_grounding<0.5(0.09)
- run1 CLIP3_W05 (C): keyword_grounding<0.5(0.14)
- run1 CLIP4_W03 (C): keyword_grounding<0.5(0.08)
- run1 CLIP4_W09 (C): keyword_grounding<0.5(0.08)
- run2 CLIP1_W01 (A): keyword_grounding<0.5(0.39)
- run2 CLIP2_W01 (A): keyword_grounding<0.5(0.48)
- run2 CLIP3_W04 (A): keyword_grounding<0.5(0.45)
- run2 CLIP1_W06 (B): keyword_grounding<0.5(0.35)
- run2 CLIP1_W09 (B): keyword_grounding<0.5(0.41)
- run2 CLIP3_W02 (B): keyword_grounding<0.5(0.28)
- run2 CLIP3_W03 (B): keyword_grounding<0.5(0.37)
- run2 CLIP4_W01 (B): keyword_grounding<0.5(0.19)
- run2 CLIP3_W01 (C): keyword_grounding<0.5(0.08)
- run2 CLIP3_W05 (C): keyword_grounding<0.5(0.14)
- run2 CLIP4_W03 (C): keyword_grounding<0.5(0.07)
- run2 CLIP4_W09 (C): keyword_grounding<0.5(0.06)
- run3 CLIP1_W01 (A): keyword_grounding<0.5(0.24)
- run3 CLIP1_W06 (B): keyword_grounding<0.5(0.33)
- run3 CLIP1_W09 (B): keyword_grounding<0.5(0.45)
- run3 CLIP3_W02 (B): keyword_grounding<0.5(0.29)
- run3 CLIP3_W03 (B): keyword_grounding<0.5(0.34)
- run3 CLIP4_W01 (B): keyword_grounding<0.5(0.19)
- run3 CLIP3_W01 (C): keyword_grounding<0.5(0.08)
- run3 CLIP3_W05 (C): keyword_grounding<0.5(0.14)
- run3 CLIP4_W03 (C): keyword_grounding<0.5(0.08)
- run3 CLIP4_W09 (C): keyword_grounding<0.5(0.06)

### llama31
- run1 CLIP3_W02 (B): keyword_grounding<0.5(0.33)
- run1 CLIP3_W01 (C): keyword_grounding<0.5(0.20)
- run1 CLIP3_W05 (C): keyword_grounding<0.5(0.38)
- run1 CLIP4_W03 (C): keyword_grounding<0.5(0.12)
- run1 CLIP4_W09 (C): keyword_grounding<0.5(0.20)
- run2 CLIP3_W04 (A): keyword_grounding<0.5(0.33)
- run2 CLIP4_W05 (A): keyword_grounding<0.5(0.46)
- run2 CLIP1_W06 (B): keyword_grounding<0.5(0.42)
- run2 CLIP3_W03 (B): keyword_grounding<0.5(0.25)
- run2 CLIP4_W01 (B): keyword_grounding<0.5(0.40)
- run2 CLIP3_W01 (C): keyword_grounding<0.5(0.25)
- run2 CLIP3_W05 (C): keyword_grounding<0.5(0.27)
- run2 CLIP4_W03 (C): keyword_grounding<0.5(0.14)
- run2 CLIP4_W09 (C): keyword_grounding<0.5(0.14)
- run3 CLIP3_W04 (A): keyword_grounding<0.5(0.38)
- run3 CLIP1_W06 (B): keyword_grounding<0.5(0.36)
- run3 CLIP3_W02 (B): keyword_grounding<0.5(0.27), case_B_utterance_origin
- run3 CLIP4_W01 (B): case_B_utterance_origin
- run3 CLIP3_W01 (C): keyword_grounding<0.5(0.20)
- run3 CLIP3_W05 (C): keyword_grounding<0.5(0.43)
- run3 CLIP4_W03 (C): keyword_grounding<0.5(0.12)
- run3 CLIP4_W09 (C): keyword_grounding<0.5(0.20)

### gemma3
- run1 CLIP1_W01 (A): keyword_grounding<0.5(0.29)
- run1 CLIP1_W02 (A): keyword_grounding<0.5(0.18)
- run1 CLIP2_W02 (A): keyword_grounding<0.5(0.42)
- run1 CLIP3_W04 (A): keyword_grounding<0.5(0.47)
- run1 CLIP1_W06 (B): keyword_grounding<0.5(0.40), case_B_utterance_origin
- run1 CLIP1_W09 (B): keyword_grounding<0.5(0.35), case_B_utterance_origin
- run1 CLIP3_W02 (B): keyword_grounding<0.5(0.20), case_B_utterance_origin
- run1 CLIP3_W03 (B): keyword_grounding<0.5(0.07), case_B_utterance_origin
- run1 CLIP4_W01 (B): case_B_utterance_origin
- run1 CLIP3_W01 (C): keyword_grounding<0.5(0.00)
- run1 CLIP3_W05 (C): keyword_grounding<0.5(0.16)
- run1 CLIP4_W03 (C): keyword_grounding<0.5(0.00)
- run1 CLIP4_W09 (C): keyword_grounding<0.5(0.08)
- run2 CLIP1_W01 (A): keyword_grounding<0.5(0.33)
- run2 CLIP1_W02 (A): keyword_grounding<0.5(0.18)
- run2 CLIP2_W02 (A): keyword_grounding<0.5(0.40)
- run2 CLIP1_W06 (B): keyword_grounding<0.5(0.47), case_B_utterance_origin
- run2 CLIP1_W09 (B): keyword_grounding<0.5(0.38), case_B_utterance_origin
- run2 CLIP3_W02 (B): keyword_grounding<0.5(0.25), case_B_utterance_origin
- run2 CLIP3_W03 (B): keyword_grounding<0.5(0.07), case_B_utterance_origin
- run2 CLIP4_W01 (B): keyword_grounding<0.5(0.38)
- run2 CLIP3_W01 (C): keyword_grounding<0.5(0.00)
- run2 CLIP3_W05 (C): keyword_grounding<0.5(0.18)
- run2 CLIP4_W03 (C): keyword_grounding<0.5(0.00)
- run2 CLIP4_W09 (C): keyword_grounding<0.5(0.07)
- run3 CLIP1_W01 (A): keyword_grounding<0.5(0.27)
- run3 CLIP1_W02 (A): keyword_grounding<0.5(0.18)
- run3 CLIP3_W04 (A): keyword_grounding<0.5(0.44)
- run3 CLIP1_W06 (B): keyword_grounding<0.5(0.36), case_B_utterance_origin
- run3 CLIP1_W09 (B): keyword_grounding<0.5(0.33), case_B_utterance_origin
- run3 CLIP3_W02 (B): keyword_grounding<0.5(0.12), case_B_utterance_origin
- run3 CLIP3_W03 (B): keyword_grounding<0.5(0.07), case_B_utterance_origin
- run3 CLIP4_W01 (B): keyword_grounding<0.5(0.47), case_B_utterance_origin
- run3 CLIP3_W01 (C): keyword_grounding<0.5(0.00)
- run3 CLIP3_W05 (C): keyword_grounding<0.5(0.16)
- run3 CLIP4_W03 (C): keyword_grounding<0.5(0.00)
- run3 CLIP4_W09 (C): keyword_grounding<0.5(0.06)
