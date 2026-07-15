# 03_LLM선정 — 융합 LLM 선정 실험 (2026-07-09)

시간 윈도우로 묶인 STT 발화+VLM 행동 입력에서 암묵지 후보(JSON)를 뽑는
**STEP5 융합 LLM**을 선정하기 위한 실사용 비교. **통제실험이 아니라 실사용 비교**다
(프롬프트=STEP5 현행 동결본, 입력=실전 파이프라인 산출물, 게이트/재시도 없는 1-shot).

※ 폴더명은 지시서의 "03_LLM선정"을 그대로 따름 — 기존 `03_전처리_모션가이드_vs_균일추출`과
번호가 겹치지만 이름 전체로 구분된다.

## 원천 데이터 (기존 산출물, 재생성 없음)
- STT: `STEP3_전처리/transcripts/<video_id>.json`
- VLM: `STEP4_YOLO_VLM관찰/output/vlm_observations/<video_id>.observations.json`
- 정렬 윈도우(위 둘을 ±4s로 묶은 것): `STEP5_.../output/aligned_windows/<video_id>.windows.json`

## 비교 대상 (전 모델 동일 정밀도 — bench_config.yaml)
| 키 | repo | 비고 |
|---|---|---|
| qwen3vl | Qwen/Qwen3-VL-8B-Instruct | 멀티모달이나 텍스트 입력만 사용(STEP4와 모델 단일화 구도) |
| llama31 | unsloth/Meta-Llama-3.1-8B-Instruct | meta 원본 gated → 공개 미러 |
| gemma3 | unsloth/gemma-3-4b-it | google 원본 gated → 공개 미러. bf16 전제 학습이라 fp16 탈선 관찰 포인트 |

## 파일
- `prepare_windows.py` → `windows_labels.json`(케이스 사전 라벨링 확정본) +
  `frozen_windows.json`(3모델 공유 payload). 15윈도우 = A(행동+발화)6 / B(행동만)5 / C(발화만)4.
- `run_bench.py` — 모델 1개 실행(웜업1+본3회), `results/{model}/{precision}/run{N}/` 저장(덮어쓰기 금지).
- `aggregate.py` → `metrics.json`, `summary.md` (지표 ①latency ②VRAM ③이행률 ④충실도+케이스 분해).
- `make_blind.py` → `blind_eval.md`(익명 채점지) + `mapping_secret.json`(매핑, 채점 전 열람 금지).
- `llmselect.sbatch` — 모델별 순차 제출용(위 주석의 제출 명령 참고).

## 판정 순서(합의)
① 충실도(1차 관문 — 환각 심하면 탈락) → ② 이행률+정성 매칭(본 승부처) → ③ latency/VRAM(타이브레이커).
정량만으로 최종 판정하지 않는다 — 블라인드 정성 채점 합산 후 결정.
