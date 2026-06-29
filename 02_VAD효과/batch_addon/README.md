# batch_addon — 배치 추론 조건 추가 실험

기존 비배치 3-arm 실험(`A` / `A′` / `B`, `experiments/` + `scripts/run_experiment.py`)은
**완료분이라 수정하지 않는다.** 이 폴더는 거기에 **배치 추론(BatchedInferencePipeline) 조건
2개를 추가**해 같은 세션에서 새로 측정하기 위한 독립 애드온이다.

## 추가 조건과 비교 축

| 조건 | 자르는 기준 | 추론 방식 | 비교 |
|---|---|---|---|
| `A_prime` | 안 자름(전체 1회) | 순차 `model.transcribe` (기존 재사용) | — |
| `A_prime_batch` | **고정 30초 연속 윈도우**(전체 100% 커버, VAD 아님) | `BatchedInferencePipeline` | `A′ ↔ A′_batch` = 배치 효과 |
| `B` | 외부 Silero VAD 청크(≤30s) | 순차 `model.transcribe` (기존 재사용) | — |
| `B_batch` | **B와 동일한** Silero VAD 청크를 `clip_timestamps`로 주입 | `BatchedInferencePipeline` | `B ↔ B_batch` = 배치 효과 |

`clip_timestamps`를 직접 주입해 배치 파이프라인의 **내장 VAD를 우회**한다. 그래서 비배치
대응 조건과 "자르는 기준"이 동일하고, **차이가 오직 배치 여부에서만** 발생한다.

## 통제 조건 (배치 여부 외 동일하게 맞춘 것)

- **`condition_on_previous_text`**: 배치 파이프라인 내부에서 `False`로 하드코딩됨
  (`faster_whisper/transcribe.py:547`, 인자로 줘도 무시). 비배치 `A′`/`B`도 이미 `False`
  (`DECODING_PARAMS_UNIFIED`) → **전 조건 동일**. 배치 비교에 혼입되지 않는다.
- **디코딩 파라미터**(beam_size, temperature, no_speech_threshold, log_prob_threshold,
  compression_ratio_threshold): 비배치 통일값(`DECODING_PARAMS_UNIFIED`)을 그대로 배치에 전달.
- **batch_size**: 8GB VRAM OOM 대비 기본 8 → OOM 시 `torch.cuda.empty_cache()` 후 4로
  자동 폴백. 사용된 값과 폴백 여부를 결과 CSV(`batch_size`, `oom_fallback`)에 기록.
- **모델**: 비배치 runner와 배치 pipeline이 **동일한 로드된 `WhisperModel`을 공유**(중복 로드 없음).

## 측정 지표 — WER / CER / 할루시네이션per시간 / RTF (warmup 1 + 측정 3회)

기존 평가 코드(`evaluation/wer_cer.py`, `evaluation/hallucination.py`)와 GT, 10개 파일을
그대로 재사용한다.

### ⚠️ `timestamp_drift` 지표를 제외한 이유 (중요 — 나중에 설명 근거)

배치 파이프라인은 `without_timestamps`가 **디폴트 `True`**이고, 비배치 `model.transcribe`는
**디폴트 `False`**다. 이 차이는 **배치 추론의 본질적 동작**이라 제거할 수 없다:

- 비배치는 각 윈도우 내부에서 **세그먼트 단위 타임스탬프**를 생성한다.
- 배치는 텍스트 토큰만 샘플링하고 타임스탬프는 **청크 경계 단위로만** 부여된다(더 거칠다).

따라서 **세그먼트 단위 `timestamp_drift`는 배치 ↔ 비배치 간 같은 척도로 비교할 수 없다.**
억지로 측정하면 "배치라서 나쁜 것"인지 "타임스탬프 입자가 달라서 그렇게 보이는 것"인지
구분이 안 된다. 그래서 이 애드온은 **`timestamp_drift`를 측정·기록에서 제외**하고,
배치 여부와 무관하게 공정 비교가 되는 **WER / CER / 할루시네이션 / RTF 4개 지표만** 쓴다.
(WER/CER/할루시/RTF는 `without_timestamps` 차이의 영향이 미미하다.)

## 결과 CSV 컬럼 (`results_batch.csv`)

`file_id, silence_ratio, condition, batched, batch_size, oom_fallback, wer, cer,
hallucination_per_hour, rtf_mean, rtf_std, vad_time_s, n_chunks`

기존 `results_10.csv`는 건드리지 않는다. RTF는 세션 민감도 때문에 **4조건을 같은 세션에서
새로 측정**하며, 기존 비배치 결과의 옛 숫자와 직접 비교하지 않는다.

## 실행

```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
PYTHONPATH=/home/piai/project-ai/vad_stt_research \
  /home/piai/anaconda3/bin/python batch_addon/run_experiment_batch.py \
  --metadata data/metadata_10.csv \
  --config configs/experiment_config.yaml \
  --output batch_addon/results_batch.csv \
  --batch_size 8 --batch_size_fallback 4 --repeats 3
```

파일별 incremental 저장이라 중간에 중단돼도 완료분은 보존된다.

## 파일

- `condition_batch.py` — `run_condition_a_prime_batch()`, `run_condition_b_batch()` + 통제 주석
- `run_experiment_batch.py` — 4조건 드라이버(기존 조건·평가 코드 재사용)
- `results_batch.csv` — 출력(실행 시 생성)
