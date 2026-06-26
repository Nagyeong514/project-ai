# 명장 암묵지 추출 — VLM 입력 조건 비교 (연구 ④)

앞 3연구(STT 선정 → VAD 효과 → VAD 엔진비교)에 이어지는 네 번째 연구.
파이프라인·통계·파라미터(pad400, turbo, WebRTC)를 재사용한다.

## 가설
> **발화가 있는 곳에 암묵지가 있다.** 명장은 의미 있는 행동을 말로 함께 설명하므로,
> 발화 구간만 봐도 암묵지를 충분히 포착할 수 있다.

## 조건 (config.conditions — use_frames/use_stt 로 기여 분리)
| 조건 | 입력 | 역할 |
|---|---|---|
| **A** | 통영상 균등 32프레임(8장×4청크 병합) + 전체 STT | upper bound 천장 |
| **B** | 발화 구간 프레임 + 구간 STT | 가설 구현 |
| **B_vid** | 발화 구간 프레임만 (STT 없음) | ablation: 영상 단독 기여 |
| **B_txt** | 구간 STT만 (프레임 없음) | ablation: 발화 단독 기여 |
| **C** | B 구간과 동일 길이 랜덤창 + STT | 대조군 (보류 — 침묵 충분한 영상 필요) |

- **ablation(B vs B_vid vs B_txt)** 으로 "영상이 STT 위에 실제로 기여하나"를 분리 → **STT 혼입 교란** 방어.
- **본 실험 메인 = B vs C**(발화 *위치* 효과), 침묵 충분한 영상 확보 후 활성화.

## 채점 (개념 키워드)
정답지 `keywords=[[동의어,...],...]` (안쪽 = 1개 필수개념). 개념 적중 = 동의어 중 하나라도 출력에 등장. 구간 recall = 적중/전체.

## 실행
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:\
/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
PY=/home/piai/anaconda3/bin/python
# data/raw/<vid>.{mp4,wav} + data/ground_truth/<vid>_answerkey.json 준비 후:
$PY -m experiments.run_experiment V03      # config.vlm.backend: stub|qwen
$PY -m evaluation.score V03 qwen
```

## 환경 / 8GB VRAM 대응
- GPU=RTX2080 **8GB** → **Qwen2.5-VL-3B-Instruct 4bit**(bitsandbytes)가 상한. 7B 불가.
- OOM 방지 레버: `frames.max_frames_per_call=8`, `vlm.max_pixels=313600`, `expandable_segments`.
  (16프레임·고해상도는 OOM. 8프레임 기준 peak ~2~3GB.)
- ffmpeg/ffprobe = `anaconda3/envs/deep/bin`. STT는 cublas/cudnn LD_LIBRARY_PATH 필요.

## 알려진 한계 (코드로 못 고침 — 데이터/사람 필요)
1. **n=1 영상** → 통계 불가, 일화 수준. 영상 여러 개 필요.
2. **B ≈ A 구조적** → 현 영상은 발화 90%라 수동 7구간이 영상 전체를 덮음. C 대조도 불가.
   발화:비발화 ≈ 50:50 영상이라야 B vs C·A vs B 차이가 의미.
3. **정답지 순환** → keywords를 STT에서 뽑음 → 채점이 STT echo를 보상할 위험.
   ablation(B_vid)로 영상 단독 기여를 따로 봐서 부분 방어. 궁극엔 영상 근거 정답지 필요.
4. **키워드 채점** → 동의어 누락(가짜 0점)·부분문자열(가짜 적중) 잔존. 최종은 LLM-judge/사람.
5. **3B/4bit 품질 천장** → 작은 모델이라 서술이 얕거나 환각 가능. 8GB의 대가.

## 상태 (2026-06-27)
- ✅ 전 파이프라인 + Qwen2.5-VL-3B 실추론 동작. ablation·A청크·프레임상한·STT 1회전사 적용.
- ✅ 데이터: V03(11:16 원본, 정답지 7구간 일치). V01(2분, 폐기)·V02(4분, 앞부분).
- 🔜 침묵 충분한 영상 + 영상 다수 확보 → C 활성화 + 본 실험.

## 구조
```
configs/config.yaml             파라미터 (조건·프레임상한·max_pixels·생성옵션)
pipeline/vad.py                 WebRTC VAD + pad400
pipeline/stt.py                 turbo 전체 1회 전사 → 구간 배분
pipeline/frames.py              extract/uniform_pick/chunk
pipeline/prompt.py              공통 프롬프트 (echo 방지, video/text-only 대응)
pipeline/vlm.py                 Qwen2.5-VL 추론 (stub/qwen, 4bit, 반복억제, JSON복구)
experiments/run_experiment.py   조건 오케스트레이션 (A청크 + ablation)
evaluation/score.py             개념 키워드 채점 + ablation/B-vs-C 비교
```
