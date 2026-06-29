# 명장 암묵지 추출 — VLM 입력 조건 비교 (연구 ④)

앞 3연구(STT 선정 → VAD 효과 → VAD 엔진비교)에 이어지는 네 번째 연구.
파이프라인·통계·파라미터(pad400, turbo, WebRTC)를 그대로 재사용한다.

## 가설
> **발화가 있는 곳에 암묵지가 있다.** 명장은 의미 있는 행동을 말로 함께 설명하므로,
> 발화 구간만 봐도 암묵지를 충분히 포착할 수 있다.

## 조건 (입력 영상 범위만 변수, 나머지 전부 고정)
| 조건 | 입력 | 역할 |
|---|---|---|
| **A** | 통영상 2fps + 전체 STT | upper bound (천장 참고치) |
| **B** | 발화 구간(pad400) 2fps + 구간 STT | 가설 구현 |
| **C** | B 각 구간과 **동일 길이의 랜덤 위치 창** 2fps + (B와 동일) 구간 STT | 대조군 |

- **메인 비교 = B vs C** (구간 단위 페어, Wilcoxon). "발화 *위치*" 효과를 분리.
  - B > C → 위치가 핵심(가설 성립) / B ≈ C → 그냥 짧아서 좋았던 것(가설 약화)
- A는 구간 통계에서 제외, 정성 천장 비교로만 사용.

## 2단계 구조
1. **방법 검증(본 코드):** 영상 + 내가 쓴 정답지로 B vs C 메커니즘 확인. 명장 검수 불필요.
2. **현실 적용(후속):** 실제 명장 영상 소수로 재현.

## 실행
```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:\
/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
PY=/home/piai/anaconda3/bin/python

# 0) 오디오 추출 (mp4 → 16k mono wav)  ※ ffmpeg: anaconda3/envs/deep/bin
#    (run_experiment 전에 data/raw/<vid>.wav 가 있어야 함)

# 1) 정답지 템플릿 생성 → 직접 채우기 → .template 떼기
$PY scripts/make_answerkey_template.py V01

# 2) 실험 (config.vlm.backend='stub' 이면 모델 없이 흐름만)
$PY -m experiments.run_experiment V01

# 3) 채점 (정답지 대조 + B vs C Wilcoxon)
$PY -m evaluation.score V01 stub
```

## 현재 상태 (2026-06-26)
- ✅ 파이프라인 전부 구현 (VAD→STT→A/B/C 프레임→VLM→채점)
- ⏳ **VLM 가중치 미도착** — 폐쇄망. `vlm.backend='stub'` 으로 흐름만 검증 중.
  - 가중치 오면 `config.yaml` 의 `vlm.backend: qwen` 으로만 변경.
  - 필요: **Qwen2.5-VL-3B-Instruct** 가중치 + `qwen_vl_utils` (8GB VRAM → 3B/4bit 상한).
- ⚠️ **데이터 한계:** 현재 V01(2분14초)은 발화 96%·비발화 4% → C 대조 무의미.
  본 실험엔 발화:비발화 ≈ 50:50, 영상 3~5개 필요. (V01은 파이프라인 검증용)

## 구조
```
configs/config.yaml          모든 파라미터 (pad400, fps2, 모델 등)
pipeline/vad.py              WebRTC VAD + pad400 → 발화 구간
pipeline/stt.py              turbo 구간/전체 전사
pipeline/frames.py           A/B/C 프레임 추출
pipeline/prompt.py           공통 프롬프트 (세 조건 동일)
pipeline/vlm.py              Qwen2.5-VL 추론 (stub/qwen 디스패치)
experiments/run_experiment.py  전 과정 오케스트레이션
evaluation/score.py          정답지 대조 + Wilcoxon
scripts/make_answerkey_template.py  정답지 골격 생성
```