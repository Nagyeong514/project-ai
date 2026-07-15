# Pilot_0707 — 3 VLM × STEP4 관찰 프롬프트 파일럿 계획

- 작성: 2026-07-07
- 대상 영상: `260704_VLM/scenario_videos_0704/CLIP2_0704.mp4` 중 **80s~113.4s 구간** (RAM 탈거 01:29~01:53 포함)
  - ⚠️ 요청은 1:20~2:00이었으나 영상 총 길이가 **1:53.4**라서 실제 구간은 80~113.4s(약 33초)로 확정
- 프롬프트: `260707_Qualitative_eval/Qwen3-VL-8B-Instruct_프롬프트_정리.md` 1장의 **본선 STEP4 관찰 프롬프트**
  (원본 `안나경/project-ai/STEP4_YOLO_VLM관찰/step4_prompts/vlm_observation.py` — 시스템 규칙 9개 + few-shot + 영상길이 문장)
- 대상 모델(vlm_compare 3종, 전부 NF4 4bit·greedy):
  1. Qwen/Qwen3-VL-8B-Instruct (`env_qwen3vl`)
  2. OpenGVLab InternVL3.5-8B (`env_internvl35`)
  3. openbmb MiniCPM-V 4.5 (`env_minicpm45`)

## 방식 (STEP4 본선 파이프라인 방식 그대로)

```
[CPU] ffmpeg: 80~113.4s @0.5fps → 프레임 ~17장          (frames/)
[GPU] YOLO best.pt(dell7920_v1): 17장 검출               (detections/pilot.json)
        └ 구간(단일 33s 청크)에서 실제 검출된 클래스만 → [고정 사실] 부품 주입 목록
[GPU] 모델별 1회 generate: STEP4 프롬프트 + 프레임 17장  (results/<model>/)
        - Qwen3-VL   : native video 입력(STEP4 어댑터와 동일, video_metadata에 실측 fps 명시)
        - InternVL3.5: 멀티프레임 표준 패턴(Frame-i: <image>)로 등가 적용
        - MiniCPM-V  : msgs에 프레임 리스트로 등가 적용
        - 공통: max_new_tokens 4000, do_sample=False, 빈 관찰 시 temp 0.3 1회 재시도
[저장] observations JSON + 모델 원출력(raw) + inference_time_sec(이번엔 호출 단위 계측 기록)
```

- 구간이 33초 = 단일 40s 청크 1개 → **모델당 generate 호출 딱 1회**로 GPU 시간 최소화.
- InternVL/MiniCPM은 STEP4 본선 코드가 Qwen 전용이라 입력 결합부만 새로 작성(프롬프트 텍스트는 3모델 동일).
- 안나경 폴더는 프롬프트 파일 읽기만(수정 없음). 모든 산출물은 이 폴더 안.

## 실행 형태

sbatch 1개(RTX6000, n5, GPU 1장): YOLO → qwen3vl → internvl35 → minicpm45 순차 실행
(각 모델은 자기 venv python으로 별도 프로세스 → 로드/해제 자동, GPU 공존 없음)

## 예상 소요 시간

| 단계 | GPU 사용 | 예상 |
|---|---|---|
| 프레임 추출(ffmpeg) | ✕ | <1분 |
| YOLO 17장 | ○ | ~1분 |
| Qwen3-VL (로드+1회 생성) | ○ | 4~7분 |
| InternVL3.5 (로드+1회 생성) | ○ | 3~6분 |
| MiniCPM-V 4.5 (로드+1회 생성) | ○ | 5~8분 |
| **합계** | GPU 점유 | **약 15~25분** (전체 wall ~20~30분, n5 현재 idle이라 대기 0) |

근거: 모션가이드 실측(19프레임 클립당 STEP4 전체 152~410초, VLM 지배적) + vlm_compare 실측
(단일 이미지 기준 InternVL 14.5s < Qwen 21s < MiniCPM 26.6s, 로드는 캐시된 가중치라 수 분).
리스크: MiniCPM은 반복생성 루프 이력(최대 138s/프레임)이 있어 상한을 넘길 수 있음 — 잡에
30분 여유를 둠.

## 산출물

```
Pilot_0707/
├── PILOT_PLAN.md               # 이 문서
├── frames/                     # 추출 프레임 ~17장
├── detections/pilot.json       # YOLO 검출(청크 주입 근거)
├── prompt_step4_observation.py # 본선 프롬프트 사본(참조 고정용)
├── run_pilot_{qwen3vl,internvl35,minicpm45}.py
├── pilot.sbatch
├── logs/
└── results/{qwen3vl,internvl35,minicpm45}/
    └── CLIP2_0704_80-113.observations.json  (+ raw, inference_time_sec 포함)
```
