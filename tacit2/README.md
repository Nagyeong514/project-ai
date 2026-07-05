# tacit2 — 전처리 파이프라인 v2 (무에서 재작성)

스마트글래스 1인칭 수리 영상(CLIP1~4) → 암묵지 후보 JSON(스키마 v1.3).
v1 에서 5일간 겪은 오류 역사를 **구조로** 막는 재설계.

## 오류 역사 → 구조 대응 매핑

| v1 에서 터진 것 | v2 의 구조적 대응 |
|---|---|
| VLM OOM (99/115프레임 통짜 입력) | **청크 호출**: 30초 창, 호출당 ≤15프레임 → vision 토큰 상한 고정 |
| 관찰 0건 (fps 0.3/0.4 대역) | fps 0.5 단일값 유지(그 대역 안 밟음) + 빈 청크 1회 재시도(temp 0.3) |
| 동일 문장 130초 반복(퇴화 루프) | 청크로 생성 활주로 축소 + 프롬프트 반복규칙 + **dedup 병합(2문장 교대 패턴까지)** |
| LLM unload 누락 → 배치 전멸 | **스테이지-우선 실행**: 모델당 로드 1회, 공존 순간 자체가 없음 + release() 표준화 |
| YOLO 의 CUDA_VISIBLE_DEVICES 오염 | 서브프로세스 완전 격리(v1 해법 계승) |
| YOLO cuDNN 로딩 실패 | 자식 env 의 LD_LIBRARY_PATH 에 venv nvidia lib **앞-추가**(덮어쓰기 금지) |
| device_map=auto 쏠림/줄다리기 | **GPU 1장 고정**(청크라 8B NF4 충분, 14B NF4 도 ~10GB) — auto 폐지 |
| 55→4 프레임 조용한 축소 | 청크마다 video_grid_thw T값 **assert** — 재발 시 즉사(조용한 실패 금지) |
| evidence 오태깅("시작합니다" 갖다붙임) | evidence/timestamp 를 LLM 에서 **뺏어서 코드가 계산**(±4초 실제 발화 대조) |
| 도미노 병합(클립 전체=윈도우 1개→STEP5 OOM) | aligner 에 **merge_cap_sec=20초** 상한 |
| _frames/ 덮어쓰기 | 클립별 폴더 frames/<clip>/ |
| exit 0 인데 결과 텅 빔 | `--smoke`: 산출물 **내용까지 assert**(관찰>0, 3연속 반복 없음, evidence 정합) |

## 구조

```
tacit2/
├── config.yaml            # ★ paths.videos 4개 + yolo_weights 만 채우면 됨
├── run_pipeline.py        # 엔트리 (스테이지-우선 러너)
├── preflight.py           # 무거운 import 전 5초 검증
├── core/    config.py paths.py schema.py timeline.py gpu.py
├── stages/  s1_stt.py s2_frames.py s3_yolo.py s3_yolo_entry.py s4_vlm.py s5_fusion.py
└── prompts/ vlm_prompt.py fusion_prompt.py
```

산출물 계약(스테이지 간 인터페이스 = 파일):
```
output/transcripts/<clip>.json → frames/<clip>/ → detections/<clip>.json
     → vlm_observations/<clip>.json → tacit_json/<clip>.tacit.json
```
각 스테이지는 출력이 이미 있으면 skip → 중간에 죽어도 그 지점부터 재개.

## 실행 순서 (반드시 이 순서로)

```bash
# 0. config.yaml 의 paths.videos / yolo_weights 채우기
# 1. 스모크 (첫 클립 1개, 산출물 assert 까지) — 인터랙티브 GPU 셸에서
python run_pipeline.py --smoke

# 2. 통과하면 전체
python run_pipeline.py

# 부분 실행 예시
python run_pipeline.py --stages vlm fusion            # VLM 부터만
python run_pipeline.py --clips CLIP4 --force          # CLIP4 만 강제 재실행
python run_pipeline.py --stages vlm --clips CLIP3 --force   # 반복 실험용
```

## 합격 기준 (이게 v2 의 검증 매트릭스)

- OOM 0회 (청크 구조상 프레임 수와 무관해야 정상)
- CLIP4 포함 4클립 전부 관찰 ≥1건
- dedup 후 동일 문장 3연속 0건
- 모든 청크 JSON 정상 닫힘 (max_new_tokens 1200 은 청크당 넉넉)
- evidence=utterance 인 step 은 전부 실제 발화 ±4초 이내 (코드가 보장)

## 미리 알아둘 것

- YOLO 가 죽어도 파이프라인은 계속 간다(빈 검출 → 부품주입만 손해). cuDNN 처방이
  안 먹으면 detections/*.json 의 error 필드 확인.
- VLM 반복이 dedup 후에도 남으면 → 그건 영상이 진짜 반복 동작이라는 뜻
  (repeat_count/duration 으로 기록되고, 융합 프롬프트 규칙 6이 "오래 걸린 것 자체"를
  암묵지 재료로 다룸).
- transformers 버전에 따라 Qwen3-VL 클래스명이 다를 수 있음 —
  AutoModelForImageTextToText 로 로드 실패 시 s4_vlm.py 의 import 를
  해당 버전 문서의 클래스로 교체(그 한 줄만).
