# Pilot_0707 결과 요약 (2026-07-07)

- 대상: CLIP2_0704.mp4 80~113.4s (RAM 탈거), 프레임 17장(0.5fps), 부품주입 [GPU, RAM, hand]
- 프롬프트: 본선 STEP4 관찰 프롬프트(`prompt_step4_observation.py`) 3모델 동일 적용
- 실행: n5 RTX6000, 모델당 GPU 1장, NF4 4bit, greedy(+빈관찰 시 temp0.3 1회 재시도)
- 결과 파일: `results/{qwen3vl,internvl35,minicpm45}/CLIP2_0704.mp4_80-113.observations.json`

## 정량 요약

| | Qwen3-VL-8B | InternVL3.5-8B | MiniCPM-V 4.5 |
|---|---|---|---|
| 관찰 건수 | 7 | 16 | 16 |
| 모델 로드 | 17.4s | 59.5s | 64.9s |
| 추론(1회 호출) | 149.7s* | 243.5s | 494.4s |
| GPU peak | 7.61GiB | 19.04GiB | 15.41GiB |
| 재시도 | ○ (greedy 빈관찰 → temp0.3) | ✕ | ✕ |
| 잡 wall time | 3분11초 | 5분41초 | 9분35초 |

\* Qwen 추론시간은 greedy 1차(빈 관찰) + temp0.3 재시도 합산.

## 정성 관전 포인트 (원문 확인 기준)

- **Qwen3-VL (7건)**: 손가락 단위 서술이 가장 프롬프트에 충실("왼손 검지와 엄지로 잡고
  오른손으로 당긴다"). 다만 greedy 1차가 빈 관찰로 얼어붙어 재시도로 살렸고(본선에서 알려진
  동결 패턴), 2문장 교대 반복(A,B,A,B) + 일본어 가나 혼입 1건("스타ンド") 존재.
- **InternVL3.5 (16건)**: 동일 문장("노트북 측면을 촉감으로 확인")을 2초 간격으로 기계 반복 —
  전형적 반복 퇴화 + 대상 오인(노트북 아님) + 금지된 추상동사("확인한다") 사용.
- **MiniCPM-V 4.5 (16건)**: 역시 동일 문장("RAM 슬롯 양끝을 집어 올린다") 2초 간격 반복.
  추론도 가장 느림(494s).

→ 관찰 로그로서의 품질은 **Qwen3-VL > MiniCPM ≈ InternVL** 경향(vlm_compare 3지표
결과와 방향 일치). 단 본선 러너의 dedup/화면관찰 압축을 여기선 적용하지 않고 모델
원출력을 그대로 저장했으므로, 반복 건수는 "모델의 반복 성향"을 그대로 보여주는 값임.

## 실행 이력 (시행착오 기록)

1. 1차(잡 2244/2245/2246): 프레임을 원본 1080×1920 그대로 입력 → Qwen·MiniCPM
   attention OOM(24.6/51GiB 할당 시도). InternVL만 성공(프레임당 448px 타일 1개 제한 덕).
2. 수정: `pilot_common.load_images_resized()` 추가 — Qwen 192²px(본선 파라미터),
   MiniCPM 448²px + `max_slice_nums=1`. transformers 5.13에서 processor `max_pixels`가
   비디오 경로에 미적용되는 것이 원인이라 코드에서 직접 축소.
3. 2차(잡 2249/2250): 둘 다 COMPLETED.

## 폴더 구성

```
frames/            추출 프레임 17장 (80~113.4s, 0.5fps)
detections/        YOLO(best.pt) CPU 검출 54건 → 부품주입 근거
prompt_step4_observation.py  본선 프롬프트 사본
pilot_common.py    공통(입력로드/프롬프트조립/파싱/저장/축소)
run_pilot_*.py     모델별 러너 3종
pilot.sbatch       sbatch 템플릿 (-w n5, gpu:1)
logs/              잡 로그(실패 이력 포함)
results/           ★ 모델별 관찰 JSON
```
