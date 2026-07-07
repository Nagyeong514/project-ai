# 03_전처리_모션가이드_vs_균일추출 (구 master/EXP_motion_guided_vs_uniform)

모션가이드 샘플링으로 잘라낸 클립들(`../motion_guieded_sampling/*.mp4`, 15개)을 기존
STEP4 YOLO→VLM 파이프라인에 통과시켜 **클립별 행동(action) 리스트 JSON**을 만드는 실험.
나중에 균일 프레임 추출(모션가이드 미사용) 결과와 비교하기 위한 **실험군** 데이터 생성용.

## 구조

- 기존 코드는 **수정/복사 없이 import** 해서 쓴다:
  - `project-ai/STEP3_전처리/step3_components/frame_extract.py` — ffmpeg 프레임 추출
  - `project-ai/STEP4_YOLO_VLM관찰/step4_runner.Step4Runner` — YOLO(서브프로세스 격리)+VLM 관찰
  - `project-ai/tacit_common/artifacts` — 중간 산출물 입출력
- `config.yaml` — 파이프라인_통합실행/config.yaml 복사본에서 `paths:`만 이 폴더 절대경로로
  교체(기존 STEP3/4 output 오염 방지). 추론 설정(detector/vlm/frame_extraction)은 원본과 동일
  — 실험군/대조군의 차이가 입력 클립에서만 나게 하기 위함.
- `run_experiment.py` — 클립 순회 러너. STT 없음(STEP3의 프레임 추출만 재사용).
  - 클립당 최소 4프레임 보장(적게 뽑히면 fps 올려 재추출). 그 외 fps 정책은 원본 그대로(0.5).
  - 클립마다 단순화 JSON을 갱신하므로 중간에 죽어도 완료분은 남는다.
  - 기본적으로 관찰 산출물이 이미 있는 클립은 skip(재실행: `--no-skip-existing`).

## 실행

GPU 노드(n5) 필요. venv는 `파이프라인_통합실행/.venv` 공용(CLAUDE.md §2 예외 조항).

```bash
cd run_logs
sbatch exp_mgs.sbatch --limit 1   # 스모크: 첫 클립 1개
sbatch exp_mgs.sbatch             # 전체 15클립(완료분 skip)
```

## 산출물

- `output/actions_by_clip.json` — **최종 통합 JSON.** 메타데이터(타임스탬프 등) 전부 제거,
  행동 문자열만:
  ```json
  { "CLIP1": { "clip01": ["...를 집어든다", "..."], "clip02": [...] }, "CLIP2": {...} }
  ```
- `output/run_summary.json` — 클립별 프레임 수/관찰 건수 + **처리한 총 프레임 수**
  (대조군과 비용 비교용).
- `output/_frames/<클립파일명>/` — 추출 프레임(중간산출물)
- `output/detections/`, `output/vlm_observations/` — STEP4 원본 형식 산출물(타임스탬프 포함
  전체 버전 — 단순화 JSON은 여기서 파생되므로 필요하면 언제든 재생성 가능)
