# 모션 가이드 샘플링_260707

AR Glass 영상 모션 가이드 샘플링 파이프라인 (타임스탬프 포함 버전, 2026-07-07)

## 폴더 구조

| 폴더 | 내용 |
|---|---|
| `01_코드` | 파이프라인 필수 코드 전체 + requirements.txt |
| `02_파일` | 영상별 모션 스코어 CSV, 전략 비교 요약(summary.txt), 클립 타임스탬프 CSV, 프로젝트 문서 |
| `03_clip` | 원본 클립 1~4 (CLIP1 정상조립과정 / CLIP2 / CLIP3 재부팅시도 전까지 / CLIP4 재부팅및BIOS) |
| `04_샘플링 영상` | 자동 샘플러가 분리한 결과 영상 (파일명에 원본 기준 시작-끝 타임스탬프 포함) |

## 실행 방법

이 폴더 루트에서:

```bash
pip install -r "01_코드/requirements.txt"   # 최초 1회
python "01_코드/run_all_clips.py"
```

`03_clip`의 모든 mp4를 A/B/C 전략 비교 후 최적 전략으로 자동 샘플링한다.

- CSV / 요약 / 타임스탬프 → `02_파일/<영상명>/`
- 분리된 영상 → `04_샘플링 영상/<영상명>/`

단일 영상만 처리하려면:

```bash
python "01_코드/abc_pipeline_runner.py" "03_clip/CLIP2.mp4"
```

## 파이프라인 단계

1. `motion_score.py` — 1초 간격 absdiff 기반 Motion Score 계산 → CSV
2. `select_clip.py` — A(Threshold+AKS) / B(CDF MGSampler) / C(AKS Window) 전략 비교, 종합점수 최고 전략 선택
3. `extract_video.py` — 선택 구간을 ffmpeg로 무손실 분리, 파일명·CSV에 타임스탬프(HH:MM:SS) 기록
4. `abc_pipeline_runner.py` — 위 단계를 연결하는 러너 / `run_all_clips.py` — 클립 1~4 일괄 실행 드라이버

## 타임스탬프

- 샘플링 영상 파일명: `clip_01_00m07s-00m37s.mp4` (원본 기준 시작-끝)
- `02_파일/<영상명>/<영상명>_clip_timestamps.csv`: 클립별 시작/끝/앵커 초 및 HH:MM:SS
- `summary.txt`에도 선택된 클립 타임스탬프 목록 포함

---

## 정리 기록 (2026-07-07)

- 원래 project-ai 루트의 `motion_guieded_sampling/`(철자 오타)이었던 것을 폴더 구조
  원칙(사전연구는 `00_사전연구/`)에 따라 여기로 이동. **코드 참조 없음 확인 후 이동**
  (STEP4·00_사전연구/03이 참조하는 것은 이 폴더가 아니라
  `~/members/안나경/master/motion_guieded_sampling`의 flat 클립 15개다).
- `03_clip/`(830M)은 `master/master_videos/`의 원본과 **md5 동일한 완전 중복**
  (예: CLIP1 eac8997f…). 용량 회수가 필요하면 이쪽을 지우면 된다.
- `04_샘플링 영상/`(547M)은 master의 flat본과 파일 크기가 다른 별도 인코딩본(타임스탬프
  파일명 버전)이라 중복 아님 — 유지.

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `01_코드/motion_score.py` | 영상을 1초 간격 샘플링해 absdiff 기반 Motion Score를 계산, CSV 저장 |
| `01_코드/select_clip.py` | Motion Score CSV로 3가지 샘플링 전략(A: Threshold+AKS / B / C) 비교 후 AKS Coverage 기준 최적 전략 선택 |
| `01_코드/extract_video.py` | 선택된 구간을 ffmpeg subprocess로 개별 클립 추출 (MoviePy 불필요) |
| `01_코드/abc_pipeline_runner.py` | 위 3단계(스코어→구간선정→추출)를 한 영상에 대해 묶어 도는 파이프라인 러너 |
| `01_코드/run_all_clips.py` | `03_clip/`의 클립 1~4를 일괄 처리하는 배치 진입점 (CSV/요약→`02_파일/`, 클립→`04_샘플링 영상/`) |
| `02_파일/<영상명>/*_motion_scores.csv` | 초 단위 모션 스코어 원자료 |
| `02_파일/<영상명>/*_clip_timestamps.csv` | 선정된 구간의 시작~끝 타임스탬프 |
| `02_파일/<영상명>/summary.txt` | 클립별 전략 비교·선정 요약 |
| `02_파일/샘플링_결과표.txt` | 4클립 종합: 652초→495초(15클립), 유지율 75.9%, 829MB→547MB |
| `02_파일/보고서_260701.md`, `보고서_모션가이드샘플링.md` | 결과 보고서(2026-07-01) |
| `02_파일/파이프라인_설계문서.md` | 설계 문서 |
| `02_파일/모션가이드샘플링_파이프라인.pptx` | 발표용 슬라이드 |
| `03_clip/*.mp4` | 원본 클립 4개 (★master_videos와 완전 중복 — 삭제 후보) |
| `04_샘플링 영상/<영상명>/clip_NN_시작-끝.mp4` | 샘플러가 잘라낸 결과 클립 15개(타임스탬프 파일명) |

후속 연구는 `00_사전연구/03_전처리_모션가이드_vs_균일추출` (이 프로토타입의 클립으로
YOLO+VLM 관찰 품질을 균일추출과 비교, 결론: 동작구간 우세/반복구간 열세, 본선 미채택).
