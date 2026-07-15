# 계획서 — 모션가이드 샘플링 코드 STEP3 정식 이식

**작성일 2026-07-08. 실행 시점: 검증 c/b/STEP4 건강검진/검증 d(재채점) 전부 완료 확인됨 —
이 문서는 계획만 담는다. 실행은 별도 승인 후.**

## 0. 배경 요약(왜 지금 이식하나)

`00_사전연구/03_전처리_모션가이드_vs_균일추출` 비교실험 + 오늘 진행한 검증으로 모션가이드
채택 결정의 근거가 갖춰졌다:
- 검증 c(좌표계 정합): CLIP1 3개 지점 육안대조 통과, 코드상 `snap_to_grid()` 보정 메커니즘 확인.
- 검증 b(CLIP4 세그먼트 중첩): 겹치는 19~37s 구간이 한 벌로만 정리됨(중복 없음) 확인.
- STEP4 VLM 관찰 전수 건강검진: 92건 중 91건 정상(1건은 사소한 문자 오염, 기능 지장 없음).
- 검증 d(새 GT 8건 재채점): 모션가이드 8/8, 균일추출(기준선) 8/8 — 동률.
- STEP5 발화손실 재현성 확인: 같은 모션 입력을 2회 실행해 서로 다른 발화 손실/경미한 환각
  패턴이 나옴 → 모션가이드 구조 문제가 아니라 LLM 생성 자체의 확률성(`do_sample=True`,
  temperature 0.2) 문제로 확인됨. 이 이슈는 균일추출 경로에도 동일하게 존재할 수 있는
  STEP5 공통 과제이므로 이식을 막을 이유가 아니다(§5 참고, 별도 과제로 분리).

현재 모션가이드는 **팀원이 만든 산출물(CSV+서브클립)을 이미 만들어진 상태로 읽기만**
하는 어댑터(`MotionSampledFrameSource`, 2026-07-07 도입)로 붙어 있다. 이번 이식의 목표는
"팀원 코드가 있어야만 동작"하는 상태에서 벗어나, **이 프로젝트 환경 안에서 그 코드
자체를 직접 실행해 서브클립을 새로 생성할 수 있게** 만드는 것이다(신규 영상 대응 포함).

## 1. 이식 범위 — 파일 목록·목적지·의존성

### 1-1. 가져올 파일

| 원본 (`00_사전연구/04_모션가이드샘플링_프로토타입/01_코드/`) | 목적지 (`STEP3_전처리/step3_components/motion_sampling/`) | 역할 |
|---|---|---|
| `motion_score.py` | `motion_score.py` | 1초 간격 absdiff 기반 Motion Score 계산 → CSV |
| `select_clip.py` | `select_clip.py` | A(Threshold+AKS)/B(CDF MGSampler)/C(AKS Window) 전략 비교, 최고 전략 선택 |
| `extract_video.py` | `extract_video.py` | 선택 구간 ffmpeg 무손실 절단(`-an`으로 오디오 스트림 자체를 안 담음 — 오디오 갈래 불가침 원칙이 서브클립 레벨에서도 물리적으로 보장됨), 타임스탬프 CSV 기록 |
| `abc_pipeline_runner.py` | `abc_pipeline_runner.py` | 위 세 개를 엮는 단일 영상 진입점 `run_abc_pipeline()` — **이식 핵심**, 이미 `files_dir`/`clips_dir` 파라미터로 출력 경로를 주입받을 수 있게 설계돼 있어 로직 수정 불필요 |
| `run_all_clips.py` | (이식 안 함, 참고용으로 사전연구 원본에만 유지) | 4클립 일괄 실행 드라이버 — 이건 "이 프로토타입 폴더 단독 실행"용 CLI라 STEP3 통합 흐름과 역할이 겹친다. 대신 §3의 자동생성 훅이 이 역할을 대체한다. |
| `__pycache__/` | 이식 안 함 | 캐시, 불필요 |

새 폴더 `step3_components/motion_sampling/__init__.py`도 추가(빈 파일 또는
`from .abc_pipeline_runner import run_abc_pipeline` 재노출용).

### 1-2. 의존성 조사 결과

`01_코드/requirements.txt`: `opencv-python`, `numpy`, `pandas`, `tqdm`, `imageio-ffmpeg`.

`파이프라인_통합실행/.venv`(STEP3가 실제로 쓰는 venv) 기준 실측 확인:

| 패키지 | 상태 |
|---|---|
| opencv-python (`cv2`) | 이미 설치됨 (4.13.0) |
| numpy | 이미 설치됨 (2.4.6) |
| tqdm | 이미 설치됨 (4.68.3) |
| imageio-ffmpeg | 이미 설치됨 (0.6.0) — `extract_video.py`도 이걸로 ffmpeg 바이너리를 동적 해석(`imageio_ffmpeg.get_ffmpeg_exe()`), 하드코딩 경로 없음. STEP3 자체 `frame_extraction.ffmpeg_bin` 설정과 동일 계열이라 충돌 없음. |
| **pandas** | **미설치 — 추가 설치 필요** (`motion_score.py`/`select_clip.py`가 사용) |

**모델 가중치 의존성 없음** — 전부 고전적 CV(absdiff) 연산이라 GPU도 필요 없다. 이식 후
단독 실행 가능 여부는 `pandas` 설치 하나로 결정된다. 설치 방법은 STEP3/4/5 공용 venv
원칙(CLAUDE.md §2)에 따라 `파이프라인_통합실행/.venv`에 `pip install pandas`로 추가.

## 2. 접합부 수정 목록 — 로직 불변, 입출력 경로/호출 인터페이스만 변경

**원칙: 4개 파일의 알고리즘 코드(모션스코어 계산식, A/B/C 전략 비교 로직, ffmpeg 절단
파라미터)는 한 글자도 안 바꾼다.** 바꾸는 건 딱 두 가지뿐이다 — (a) import 경로,
(b) `run_abc_pipeline()`을 호출하는 쪽(`frame_source.py`)의 출력 경로 인자.

### 2-1. 이식되는 4개 파일 자체의 diff

파일 안쪽 로직은 무변경. 유일하게 손댈 곳은 `abc_pipeline_runner.py`의 import 3줄뿐이며,
그마저도 **상대경로 import로 바꾸는 것뿐**(같은 패키지 안으로 옮기므로):

```diff
--- 00_사전연구/04_모션가이드샘플링_프로토타입/01_코드/abc_pipeline_runner.py
+++ STEP3_전처리/step3_components/motion_sampling/abc_pipeline_runner.py
@@
-from extract_video import extract_clips, format_ts
-from motion_score import calculate_motion_score
-from select_clip import compare_strategies
+from .extract_video import extract_clips, format_ts
+from .motion_score import calculate_motion_score
+from .select_clip import compare_strategies
```

`motion_score.py`, `select_clip.py`, `extract_video.py`는 서로를 import하지 않는 순수
함수 모음이라 **이 3개 파일은 diff 자체가 없다**(바이트 단위로 그대로 복사).

`abc_pipeline_runner.py`의 `if __name__ == "__main__":` 블록(211~214행, 하드코딩된
기본 영상 파일명)은 이식본에서는 호출되지 않는 죽은 코드지만, "원본과 다르다"는 혼동을
피하려고 지우지 않고 그대로 둔다(어차피 프로토타입 단독 실행용으로만 쓰이던 부분).

### 2-2. 호출부(`frame_source.py`) 수정 — 여기가 실제 접합부

`MotionSampledFrameSource.extract()`(현재 112~198행)에서 바뀌는 건 "CSV가 없을 때"의
동작 하나뿐이다. 현재는 없으면 `FileNotFoundError`를 던지는데(114~119행 근처), 이걸
`run_abc_pipeline()` 호출로 교체한다. 상세는 §3.

## 3. 자동화 연결 — `sampling: motion`인데 서브클립이 없을 때

### 3-1. 트리거 위치

`frame_source.py`의 `MotionSampledFrameSource.extract()` 안, 현재 이 부분:

```python
csv_path = clip_dir / "clip_timestamps.csv"
if not csv_path.exists():
    raise FileNotFoundError(...)
rows = self._read_csv(csv_path)
```

이걸 아래처럼 바꾼다(diff 형태로 제시, **아직 적용 안 함**):

```diff
 csv_path = clip_dir / "clip_timestamps.csv"
 if not csv_path.exists():
-    raise FileNotFoundError(
-        f"모션가이드 샘플링 산출물 없음: {csv_path}\n"
-        f"  → video_id({video_id})에 대응하는 '{stem}' 폴더/CSV가 있는지 확인하세요."
-    )
+    if not self.auto_generate:
+        raise FileNotFoundError(
+            f"모션가이드 샘플링 산출물 없음: {csv_path}\n"
+            f"  → video_id({video_id})에 대응하는 '{stem}' 폴더/CSV가 있는지 확인하세요.\n"
+            f"  → auto_generate=true로 config를 바꾸면 자동 생성됩니다."
+        )
+    from step3_components.motion_sampling import run_abc_pipeline
+    print(f"[MOTION-SAMPLING] {video_id}: 서브클립 산출물 없음 → 자동 생성 시작")
+    run_abc_pipeline(
+        video_path=video_path,
+        files_dir=str(clip_dir),   # CSV/summary.txt도 서브클립과 같은 폴더에
+        clips_dir=str(clip_dir),
+    )
 rows = self._read_csv(csv_path)
```

- `run_abc_pipeline()`이 이미 `files_dir`/`clips_dir` 파라미터를 받게 설계돼 있어서
  (§1-1 참고) **`run_abc_pipeline()` 내부는 손 안 대고 호출 인자만으로 원하는 위치에
  산출물을 쓰게 만들 수 있다.**
- `auto_generate` 플래그를 두는 이유: 팀원이 이미 만들어둔 기존 4클립(CLIP1~4)은 계속
  기존 산출물을 그대로 읽게 하고(재생성 시 A/B/C 전략이 우연히 다르게 뽑히면 지금까지의
  검증이 무효화됨 — 재현성 문제), **새 영상에만** 자동생성이 걸리게 한다. 기본값은
  `false`로 두고, 신규 영상 파이프라인에만 `true`로 켠다.
- `MotionSampledFrameSource.__init__`에 `auto_generate: bool = False` 파라미터 추가
  (config `sampling.params.auto_generate`로 노출).

### 3-2. 산출물 정식 위치

지금은 팀원 산출물 위치(`00_사전연구/04_.../04_샘플링 영상/`)를 그대로 참조하고 있는데,
정식 이식 후 **신규 생성분**은 이 사전연구 폴더가 아니라 파이프라인 표준 output 위치로
옮긴다:

```yaml
sampling:
  impl: "motion"
  params:
    motion_root: "STEP3_전처리/output/motion_segments"   # 기존: 00_사전연구/04_.../04_샘플링 영상
    auto_generate: true    # 새 영상용. 기존 4클립 재현시엔 false + motion_root를 사전연구 경로로.
    dedup_window_sec: 0.5
```

`output/motion_segments/<video_id>/` 아래에 `clip_timestamps.csv` + 서브클립 mp4들이
쌓이는 구조 — 기존 `_frames/<video_id>/`, `transcripts/<video_id>.json`과 같은 급의
STEP3 표준 출력물이 된다. `frame_source.py`는 `motion_root` 설정값만 이 경로로 바뀌면
되고 내부 로직(`_resolved_motion_root()`, `_read_csv()`)은 무변경.

**기존 CLIP1~4 사전연구 산출물은 그대로 `00_사전연구/04_.../04_샘플링 영상/`에 보존**
(§5 원칙과 연결 — 이미 검증에 쓰인 원본이므로 이동/삭제하지 않는다). config의
`motion_root`를 클립별로 다르게 줄 필요는 없고, 검증된 4클립은 지금처럼 사전연구
경로를 가리키는 별도 config(`config.motion_clip1_test.yaml`류)를 계속 쓰고, 신규
영상용 정식 config만 `output/motion_segments`를 가리키게 분리한다.

## 4. 재현 확인 절차 — "이식이 검증을 안 건드렸다"의 증명

**목적:** 이식 전/후로 CLIP2 하나를 같은 방식(팀원 기존 산출물 그대로 읽기, `auto_generate`
안 켠 경로)으로 돌려서 `frames_meta.json`의 `times`가 완전히 동일한지 확인. 이건 §3-1의
자동생성 훅과는 별개로, **"파일을 옮기고 import 경로만 바꾼 것이 결과에 영향을 안 줬다"**를
검증하는 것이지 "팀원 코드를 재실행해서 같은 결과가 나오는가"를 보는 게 아니다(그건
별도로 §4-2에서 다룬다).

### 4-1. 기본 재현 확인(필수)

1. 이식 전: 현재 `STEP3_전처리/output/_frames/CLIP2.mp4/frames_meta.json`을
   `frames_meta_before_port.json`으로 백업.
2. 이식 적용(§1, §2만 — §3 자동화 훅은 아직 비활성 상태로 둬도 됨, `auto_generate`
   기본값 false라 기존 4클립 경로에는 영향 없음).
3. `run_step3.py --config <기존 motion 테스트 config> --video CLIP2.mp4` 재실행.
4. 새로 생성된 `frames_meta.json`의 `times` 배열을 `frames_meta_before_port.json`의
   `times`와 원소 단위로 비교(`==` 또는 `assert times_before == times_after`).
5. **완전히 동일해야 통과.** 하나라도 다르면 이식 과정에서 뭔가 바뀐 것이므로 §2-1의
   diff가 정말 import 경로뿐이었는지부터 재확인.

### 4-2. 선택 확인(신규 생성 경로까지 검증하고 싶을 때)

CLIP2의 팀원 산출물 폴더를 임시로 다른 이름으로 옮겨두고, `auto_generate: true`로
`run_abc_pipeline()`이 처음부터 새로 서브클립을 만들게 한 뒤, 그 결과로 나온
`frames_meta.json.times`를 §4-1의 `frames_meta_before_port.json`과 비교. absdiff 기반
모션스코어 계산은 결정적 알고리즘(랜덤 시드 없음)이라 **완전히 동일한 결과가 나와야
정상**이다 — 다르면 A/B/C 전략 선택 로직 어딘가에 비결정성이 숨어있다는 뜻이라 별도
조사가 필요하다. 확인 후 임시로 옮겨둔 팀원 산출물 폴더는 원위치.

## 5. 출처 표기

- 새로 만드는 `step3_components/motion_sampling/` 폴더 최상단에 `__init__.py`가 아니라
  `README.md`(또는 각 파일 상단 docstring)로 다음을 명시:
  - 원본 위치: `00_사전연구/04_모션가이드샘플링_프로토타입/01_코드/`
  - 작성자: 팀원 산출물(원 작성자 이름은 사용자 확인 후 채움 — 이 계획서 단계에서는
    "팀원 제공"으로만 표기, 실행 시 실명 확정)
  - 이식일: 실행 승인일 기준으로 기입
  - "로직 무변경 이식, 변경분은 import 경로/출력 경로 인자뿐(계획서: `docs/계획서_모션가이드_STEP3_이식.md` 참고)"
- **사전연구 원본(`00_사전연구/04_모션가이드샘플링_프로토타입/`)은 이식 후에도 그대로
  보존**한다 — 삭제·이동 금지(이미 검증 c/b/d에 쓰인 산출물의 근거이자, 문제 발생 시
  대조군). `README.md`의 기존 "정리 기록"에 "2026-07-08: 코드가 STEP3로 정식
  이식됨(경로: step3_components/motion_sampling/) — 이 폴더는 원본 보존용으로 유지"
  한 줄만 추가.

## 6. 실행 순서 요약(승인 후)

1. `pip install pandas` (파이프라인_통합실행/.venv)
2. `step3_components/motion_sampling/` 생성, §1-1 파일 4개 복사 + import 3줄만 수정
3. `frame_source.py`에 §3-1 diff 적용(`auto_generate` 파라미터 추가, 기본 false)
4. §4-1 재현 확인(CLIP2, times 완전 일치) — **여기서 실패하면 5번 진행 안 함**
5. (선택) §4-2 재현 확인
6. 신규 영상용 config에 §3-2 경로/`auto_generate: true` 반영, 기존 4클립 config는 무변경
7. 출처 표기(§5) 추가
8. 커밋(발화손실 이슈는 "STEP5 공통 과제로 별도 분리, 이번 이식과 무관"이라고 커밋
   메시지에 명시 — 이전 CLIP4 OOM 버그 기록 원칙과 동일하게 적용)
