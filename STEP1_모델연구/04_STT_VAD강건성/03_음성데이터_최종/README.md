# STT/VAD 연구 인수인계 (handover)

작업자: 길예진 · 작성일: 2026-07-13
원본 프로젝트 루트: `/home/ai_user/team_a2/members/길예진/stt-project/`

## 0. 이 폴더의 성격 (중요)

- 이 폴더는 **인수인계용 코드 사본**입니다. 원본 스크립트는 `scripts/`와 `dataset/`에 그대로 있습니다.
- 모든 파이썬 스크립트는 경로를 **자기 파일 위치 기준**(`Path(__file__).parent.parent`)으로 계산하므로,
  **실제 실행은 반드시 원본 위치의 파일로** 하세요 (프로젝트 루트에서 `.venv/bin/python scripts/xxx.py`).
  handover 안의 사본을 그대로 실행하면 `handover/`를 프로젝트 루트로 착각해 입력 파일을 못 찾습니다.
- 각 사본 상단에 `[인수인계 헤더]`(용도/입력/출력/실행 예시)만 추가했고, 코드 로직은 원본과 동일합니다.
- `results*/` 실험 결과 폴더는 복사하지 않았습니다(코드만 정리). 결과·보고서 위치는 9절 참고.

## 1. 파이프라인 전체 흐름

원시 음성(AI 글래스 촬영 mp4, `dataset/raw_videos/`)을 ffmpeg로 16kHz mono wav로 변환한 뒤
(`01_preprocess`), **Silero VAD**로 무음·소음 구간을 잘라내 발화 구간만 이어붙이고(`02_vad`,
→ `dataset/audio_vad/`), 이를 **faster-whisper large-v3-turbo + 도메인 프롬프트**(`language=ko`,
`float16`, GPU)에 입력해 전사하고(`03_stt`), 정답 대본(`dataset/transcripts/`) 대비
CER/WER/한영혼용용어/핵심키워드/RTF를 계산해 평가합니다(`04_evaluate`). 이 확정 파이프라인 위에서
노이즈 강건성 추가 실험(`05_noise_robustness`)을 수행했습니다. VAD를 쓰는 이유: 원본 오디오를
그대로 넣으면 무음 구간에서 whisper 계열 할루시네이션으로 CER이 심각하게 왜곡되며, VAD 적용으로
CER·할루시네이션이 87~94% 개선됨을 raw vs vad 비교 실험으로 확인했습니다
(`results_raw_vs_vad/conclusion.md`).

## 2. 수행한 실험 4가지와 스크립트 매핑

| 실험 | 사용 스크립트 (handover 내 위치) | 결과 폴더 |
|---|---|---|
| raw vs vad 비교 | `03_stt/run_stt.py`(입력 폴더 바꿔 2회), `04_evaluate/evaluate.py` | `results_raw_vs_vad/`, `results_novad_backup/` |
| VAD 방식 비교 (A 무VAD / B Silero / C WebRTC) | `02_vad/apply_vad.py`, `02_vad/apply_webrtc_vad.py`, `02_vad/run_stt_webrtc.py`, `02_vad/evaluate_vad_compare.py` | `results_vad_compare/` |
| STT 모델 비교 (faster-whisper ±프롬프트 / speechbrain / OWSM) | `03_stt/run_stt.py`, `04_evaluate/evaluate.py` | `results/` |
| 노이즈 강건성 (D0~D4) | `05_noise_robustness/add_noise.py`, `run_stt_noise.py`, `evaluate_noise.py`, `slurm/run_stt_noise.sbatch` | `results_noise_robustness/` |

- 어느 실험에 쓰였는지 불확실한 스크립트는 없어서 `99_unsorted/`는 만들지 않았습니다.
- raw vs vad 비교의 `comparison.csv`는 전용 스크립트가 아니라 `results/`(VAD)와
  `results_novad_backup/`(raw)의 기존 산출물을 대조해 작성된 것입니다.

## 3. 환경 설정

```bash
cd /home/ai_user/team_a2/members/길예진/stt-project

# 1) venv 생성 및 패키지 설치 (기존 .venv가 이미 있으므로 새로 만들 필요는 없음)
python3.12 -m venv .venv
.venv/bin/pip install -r handover/requirements.txt

# 2) 시스템 의존성: ffmpeg (01_preprocess 단계에만 필요)
ffmpeg -version   # 없으면 관리자에게 설치 요청
```

### CUDA 12 라이브러리 이슈 (반드시 알아둘 것)

- 이 venv의 torch는 CUDA 13 빌드지만, ctranslate2(faster-whisper 백엔드)는 **CUDA 12용
  `libcublas.so.12` / `libcudnn.so.9`**를 요구합니다. 그냥 실행하면 GPU 노드에서
  `libcublas.so.12 not found`로 죽습니다.
- 해결: venv에 pip으로 설치된 `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`의 라이브러리 경로를
  실행 시점에 `LD_LIBRARY_PATH`에 추가합니다 (`slurm/run_stt_noise.sbatch`에 적용돼 있음):

```bash
SITE=.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$PWD/$SITE/nvidia/cublas/lib:$PWD/$SITE/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

- 출처: `results_noise_robustness/README.md` "실행 관련 특이사항", `STT_연구계획서_CLI.md` 6-2절.

## 4. SLURM 사용법 (GPU 작업)

- 로그인 노드에는 GPU가 없습니다. **STT 실행(faster-whisper/speechbrain/OWSM)은 반드시
  SLURM으로 GPU 노드 n5(RTX6000)에 제출**해야 합니다. CPU로 돌린 STT 결과는 RTF 비교
  공정성 때문에 사용하지 않습니다(기존 측정이 전부 n5/RTX6000/float16 기준).
- VAD 전처리·노이즈 합성·평가는 CPU 작업이라 로그인 노드에서 그대로 실행해도 됩니다.

```bash
# 배치 제출 (권장 — LD_LIBRARY_PATH 설정이 sbatch 스크립트에 포함됨)
sbatch scripts/run_stt_noise.sbatch

# 대화형 실행 (LD_LIBRARY_PATH를 직접 export한 뒤)
srun -p RTX6000 -w n5 --gres=gpu:1 .venv/bin/python scripts/run_stt.py --model all

# 상태 확인 / 로그
squeue -u $USER
# 로그 위치: results_noise_robustness/slurm_stt_<jobid>.log (sbatch --output 지정 경로)
```

- n5 GPU 2장이 팀 작업에 점유된 경우가 많아 큐 대기가 발생할 수 있습니다. 기다리면 됩니다.

## 5. 폴더별 실행 순서 (위에서 아래로 따라 하면 재현됨)

실행은 전부 **프로젝트 루트에서**, 원본 `scripts/`·`dataset/`의 파일로 합니다.

### 01_preprocess — 영상 → wav 변환
| | |
|---|---|
| 하는 일 | mp4/mov/mkv → 16kHz mono PCM16 wav (ffmpeg) |
| 실행 | `cd dataset && ./video_to_wav.sh` |
| 입력 | `dataset/raw_videos/clip{1..4}.mp4` |
| 출력 | `dataset/audio/clip{1..4}.wav`, `conversion_log.txt` |

### 02_vad — VAD 전처리 (+ VAD 비교 실험)
확정 파이프라인에 필요한 것은 `apply_vad.py` 하나입니다. 나머지 3개는 VAD 비교 실험 재현용.

| 순서 | 명령 | 입력 → 출력 | 노드 |
|---|---|---|---|
| 1 | `.venv/bin/python scripts/apply_vad.py` | `dataset/audio/` → `dataset/audio_vad/` + trim/timing 로그 | CPU |
| (비교C) | `.venv/bin/python scripts/apply_webrtc_vad.py` | `dataset/audio/` → `dataset/audio_webrtc/` | CPU |
| (비교C) | `srun -p RTX6000 -w n5 --gres=gpu:1 .venv/bin/python scripts/run_stt_webrtc.py` | `dataset/audio_webrtc/` → `results_vad_compare/stt_outputs/C_webrtc/` | **GPU** |
| (비교) | `.venv/bin/python scripts/evaluate_vad_compare.py` | A/B/C 결과 → `results_vad_compare/metrics/vad_metrics_summary.csv`, `final_vad_report.md` | CPU |

※ `evaluate_vad_compare.py`는 A(무VAD) 조건 데이터로 `results_novad_backup/`과
`results_vad_compare/stt_outputs/A_no_vad/`를 읽으므로, 기존 결과 폴더가 있어야 돌아갑니다.

### 03_stt — STT 실행 (모델 비교)
| | |
|---|---|
| 하는 일 | faster-whisper large-v3-turbo(프롬프트 유/무), speechbrain KsponSpeech, OWSM v4를 동일 입력으로 실행하고 RTF 기록 |
| 실행 | `srun -p RTX6000 -w n5 --gres=gpu:1 .venv/bin/python scripts/run_stt.py --model all` (개별: `--model faster_whisper\|speechbrain\|owsm`) |
| 입력 | `dataset/audio_vad/clip{1..4}.wav` |
| 출력 | `results/stt_outputs/{모델태그}/clip{n}.txt`, `results/metrics/timing_log.csv` |

이미 완료된 clip은 자동으로 건너뜁니다(재실행 안전). raw vs vad 실험의 raw 쪽은 이 스크립트의
`AUDIO_DIR`를 `dataset/audio`로 바꿔 실행한 결과(`results_novad_backup/`)입니다 — 사본 코드는
현재 확정값(`audio_vad`) 기준.

### 04_evaluate — 평가
| | |
|---|---|
| 하는 일 | CER/WER/한영혼용용어(0.5점 음차 허용)/핵심키워드/RTF 계산, 가중합 최종 점수·리포트 생성 |
| 실행 | `.venv/bin/python scripts/evaluate.py` |
| 입력 | `results/stt_outputs/`, `dataset/transcripts/clip{n}_gt.txt`, `dataset/keywords/*.json`, `results/metrics/timing_log.csv` |
| 출력 | `results/metrics/metrics_summary.csv`, `results/qualitative_review.csv`, `results/error_analysis.csv`, `results/final_report.md` |

⚠ `evaluate.py`는 `evaluate_noise.py`·`evaluate_vad_compare.py`가 **import하는 공용 모듈**이기도
합니다. 원본 `scripts/evaluate.py`를 이동·삭제하면 두 스크립트가 깨집니다.

### 05_noise_robustness — 노이즈 강건성 실험
| 순서 | 명령 | 입력 → 출력 | 노드 |
|---|---|---|---|
| 1 | `.venv/bin/python scripts/add_noise.py` | `dataset/audio_vad/` + `dataset/ESC-50/` → `dataset/audio_noise/{D1..D4}/`, `d4_noise_log.txt` | CPU |
| 2 | `sbatch scripts/run_stt_noise.sbatch` | `dataset/audio_noise/` → `results_noise_robustness/stt_outputs/{D1..D4}/`, `timing_log.csv` | **GPU** |
| 3 | `.venv/bin/python scripts/evaluate_noise.py` | stt_outputs + timing 로그 → `results_noise_robustness/metrics_summary.csv` | CPU |

D0(무노이즈 기준선)는 새로 실행하지 않고 `results/stt_outputs/faster_whisper_prompt/` 결과를
재사용합니다. ESC-50 데이터셋은 `dataset/ESC-50/`에 이미 있습니다(외부 공개 데이터셋이라
handover에 복사하지 않음).

### slurm — SLURM 제출 스크립트
`run_stt_noise.sbatch` 1개 (다른 GPU 실행은 전부 `srun` 대화형으로 수행했음).

## 6. 확정 설정 요약

| 항목 | 값 |
|---|---|
| Silero VAD | `threshold=0.5`, `min_speech_duration_ms=200`, **`min_silence_duration_ms=500`, `speech_pad_ms=300`** (`apply_vad.py`) |
| STT 모델 | faster-whisper **large-v3-turbo**, `device=cuda`, **`compute_type=float16`** |
| STT 옵션 | **`language="ko"`**, `initial_prompt="Dell Precision 7920 조립, Motherboard, RAM, BIOS, GPU, 슬롯"`, 그 외 디코딩 파라미터는 기본값(임의 튜닝 금지 원칙) |
| 노이즈 실험 | **seed=42** 고정(화이트 노이즈·ESC-50 파일 선택·삽입 위치), SNR = `10·log10(P_speech/P_noise)`, D4 이벤트층은 hand_saw만 사용(drilling은 ESC-50에 없음) |
| WebRTC VAD (비교용) | `aggressiveness=2`, `frame_duration_ms=30`, 병합/패딩 규칙은 Silero와 동일하게 통일 |
| 실행 환경 | SLURM n5 / RTX6000 / GPU 1장 (CPU STT 결과는 사용 금지) |

## 7. 경로 주의 (하드코딩 절대경로)

코드 수정 없이 그대로 복사했으므로, 프로젝트를 다른 위치로 옮기면 아래 파일을 수정해야 합니다.

| 파일 | 위치 | 내용 |
|---|---|---|
| `slurm/run_stt_noise.sbatch` | 8행 `#SBATCH --output=` | `/home/ai_user/team_a2/members/길예진/stt-project/results_noise_robustness/slurm_stt_%j.log` |
| `slurm/run_stt_noise.sbatch` | 10행 `cd` | `/home/ai_user/team_a2/members/길예진/stt-project` |

그 외 파이썬 스크립트는 절대경로 하드코딩이 없고 전부 `Path(__file__).resolve().parent.parent`
기준 상대 경로입니다. 단, 그 때문에 **스크립트 파일이 `프로젝트루트/scripts/` 위치에 있어야**
경로가 맞습니다(0절 참고). `run_stt_noise.py`의 docstring에 있는 `srun ... scripts/run_stt_noise.py`
경로 언급은 주석이라 동작에는 영향 없습니다.

## 8. 알려진 특이사항 (인수인계 시 헷갈리기 쉬운 것)

- **speechbrain 모델**: VAD 적용 후에도 RTF 7.7~14.8(실시간 기준 미달), CER 2.39~3.25로
  실사용 부적합 판정. 계획서 "임의 튜닝 금지" 원칙에 따라 beam size 등은 조정하지 않았음
  (`STT_연구계획서_CLI.md` 5-2절).
- **run_stt.py는 재실행 안전**: 출력 txt가 이미 있으면 건너뛰므로, 다시 돌리려면 해당
  `results/stt_outputs/{태그}/clip{n}.txt`를 먼저 지워야 합니다.
- **평가 지표는 휴리스틱 근사치**: 키워드 인식률·순서바뀜은 토큰 등장 기반 자동 판정이라,
  최종 판단은 `results/qualitative_review.csv` 수동 채점 병행이 전제입니다.

## 9. 관련 문서 위치

| 문서 | 경로 |
|---|---|
| 연구계획서 + 실행 중 변경 이력 (5절, 6절이 특히 중요) | `STT_연구계획서_CLI.md` |
| STT 모델 비교 최종 보고서 | `results/final_report.md`, `results/final_report_3.2.3_음성인식및발화정제.md` |
| raw vs vad 결론 | `results_raw_vs_vad/conclusion.md`, `comparison.csv` |
| VAD 방식 비교 보고서 | `results_vad_compare/final_vad_report.md` |
| 노이즈 강건성 실험 설계·재현법·보고서 | `results_noise_robustness/README.md`, `noise_robustness_report.md` |
| raw 조건(무VAD) 백업 결과 | `results_novad_backup/` |
