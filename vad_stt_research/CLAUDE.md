# CLAUDE.md — VAD STT 연구 프로젝트 컨텍스트

이 파일은 Claude Code가 대화를 시작할 때 자동으로 읽는 컨텍스트 파일입니다.
새 Claude 세션을 시작하면 이 파일과 PROGRESS_REPORT.md를 먼저 읽으세요.

---

## 전체 연구 로드맵 (이 파일의 위치)

```
[Stage 1] stt_comparison_research/ — STT 모델 비교 → 최선 모델 선정
               ↓
[Stage 2] vad_stt_research/ — 이 연구
  Phase 1 (현재): 단일 화자 롱폼에서 Silero VAD 전처리 효과 정량화
  Phase 2 (예정): 다중 화자 — PyAnnote Diarization + Stage 1 선정 모델로 화자별 전사
```

## 한 줄 요약 (Phase 1)

롱폼 오디오(1시간+)에 VAD 전처리를 붙이면 빨라지고 정확해지는지,
그리고 빨라진 게 VAD 덕인지 배치 추론 덕인지까지 축을 갈라 정량화하는 실험.

---

## 실험 조건 3가지

- **A**: Vanilla — faster-whisper 기본값 그대로, VAD 없음 (대조군)
- **A′**: 배치만 — 통일 디코딩 파라미터 적용, VAD 없음 (배치 효과 분리)
- **B**: VAD + 배치 — Silero VAD → 청크 분리 → faster-whisper (제안 파이프라인)

비교 축: A→A′ = 배치 효과 / A′→B = VAD 순수 효과 / A→B = 전체 효과

---

## 현재 진행 상태 (2026-06-24 기준)

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | configs 세팅 | ✅ |
| 2 | 버그 수정 | ✅ |
| 3 | 파이프라인 전체 구현 | ✅ |
| 4 | compute_silence_ratio → metadata.csv | ✅ 토이셋 기준 |
| 5 | run_experiment → results.csv | ✅ smoke test + YouTube 실험 완료 |
| 6 | statistical_tests + breakeven_analysis | 🔲 데이터 부족 |
| 7 | plot_generators.py (5종 시각화) | 🔲 |

실험 결과 보고서:
- `results/SMOKE_TEST_REPORT.md` — 토이셋 (유튜브 2개, 15분 이하, ground truth 없음)
- `results/YT_TEST_REPORT.md` — xsbdRlpLYhc (세바시 72.9분, 수동 자막 ground truth 포함, WER/CER 최초 측정)

---

## 환경 설정 (반드시 숙지)

**실행 환경**: Python 3.13, Anaconda, RTX 2080 8GB × 2

**실험 실행 전 필수 환경변수 설정**:
```bash
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
```
이유: 시스템에 `libcublas.so.13`만 있고 CTranslate2는 `.so.12`를 요구함.
`nvidia-cublas-cu12` pip 패키지로 해결. 매 세션마다 설정 필요.

**실험 실행 명령어**:
```bash
cd /home/piai/project-ai/vad_stt_research
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
PYTHONPATH=/home/piai/project-ai/vad_stt_research python scripts/run_experiment.py \
  --metadata data/metadata.csv \
  --config configs/experiment_config.yaml \
  --output results/raw/results.csv \
  --repeats 3
```

---

## 다음 할 일 (우선순위 순)

### Phase 1 (단일 화자 롱폼) — 진행 중

1. **AI Hub 데이터 준비** (신청 완료, 승인 대기)
   - 목표: low_silence 10개 + high_silence 10개, 각 60분 이상
   - ground_truth JSON 형식: `{"text": "전체 텍스트", "segments": [{"start": 0.0, "end": 2.5}]}`
   - 저장 위치: `data/raw/*.wav`, `data/ground_truth/{file_id}.json`

2. **정식 실험 실행**
   ```bash
   python scripts/compute_silence_ratio.py data/raw/ --output data/metadata.csv
   python scripts/run_experiment.py --repeats 3
   ```

3. **통계 분석 실행**
   ```bash
   # statistical_tests.py, breakeven_analysis.py 구현 완료
   # results.csv 로드 후 run_all_comparisons() 호출
   ```

4. **plot_generators.py 구현** (데이터 수집 후 착수)
   - 5종 시각화: Grouped Bar / Waterfall / Scatter+회귀 / Multi-line / Timeline

### Phase 2 (다중 화자 Diarization) — STT Stage 1 완료 후 착수

- `stt_comparison_research/`에서 최선 모델 확정 후 STT 백엔드로 투입
- PyAnnote Audio를 화자 분리 엔진으로 사용 (VAD 내장 → Silero VAD와 역할 중복 없음)
- 평가 지표: DER(Diarization Error Rate), 화자별 WER/CER

---

## 알려진 버그 및 수정 이력

| 항목 | 내용 |
|------|------|
| `temperature_increment_on_fallback` | faster-whisper 미지원 파라미터. `temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` 리스트로 수정 완료 |
| `libcublas.so.12 not found` | `nvidia-cublas-cu12` 설치 + LD_LIBRARY_PATH 설정으로 해결. 매 세션마다 export 필요 |
| `pipeline/vad/__init__.py` eager import | lazy import (`importlib`) 방식으로 수정 완료. SpeechSegment, BaseVAD 패키지 레벨 노출 |
| PYTHONPATH 미설정 | 스크립트 실행 시 `PYTHONPATH=/home/piai/project-ai/vad_stt_research` 명시 필요 |

---

## 주요 파일 위치

```
vad_stt_research/
├── CLAUDE.md                          # 이 파일
├── PROGRESS_REPORT.md                 # 팀 공유용 전체 진행 보고서
├── configs/experiment_config.yaml     # 모든 실험 파라미터 (수정 금지 원칙)
├── data/
│   ├── raw/                           # 오디오 WAV (git 제외)
│   ├── ground_truth/                  # {file_id}.json
│   └── metadata.csv                   # 무음 비율 사전 계산 결과
├── results/
│   ├── SMOKE_TEST_REPORT.md           # 토이셋 smoke test 결과 (GT 없음)
│   ├── YT_TEST_REPORT.md              # xsbdRlpLYhc YouTube 실험 결과 (GT 있음)
│   └── raw/results.csv                # 실험 결과 (git 제외)
├── pipeline/vad/__init__.py           # get_vad() 팩토리 — 여기서 엔진 선택
├── experiments/condition_a_prime.py   # DECODING_PARAMS_UNIFIED 정의 위치
└── scripts/run_experiment.py          # 메인 실행 진입점
```

---

## 협업 규칙

1. 한 번에 하나의 파일 또는 태스크만 진행
2. 전체 코드를 한 번에 쏟아내지 않음
3. 코드 주석에 이모티콘 사용 금지
4. 테스트 성공 확인 후에만 다음 단계로 진행
