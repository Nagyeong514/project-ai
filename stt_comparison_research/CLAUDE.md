# CLAUDE.md — STT 엔진 비교 연구 컨텍스트

새 세션을 시작하면 이 파일을 먼저 읽으세요.

---

## 한 줄 요약

한국어 STT 6개 모델(자체구축 4종 + 상용 API 2종)을 동일 오디오·동일 지표로 비교해
폐쇄망에서 채택 가능한 최선 모델을 정량적으로 선정하는 연구.

---

## 비교 대상 (6-arm)

| 키 | 모델 | 엔진 | 채택 가능 |
|----|------|------|----------|
| faster_whisper_large_v3 | large-v3 | faster-whisper | ✅ |
| faster_whisper_large_v3_turbo | large-v3-turbo | faster-whisper | ✅ |
| whisper_medium_ko | seastar105/whisper-medium-ko-zeroth | faster-whisper (CT2 변환) | ✅ |
| kospeech | KsponSpeech LAS checkpoint | kospeech 패키지 | ⚠️ 하한선 |
| clova | CLOVA Speech API | REST | ✕ 상용 상한선 |
| kakao | Kakao Speech API | REST | ✕ 상용 상한선 |

---

## 핵심 원칙

- **단일 변수**: STT 엔진만 변수. VAD 미적용, 동일 오디오, 동일 정규화 규칙.
- **Whisper 계열 공정 비교**: 동일 faster-whisper 엔진에 가중치만 교체.
- **API RTF**: 네트워크 지연 포함 → 별도 표기, 자체구축과 직접 비교 금지.
- **주지표 CER**, 보조 WER (한국어 교착어 특성 반영).

---

## 환경

- Python 3.13, Anaconda, RTX 2080 8GB × 2
- compute_type: int8_float16
- 매 세션 필수:
  ```bash
  export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
  ```

---

## 실험 실행

```bash
cd /home/piai/project-ai/stt_comparison_research
export LD_LIBRARY_PATH="..."

# 1. 오디오 준비
PYTHONPATH=$(pwd) python scripts/download_audio.py \
  --url "https://youtu.be/..." --file-id F01 --type A

# 2. GT 준비
PYTHONPATH=$(pwd) python scripts/prepare_ground_truth.py \
  --url "https://youtu.be/..." --file-id F01
# 또는 로컬 SRT: --srt path/to/sub.srt --file-id F01

# 3. 실험 실행 (전체 6-arm)
PYTHONPATH=$(pwd) python scripts/run_experiment.py \
  --metadata data/metadata.csv \
  --config configs/experiment_config.yaml \
  --output results/raw/results.csv \
  --kospeech-ckpt /path/to/las_checkpoint.pt

# API 건너뛰고 자체구축만 먼저:
  --skip-api
```

---

## 주요 파일

```
stt_comparison_research/
├── CLAUDE.md
├── configs/experiment_config.yaml      # 파라미터 고정 (수정 금지)
├── data/
│   ├── raw/                             # WAV (git 제외)
│   ├── ground_truth/{file_id}.txt       # 수동 자막 원문
│   └── metadata.csv                     # file_id, type, duration, path
├── pipeline/stt/
│   ├── base.py                          # STTResult, BaseSTT ABC
│   ├── faster_whisper_runner.py         # large-v3 / turbo / medium-ko 공용
│   ├── kospeech_runner.py               # Kospeech 래퍼
│   └── api_runner.py                    # CLOVA / Kakao REST
├── evaluation/
│   ├── normalizer.py                    # 정규화 규칙 (공정 비교 핵심)
│   └── metrics.py                       # CER, WER
├── experiments/run_all_models.py        # 6-arm RTF 측정 + 평가
├── analysis/
│   ├── statistical_tests.py             # Wilcoxon + gap 테이블
│   └── plot_generators.py               # 3종 시각화
└── scripts/run_experiment.py            # 메인 진입점
```

---

## 다음 할 일 (우선순위)

1. **데이터 준비**: 낭독(A) 2~3편 + 대화(B) 2~3편 수동 자막 영상 선정 → download_audio + prepare_ground_truth
2. **Kospeech checkpoint 확보**: https://github.com/sooftware/kospeech 에서 KsponSpeech LAS 모델 다운로드
3. **환경 변수 확인**: CLOVA_API_KEY, KAKAO_API_KEY .env에 입력
4. **smoke test**: 파일 1개 + --skip-api로 faster-whisper 3종만 먼저 실행
5. **전체 실행**: 모든 파일 × 6-arm
6. **분석**: statistical_tests.py → Wilcoxon / plot_generators.py → 3종 시각화

---

## 알려진 이슈 (예상)

| 항목 | 내용 |
|------|------|
| medium-ko CT2 변환 | seastar105 모델은 HF에 CT2 포맷이 없을 수 있음 → `ct2-transformers-converter` 로 변환 필요 |
| Kospeech vocab | `_decode_output` 이 token index 반환 시 vocab 파일 필요 — checkpoint과 함께 배포 여부 확인 |
| Kakao API 응답 포맷 | 실제 응답 구조 확인 후 `api_runner.py` 수정 필요 |

---

## VAD 연구와의 관계

본 연구에서 선정된 STT(CER 최소 자체구축 모델)가 `vad_stt_research`의 고정 백엔드로 투입됨.
두 실험은 변수를 섞지 않기 위해 분리 운영.
