# VAD 모델 비교 (Silero vs WebRTC)

Phase 1(`../vad_stt_research/`) 후속 연구. 계획서: `../VAD모델비교계획서.md`

**질문:** VAD를 쓴다면 어느 엔진인가 — **Silero**(신경망) vs **WebRTC**(GMM)를
**실행비용 · 검출정확도 · downstream(STT) 영향** 3축으로 비교.

---

## 설계

- **3조건:** `reference`(VAD 없음=Phase1 A′) / `silero` / `webrtc`
- **단일 변수:** VAD 엔진만 변동. STT(large-v3-turbo)·청크추출·평가·디코딩 파라미터·후처리(padding/merge/split)는 Phase 1 재사용 → 공정 비교.
- **데이터:** Phase 1 10파일 (`../vad_stt_research/data/metadata_10.csv` + `ground_truth/`)

## 측정 지표 (→ `results/comparison.csv`)

| 축 | 컬럼 |
|----|------|
| 실행비용 | `vad_rtf` (VAD 처리/오디오, warmup1+3평균) |
| 검출정확도 | `precision` `recall` `f1` (프레임 단위 vs VTT GT), `n_chunks` `avg_chunk_s` (과분할) |
| downstream | `wer` `cer` `hallucination_per_hour`, `stt_rtf` |

> 메모리(peak)는 계획서 4.1 항목이나 GPU(Silero)/CPU(WebRTC) 공정 측정 이슈로 **현재 보류** — 추후 추가 예정.

## 실행

```bash
cd vad_model_comparison
export LD_LIBRARY_PATH="/home/piai/anaconda3/lib/python3.13/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

# 스모크 (앞 120초, 1파일)
PYTHONPATH=. python run_comparison.py --file_ids H03 --max_seconds 120 --output results/smoke.csv

# 전체 10파일
PYTHONPATH=. python run_comparison.py --output results/comparison.csv
```

python 경로: `/home/piai/anaconda3/bin/python` (PATH에 없음).

## 구조

```
vad_model_comparison/
├── engines/
│   ├── __init__.py        # get_engine("silero"|"webrtc") + Phase1 경로 주입
│   └── webrtc_vad.py      # WebRTC 엔진 (BaseVAD 상속, 후처리 공유)
├── metrics/detection_f1.py # speech/non-speech 프레임 F1
├── run_comparison.py      # 3조건 러너 (incremental 저장)
└── results/comparison.csv # 결과
```
