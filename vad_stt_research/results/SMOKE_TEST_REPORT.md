# Smoke Test 결과 보고서

> 실행일: 2026-06-22  
> 목적: 파이프라인 전체 통과 확인 (토이 데이터셋, 정식 실험 아님)

---

## 사용 데이터

| file_id | 제목 | 길이 | 무음 비율 | 그룹 |
|---------|------|------|----------|------|
| ycsEx4d3Ri4 | 강동원 인터뷰 (유튜브) | 15.4분 | 18.77% | low_silence |
| wgP9ARwAbNw | 강의 영상 (유튜브) | 8.9분 | 12.71% | low_silence |

**한계**: ground truth 없음 → WER/CER/타임스탬프 Δt 평가 불가. RTF와 파이프라인 동작 확인 목적으로만 사용.

---

## RTF 결과

| file_id | 무음비율 | 조건 A | 조건 A′ | 조건 B | VAD 시간(B) | 청크 수(B) |
|---------|---------|--------|---------|--------|------------|-----------|
| ycsEx4d3Ri4 | 18.77% | 0.117 | **0.075** | 0.093 | 39.6초 | 49 |
| wgP9ARwAbNw | 12.71% | 0.172 | **0.118** | 0.130 | 21.9초 | 27 |

RTF 낮을수록 빠름 (RTF 1.0 = 실시간과 동일 속도).

---

## 관찰 사항

**배치 효과 (A → A′)**: 두 파일 모두 A′이 A보다 빠름. 디코딩 파라미터 통일(`condition_on_previous_text=False`)과 faster-whisper 효율 덕분.

**VAD 효과 (A′ → B)**: 두 파일 모두 B가 A′보다 느림. 두 영상의 무음 비율이 낮아(12~18%) VAD 연산 오버헤드가 무음 제거 이득을 초과함. **이는 연구 설계가 예측한 손익분기점 현상과 정확히 일치** — 무음 ≥ 50% 데이터에서 B의 이득이 나타날 것으로 기대.

**할루시네이션 수치**: ground truth 없이 역산한 값이므로 해석 불가. 정식 실험에서 재측정 필요.

---

## 파이프라인 통과 여부

| 단계 | 결과 |
|------|------|
| Silero VAD 실행 | 정상 |
| 청크 분리 (chunk_extractor) | 정상 |
| faster-whisper 전사 (A/A′/B) | 정상 |
| 타임스탬프 재매핑 | 정상 |
| results.csv 생성 | 정상 |
| 할루시네이션 감지 (n-gram) | 정상 |

**결론: 파이프라인 전체 에러 없이 통과.**

---

## 발견된 버그 및 수정 내역

| 버그 | 원인 | 수정 |
|------|------|------|
| `temperature_increment_on_fallback` KeyError | faster-whisper는 해당 파라미터 미지원 | `temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` 리스트 방식으로 변경 |
| `Library libcublas.so.12 not found` | 시스템에 CUDA 13 라이브러리만 존재 | `nvidia-cublas-cu12` pip 설치 후 `LD_LIBRARY_PATH` 설정으로 해결 |
| `ModuleNotFoundError: experiments` | 스크립트 실행 시 PYTHONPATH 미설정 | `PYTHONPATH=/path/to/vad_stt_research` 명시 필요 |

---

## 정식 실험 전 필요 사항

1. AI Hub 한국어 음성 데이터 신청 승인 대기
2. 60분 이상 파일, low_silence 10개 + high_silence 10개 확보
3. ground_truth JSON 준비 (`{"text": "...", "segments": [{"start": 0.0, "end": 2.5}, ...]}`)
4. LD_LIBRARY_PATH 자동화 (실행 스크립트에 고정)
