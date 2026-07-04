# 현장 암묵지 어시스턴트 — 신입 (지식 활용) 파트

음성 질문 → STT → Vector DB 검색(RAG) → LLM 답변 생성 → TTS 음성 출력.
전문가 영상에서 추출한 암묵지 JSON(스키마 v1.3)을 지식 베이스로 사용한다.

## 아키텍처

```
[브라우저]                        [FastAPI 서버]
 마이크 ─ Web Speech API(STT) ──▶ POST /ask
                                   ├─ ① 질문 임베딩 (bge-m3)
                                   ├─ ② ChromaDB top-k 검색 (cosine)
                                   ├─ ③ 유사도 임계값 필터
                                   └─ ④ Claude LLM 답변 생성 (출처 포함)
 스피커 ◀─ speechSynthesis(TTS) ── JSON {answer, sources}
```

설계 포인트 (발표용):
- **STT/TTS를 브라우저에서 처리** → 서버 비용 0원, API 키 불필요, 지연 최소화
- **원본 JSON 전체를 메타데이터로 보존** → LLM이 diagnostic_steps, 타임스탬프까지 활용해
  "출처: 영상ID (타임스탬프)"를 답변에 명시 → 신입이 원본 영상 클립으로 바로 이동 가능
- **유사도 임계값** → 관련 없는 질문에 억지 답변하지 않고 재질문 유도 (환각 방지)

## 설치 및 실행

```bash
pip install -r requirements.txt        # 최초 1회 (bge-m3 다운로드 ~2GB)
export ANTHROPIC_API_KEY=sk-ant-...    # LLM 답변 생성용

python ingest.py --input data/sample_knowledge.json   # ① 지식 DB 구축
uvicorn app:app --port 8000                           # ② 서버 실행
# → http://localhost:8000 접속, 마이크 버튼(Chrome 권장)
```

암묵지 JSON을 추가하려면 data/ 폴더에 넣고 `python ingest.py --input data/` 재실행.

## 하이퍼파라미터 비교 실험

| 변수 | 비교값 | 바꾸는 법 |
|---|---|---|
| top-k | 2 / 3 / 5 | `TOP_K=5 uvicorn app:app` |
| 유사도 임계값 | 0 / 0.35 / 0.5 | `SIM_THRESHOLD=0.5 ...` |
| 임베딩 대상 | insight만 vs 전체 | `python ingest.py --embed-mode insight` 후 `COLLECTION=tacit_knowledge_insight ...` |
| 임베딩 모델 | bge-m3 vs ko-sroberta | ingest.py의 EMBED_MODEL 변경 |

평가: `python evaluate.py --collection tacit_knowledge_full --top-k 3`
→ 정답 문서가 top-k에 포함되는 비율(Recall@k) 출력. EVAL_SET에 질문 추가 가능.

## 향후 확장 (스마트 글래스 연동)

Meta Wearables Device Access Toolkit(2025.12 개발자 프리뷰 공개)과 연동 시:
- 글래스 마이크(5-mic array)/스피커 접근 가능 (현재는 블루투스 프로파일 경유)
- 시선 영상(카메라 스트림) → VLM 상황 인식 → 질문 없이도 선제적 안내 가능
- Ray-Ban Display 글래스 디스플레이 출력 지원 문서화됨
- 단, 일반 개발자 배포는 2026년 GA 이후 가능 → 현 단계는 웹 기반 구현이 합리적
