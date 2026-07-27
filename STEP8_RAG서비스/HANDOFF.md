# Voice-RAG 인수인계 (서시은 → 안나경)

음성 질의응답 RAG 웹앱입니다. 파이프라인: 브라우저 녹음 → STT(faster-whisper) → 질문 임베딩(BAAI/bge-m3) → Qdrant 검색 → LLM 답변(ollama qwen2.5:14b) → TTS(Supertonic 3, F1 보이스).

## 바로 실행하기
보시
```bash
cd ~/team_a2/members/안나경/voice-rag
bash 서버시작.sh       # 기본 n5 노드. 비어있는지 sinfo로 먼저 확인, 다른 노드는 bash 서버시작.sh n6
```

- 뜨면 VS Code 포트 탭에서 **8001** 포워딩 → http://localhost:8001
- **8000 포트는 다른 사용자가 쓰고 있으니 절대 쓰지 말 것** (이 앱은 8001 고정)
- 종료: `fuser -k 8001/tcp && scancel <JOBID>` (JOBID는 서버시작.sh 출력에 나옴)

## 환경

- `.venv/`는 이미 세팅된 걸 그대로 가져온 것 → **재설치 불필요, 바로 실행됨** (import 확인 완료)
- 새로 만들어야 할 일이 생기면 `requirements.txt`(버전 고정됨) 사용. 단 **torchcodec은 재설치 금지** (기존에 문제 있었음)
- 모델 캐시(bge-m3, whisper, supertonic)는 팀 계정 `~/.cache`에, ollama는 `~/.local/ollama`에 있어서 그대로 공유됨

## 주요 파일

| 파일 | 설명 |
|---|---|
| `서버.py` | FastAPI 서버 (STT `/transcribe`, 질의 `/ask`, TTS `/tts`). 발음 사전(RAM→램 등)도 여기 |
| `static/메인화면.html` | 프론트엔드 — `/` 접속 시 서비스되는 화면 (orb 애니메이션, 자막, 마이크) |
| `지식DB구축.py` | 지식 DB(Qdrant) 구축 — `data/sample_knowledge.json` → `qdrant_db/` (이미 구축돼 있어 재실행 불필요) |
| `검색평가.py` | 검색 품질 평가 스크립트 |
| `서버시작.sh` | GPU 노드 할당 + ollama + 앱 서버 일괄 기동 |
| `REPORT_v2.md` | 프로젝트 보고서 (아키텍처·실험 결과 정리, 여기 먼저 읽으면 좋음) |

주로 고치게 될 건 `서버.py`(백엔드)와 `static/메인화면.html`(화면) 두 개.

## 튜닝 포인트 (환경변수)

- `TOP_K=3` 검색 결과 수, `SIM_THRESHOLD=0.35` 유사도 컷오프
- `TTS_VOICE=F1`, `TTS_STEPS=5` (8→5로 줄여 약 35% 단축, 품질 차이 미미)
- 현재 음성 입력→답변 재생까지 약 10초

궁금한 건 서시은에게 문의.
