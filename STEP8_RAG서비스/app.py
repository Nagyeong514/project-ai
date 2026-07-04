# -*- coding: utf-8 -*-
"""
음성 RAG 질의응답 서버 (STEP8_RAG서비스 — 서시은 app.py 베이스 + 안나경 방어 이식).

파이프라인: 브라우저 STT → POST /ask → search.search()(질문 임베딩+Qdrant 검색+임계값
필터) → [통과 못하면 즉시 반환, LLM 호출 안 함] → Ollama LLM 답변 생성 → 브라우저 TTS.

이중 방어(환각 금지):
  1) search.search()가 min_similarity_threshold 미만 결과를 이미 걸러냄 →
     전부 걸러지면 빈 리스트 반환 → 여기서 LLM을 아예 호출하지 않고 정직하게 실패 응답.
  2) 그래도 통과한 컨텍스트가 부실할 수 있으니 시스템 프롬프트로도 "근거 부족하면
     부족하다고 말하라"를 이중으로 못박는다.

실행:
    OLLAMA_MODELS=~/.local/ollama/data ~/.local/ollama/bin/ollama serve &   # 최초 1회
    .venv/bin/python3 ingest.py                                # DB 구축(파트1)
    .venv/bin/uvicorn app:app --port 8000                      # http://localhost:8000

하이퍼파라미터 (환경변수, 서시은 방식 그대로):
    TOP_K, SIM_THRESHOLD, COLLECTION, EMBED_MODE, ACCEPT_ONLY, LLM_MODEL, OLLAMA_URL
"""
from __future__ import annotations

import json

# fastapi/uvicorn 자체는 "무거운 모델"이 아니라 가볍게 import 가능 — uvicorn이 `app:app`을
# 찾으려면 이 모듈 최상단에 app 객체가 있어야 하므로 여기서 import한다.
# 무거운 것(임베딩 모델, Qdrant 클라이언트)은 run_preflight() 통과 후에만 로드한다
# (STEP3/STEP6/STEP7_DB와 동일 원칙 — docs/실행전_방어_체크리스트.md).
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

from preflight import run_preflight

run_preflight()

from config import CONFIG
import search as se

SYSTEM_PROMPT = """너는 조립 현장의 신입 작업자를 돕는 음성 작업 보조 AI다.
아래 [검색된 암묵지]만 근거로 답하고, 근거가 부족하면 솔직히 부족하다고 말하라.
규칙:
- 첫 1~2문장에 핵심 답을 먼저 말한다 (음성으로 읽힐 것을 가정, 간결한 존댓말)
- 절차가 있으면 번호 단계로 나열
- 마지막 줄에 "출처: 영상ID (시작~끝 타임스탬프)" 표기
- 마크다운 강조 기호 사용 금지, 일반 텍스트로만"""

print("모델 로딩 중…")
embedder, qdrant_client = se.load_backend(CONFIG)
print(
    f"준비 완료: collection={CONFIG.qdrant_collection}, top_k={CONFIG.top_k_default}, "
    f"threshold={CONFIG.min_similarity_threshold}, accept_only={CONFIG.accept_only_default}, "
    f"llm={CONFIG.llm_model}"
)

app = FastAPI(title="Tacit Knowledge Voice RAG (STEP8 통합)")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    # ①질문 임베딩 ②Qdrant 검색 ③유사도 임계값 필터 (search.py, 파트2)
    retrieved = se.search(embedder, qdrant_client, CONFIG, req.question)

    # [방어선 1] 임계값을 넘는 결과가 하나도 없으면 LLM을 아예 호출하지 않는다.
    # 검색 실패 상태를 LLM에 넘기면 "그럴듯한 말"을 지어낼 위험이 있으므로
    # 여기서 즉시 정직한 실패 응답을 반환한다(docs/실행전_방어_체크리스트.md 5번 원칙과 동일 취지).
    if not retrieved:
        return {
            "answer": "관련 암묵지를 찾지 못했어요. 부품명이나 증상을 포함해 다시 질문해주시겠어요?",
            "sources": [],
        }

    # ③ LLM 답변 생성 (Ollama). [방어선 2] 시스템 프롬프트로 "검색된 근거만" 이중 강제.
    context = json.dumps(retrieved, ensure_ascii=False, indent=1)
    resp = requests.post(
        CONFIG.ollama_url,
        json={
            "model": CONFIG.llm_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[검색된 암묵지]\n{context}\n\n[신입의 질문]\n{req.question}"},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    answer = resp.json()["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": [
            {
                "id": r["id"],
                "similarity": r["similarity"],
                "routing": r["routing"],
                "task": r["entry"]["metadata"]["task"],
                "insight": r["entry"]["knowledge"]["tacit_insight"],
                "video": r["entry"]["metadata"]["source"]["video_id"],
                "clip": f'{r["entry"]["metadata"]["source"]["clip_start"]}–{r["entry"]["metadata"]["source"]["clip_end"]}',
            }
            for r in retrieved
        ],
    }


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
