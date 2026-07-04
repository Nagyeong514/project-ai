# -*- coding: utf-8 -*-
"""
음성 RAG 질의응답 서버 (신입 지식 활용 파트).

파이프라인: 브라우저 STT → POST /ask → 질문 임베딩 → ChromaDB 검색 → LLM 답변 → 브라우저 TTS

실행:
    export ANTHROPIC_API_KEY=sk-ant-...
    python ingest.py                      # 최초 1회, DB 구축
    uvicorn app:app --reload --port 8000  # http://localhost:8000 접속

하이퍼파라미터 (환경변수로 실험):
    TOP_K=3            검색 결과 수 (비교: 2 / 3 / 5)
    SIM_THRESHOLD=0.35 코사인 유사도 컷오프 (비교: 0 / 0.35 / 0.5)
    COLLECTION=tacit_knowledge_full  (비교: _insight)
"""
import json
import os

import requests
from qdrant_client import QdrantClient
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

TOP_K = int(os.getenv("TOP_K", 3))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.35))
COLLECTION = os.getenv("COLLECTION", "tacit_knowledge_full")
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

SYSTEM_PROMPT = """너는 조립 현장의 신입 작업자를 돕는 음성 작업 보조 AI다.
아래 [검색된 암묵지]만 근거로 답하고, 근거가 부족하면 솔직히 부족하다고 말하라.
규칙:
- 첫 1~2문장에 핵심 답을 먼저 말한다 (음성으로 읽힐 것을 가정, 간결한 존댓말)
- 절차가 있으면 번호 단계로 나열
- 마지막 줄에 "출처: 영상ID (시작~끝 타임스탬프)" 표기
- 마크다운 강조 기호 사용 금지, 일반 텍스트로만"""

print("모델 로딩 중…")
embedder = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(path=os.getenv("DB_PATH", "qdrant_db"))
print(f"준비 완료: collection={COLLECTION}, top_k={TOP_K}, threshold={SIM_THRESHOLD}, llm={LLM_MODEL}")

app = FastAPI(title="Tacit Knowledge Voice RAG")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    # 1) 질문 임베딩 → 2) Vector DB 검색 (Qdrant는 cosine similarity를 score로 바로 반환)
    q_emb = embedder.encode([req.question], normalize_embeddings=True)[0].tolist()
    points = qdrant.query_points(COLLECTION, query=q_emb, limit=TOP_K).points

    retrieved = []
    for p in points:
        if p.score < SIM_THRESHOLD:
            continue
        retrieved.append(
            {
                "id": p.payload["doc_id"],
                "similarity": round(p.score, 3),
                "entry": json.loads(p.payload["raw_json"]),
            }
        )

    if not retrieved:
        return {
            "answer": "관련 암묵지를 찾지 못했어요. 부품명이나 증상을 포함해 다시 질문해주시겠어요?",
            "sources": [],
        }

    # 3) LLM 답변 생성 (Ollama)
    context = json.dumps(retrieved, ensure_ascii=False, indent=1)
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[검색된 암묵지]\n{context}\n\n[신입의 질문]\n{req.question}"},
            ],
        },
    )
    resp.raise_for_status()
    answer = resp.json()["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": [
            {
                "id": r["id"],
                "similarity": r["similarity"],
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
