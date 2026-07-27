# -*- coding: utf-8 -*-
"""
음성 RAG 질의응답 서버 (STEP8_RAG서비스 — 서시은 app.py 베이스 + 안나경 방어 이식).

파이프라인(2026-07-08 음성 계층 역머지 — 서시은 voice-rag v2, HANDOFF.md 참고):
  브라우저 녹음 → POST /transcribe (서버 STT, faster-whisper)
  → /ask 로직: search.search()(질문 임베딩+Qdrant 검색+임계값 필터)
  → [통과 못하면 즉시 반환, LLM 호출 안 함] → Ollama LLM 답변 생성
  → POST /tts (서버 TTS, Supertonic 3) → 브라우저 재생.

이중 방어(환각 금지):
  1) search.search()가 min_similarity_threshold 미만 결과를 이미 걸러냄 →
     전부 걸러지면 빈 리스트 반환 → 여기서 LLM을 아예 호출하지 않고 정직하게 실패 응답.
  2) 그래도 통과한 컨텍스트가 부실할 수 있으니 시스템 프롬프트로도 "근거 부족하면
     부족하다고 말하라"를 이중으로 못박는다.

실행:
    OLLAMA_MODELS=/home/ai_user/team_a2/.ollama/models ~/.local/ollama/bin/ollama serve &   # 최초 1회
    # (2026-07-12: qwen3:14b는 공유 경로에만 있음 — 구경로 ~/.local/ollama/data는 qwen2.5 롤백용)
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
from fastapi import FastAPI, File, Response, UploadFile
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

# ── 음성 계층 (서시은 voice-rag v2 역머지) ─────────────────────────────────
# TTS 발음 사전: 영어 약어를 알파벳 낭독("알에이엠") 대신 관용 발음으로 교정
TTS_PRONUNCIATION = {
    "RDIMM": "알딤",
    "DIMM": "딤",
    "RAM": "램",
    "ROM": "롬",
    "BIOS": "바이오스",
    "CMOS": "씨모스",
    "SATA": "사타",
    "LAN": "랜",
    "FAN": "팬",
    "HEATSINK": "히트싱크",
}


def normalize_tts_text(text: str) -> str:
    import re
    for eng, kor in TTS_PRONUNCIATION.items():
        text = re.sub(rf"\b{eng}\b", kor, text, flags=re.IGNORECASE)
    return text


print("모델 로딩 중…")
embedder, qdrant_client = se.load_backend(CONFIG)

# 무거운 음성 모델은 preflight 통과 후 여기서 로드(임베딩 모델과 동일 원칙).
from supertonic import TTS as SupertonicTTS  # noqa: E402

tts_engine = SupertonicTTS(model=CONFIG.tts_model, intra_op_num_threads=8)
tts_voice = tts_engine.get_voice_style(CONFIG.tts_voice)

import ctranslate2  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

if ctranslate2.get_cuda_device_count() > 0 and CONFIG.stt_device != "cpu":
    whisper_model = WhisperModel(CONFIG.stt_model, device="cuda", compute_type="int8_float16")
else:
    # CPU 전사 병목 완화: 기본 4스레드 → 16스레드 (로그인 노드 40코어, 서시은 실측)
    whisper_model = WhisperModel(CONFIG.stt_model, device="cpu", compute_type="int8", cpu_threads=16)

print(
    f"준비 완료: collection={CONFIG.qdrant_collection}, top_k={CONFIG.top_k_default}, "
    f"threshold={CONFIG.min_similarity_threshold}, accept_only={CONFIG.accept_only_default}, "
    f"llm={CONFIG.llm_model}, stt={CONFIG.stt_model}, tts={CONFIG.tts_model}/{CONFIG.tts_voice}"
)

app = FastAPI(title="Tacit Knowledge Voice RAG (STEP8 통합)")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    # ①질문 임베딩 ②Vector DB 검색(chroma 기본/qdrant 롤백 — config.vector_backend) ③유사도 임계값 필터 (search.py)
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
    payload = {
        "model": CONFIG.llm_model,
        "stream": False,
        "keep_alive": -1,  # 모델을 VRAM에 상주시켜 유휴 후 재로딩(~30초) 제거
        "options": {"num_predict": 256},  # 답변 길이 상한 → 생성 시간 단축 (음성 답변은 짧을수록 UX도 좋음)
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[검색된 암묵지]\n{context}\n\n[신입의 질문]\n{req.question}"},
        ],
    }
    # 2026-07-12: qwen3 계열(hybrid thinking)은 사고 모드를 꺼야 지연/사고문 유출이 없다
    # (실DB 벤치에서 think:false 7/7 정상 실측). qwen2.5 등 비지원 모델에 보내면 400이
    # 날 수 있어 조건부로만 추가(STEP5 llm_fusion의 "qwen3" 매칭 규칙과 동일 계열).
    if CONFIG.llm_model.lower().startswith(("qwen3:", "qwen3.")):
        payload["think"] = False
    resp = requests.post(
        CONFIG.ollama_url,
        json=payload,
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


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
def tts(req: TTSRequest):
    import io

    import numpy as np
    import scipy.io.wavfile as wavfile

    waveform, _ = tts_engine.synthesize(
        normalize_tts_text(req.text), voice_style=tts_voice, lang="ko", total_steps=CONFIG.tts_steps
    )
    pcm = np.clip(waveform.squeeze(), -1.0, 1.0)
    buf = io.BytesIO()
    wavfile.write(buf, tts_engine.sample_rate, (pcm * 32767).astype(np.int16))
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        # beam_size=1: 기본값(5) 대비 수 배 빠름, 짧은 질의에서 정확도 차이 미미
        # vad_filter: 앞뒤 무음 구간을 잘라 전사 시간 단축 (서시은 v2 그대로)
        segments, _ = whisper_model.transcribe(tmp_path, language="ko", beam_size=1, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
    res = ask(AskRequest(question=text))
    res["question"] = text
    return res


@app.get("/")
def index():
    # 음성 UI(서시은 v2). 텍스트 전용 구화면은 /static/index.html 로 접근 가능.
    return FileResponse("static/메인화면.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
