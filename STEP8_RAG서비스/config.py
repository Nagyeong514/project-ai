# -*- coding: utf-8 -*-
"""
STEP8_RAG서비스 설정.

서시은(voice-rag-upload)의 "환경변수로 하이퍼파라미터 노출" 방식은 그대로 유지하되,
안나경(STEP7_DB)의 원칙대로 pydantic BaseModel로 감싸서 값이 이상하면(오타·범위 밖)
config를 import하는 시점에 즉시 예외로 죽는다(docs/실행전_방어_체크리스트.md 참고).

입력 데이터는 이 폴더 안에 복제하지 않고 STEP7_DB/gold_records를 그대로 참조한다
(요청사항: "다른 파일럿 데이터는 안 쓰고 스텝 7의 JSON을 기준으로").
"""
from __future__ import annotations

import os

from pydantic import BaseModel, field_validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(BaseModel):
    # ---------------- 입력 (STEP6 run6 accept = 실제 STEP2~6 파이프라인 산출물) ----------------
    # 2026-07-13 고정: 기본 입력을 gold_records(파일럿 예시 9건)에서 run6 accept
    # (input_motion_obs_run6_accept, 실제 파이프라인 산출물 6건 — 현재 서비스 DB의 출처)로 변경.
    # 인자 없이 ingest.py를 다시 돌려도 현재 chroma_db(run6_accept)를 파일럿셋으로 덮어쓰지 않게 함.
    # gold_records/다른 세트는 필요 시 `ingest.py --input <경로>`로 명시 지정.
    input_dir: str = os.path.join(BASE_DIR, "..", "STEP7_DB", "input_motion_obs_run6_accept")

    # ---------------- 임베딩 ----------------
    embedding_model_name: str = "BAAI/bge-m3"
    device: str = "cpu"  # 문서 9건 임베딩엔 GPU 불필요
    # 서시은의 embed-mode 실험 스위치 유지: full(상황+노하우+이유+절차+키워드) vs insight(단독)
    embed_mode: str = os.getenv("EMBED_MODE", "full")

    # ---------------- Vector DB 백엔드 ----------------
    # 2026-07-12 전환(지시): Qdrant(embedded) → ChromaDB. STEP7_DB와 동일 스위치 패턴.
    # 근거: 00_사전연구/07_VectorDB비교 실측(Recall 동일, 1만+ 건 Chroma 우위) — 상세는
    # STEP7_DB/config.py 주석 참고. 롤백: VECTOR_BACKEND=qdrant (qdrant_db 폴더 무수정 보존).
    vector_backend: str = os.getenv("VECTOR_BACKEND", "chroma")
    chroma_path: str = os.path.join(BASE_DIR, "chroma_db")

    # ---------------- Vector DB (Qdrant — 롤백용 보존) ----------------
    qdrant_path: str = os.path.join(BASE_DIR, "qdrant_db")
    # 서시은 방식대로 embed_mode별 컬렉션 분리(비교 실험용) — COLLECTION env로 강제 지정도 가능
    qdrant_collection: str = os.getenv("COLLECTION", "")  # ""면 ingest.py가 embed_mode로 자동 결정

    # ---------------- 검색(RAG) — 서시은의 환경변수 스위치 유지 ----------------
    top_k_default: int = int(os.getenv("TOP_K", "3"))
    # 2026-07-04: 서시은 원본 기본값(0.35)은 애매한 질의("오늘 점심 뭐 먹지", top1=0.372)를
    # 못 걸러냄(방어선2=시스템 프롬프트가 겨우 막음). STEP7_DB가 실측으로 검증한 값(무관 질의
    # baseline 0.30~0.36 vs 관련 질의 0.44+)으로 올려 방어선1 자체를 더 확실하게 만듦.
    min_similarity_threshold: float = float(os.getenv("SIM_THRESHOLD", "0.40"))
    # STEP7_DB의 --accept_only 훅과 동일 개념. 기본 꺼짐(진짜 STEP3 데이터 나오기 전까지).
    accept_only_default: bool = os.getenv("ACCEPT_ONLY", "false").lower() == "true"

    # ---------------- LLM (Ollama) ----------------
    # 2026-07-12: qwen2.5:14b-instruct → qwen3:14b 교체(지시). 근거: 실DB 벤치 추가 실측
    # (05_LLM모델링평가_RAG답변용/평가보고서_realdb_qwen3_14b.md — 7/7 성공, 4.95초/45.3tok/s,
    # 형식·어조 표류 0건) + STEP5 융합과 모델 계열 통일 + "VLM은 STEP4만" 정책(구 채택안
    # qwen3-vl:8b-instruct는 VLM이라 배제). qwen3:14b는 hybrid thinking 모델 — app.py가
    # think:false를 보낸다(안 보내면 사고 모드로 지연/유출). 태그는 공유 경로
    # (/home/ai_user/team_a2/.ollama/models)에 pull 완료(2026-07-11). 롤백값: "qwen2.5:14b-instruct"
    # (~/.local/ollama/data 구경로에도 있음 — 롤백 시 think 파라미터는 자동 미전송).
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:14b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    # ---------------- 음성 계층 (2026-07-08 서시은 voice-rag v2에서 역머지) ----------------
    # STT: faster-whisper. GPU 있으면 cuda/int8_float16, 없으면 cpu/int8(스레드 늘림).
    stt_model: str = os.getenv("STT_MODEL", "large-v3-turbo")
    stt_device: str = os.getenv("STT_DEVICE", "auto")  # auto|cuda|cpu
    # TTS: Supertonic 3. 합성 스텝 8→5로 약 35% 단축, 품질 차이 미미(서시은 실측).
    tts_model: str = os.getenv("TTS_MODEL", "supertonic-3")
    tts_voice: str = os.getenv("TTS_VOICE", "F1")
    tts_steps: int = int(os.getenv("TTS_STEPS", "5"))

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError(f"device는 'cuda' 또는 'cpu'여야 함 (받은 값: {v!r})")
        return v

    @field_validator("embed_mode")
    @classmethod
    def _validate_embed_mode(cls, v: str) -> str:
        if v not in ("full", "insight"):
            raise ValueError(f"embed_mode는 'full' 또는 'insight'여야 함 (받은 값: {v!r})")
        return v

    @field_validator("min_similarity_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_similarity_threshold는 0.0~1.0 범위여야 함 (받은 값: {v!r})")
        return v

    @field_validator("top_k_default")
    @classmethod
    def _validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"top_k_default는 1 이상이어야 함 (받은 값: {v!r})")
        return v

    @field_validator("stt_device")
    @classmethod
    def _validate_stt_device(cls, v: str) -> str:
        if v not in ("auto", "cuda", "cpu"):
            raise ValueError(f"stt_device는 'auto'/'cuda'/'cpu'여야 함 (받은 값: {v!r})")
        return v

    @field_validator("tts_steps")
    @classmethod
    def _validate_tts_steps(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tts_steps는 1 이상이어야 함 (받은 값: {v!r})")
        return v


CONFIG = Config()

# COLLECTION env를 안 줬으면 embed_mode로 컬렉션명 자동 결정(서시은 ingest.py와 동일 규칙)
if not CONFIG.qdrant_collection:
    CONFIG.qdrant_collection = f"tacit_knowledge_{CONFIG.embed_mode}"
