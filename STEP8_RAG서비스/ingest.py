# -*- coding: utf-8 -*-
"""
STEP7_DB 산출물(암묵지 JSON, 스키마 v1.3)을 Vector DB(Qdrant)에 적재하는 스크립트.

베이스: voice-rag-upload/ingest.py (raw_json 통째 저장 + embed-mode 실험 스위치).
이식: STEP7_DB/preflight.py 원칙(무거운 임베딩 모델 로드 전 프리플라이트 통과 필수),
      스키마 불일치 파일은 건너뛰고 경고만(관용적 실패), timestamp 범위 검증,
      적재 후 assert로 결과 검증.

사용법:
    .venv/bin/python3 ingest.py                          # config.py의 input_dir(STEP7_DB/gold_records) 사용
    .venv/bin/python3 ingest.py --input <다른 폴더>
    EMBED_MODE=insight .venv/bin/python3 ingest.py        # tacit_insight만 임베딩(비교 실험)
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

# 무거운 임베딩 모델/DB 클라이언트는 프리플라이트 통과 후에만 import한다
# (STEP3/STEP6/STEP7_DB와 동일 원칙 — docs/실행전_방어_체크리스트.md).
from preflight import run_preflight

TIME_RE_GROUPS = 3  # "HH:MM:SS"


def parse_timestamp(ts):
    """'HH:MM:SS' -> 총 초. 형식이 잘못되면 None. (STEP7_DB search_tacit.py와 동일 규칙)"""
    if not ts:
        return None
    parts = ts.strip().split(":")
    if len(parts) != TIME_RE_GROUPS:
        return None
    try:
        h, mi, s = map(int, parts)
    except ValueError:
        return None
    return h * 3600 + mi * 60 + s


def validate_timestamps(entry: dict) -> list[str]:
    """diagnostic_steps의 timestamp가 source.clip_start~clip_end 범위 밖이면 경고만 반환
    (탈락시키지 않음 — STEP7_DB validate_timestamps와 동일 규칙)."""
    warnings: list[str] = []
    source = entry.get("metadata", {}).get("source", {})
    clip_start = parse_timestamp(source.get("clip_start"))
    clip_end = parse_timestamp(source.get("clip_end"))
    if clip_start is None or clip_end is None:
        warnings.append(f"id={entry.get('id')}: clip_start/clip_end 형식 오류 — timestamp 검증 불가")
        return warnings

    for step in entry.get("knowledge", {}).get("diagnostic_steps", []):
        ts = parse_timestamp(step.get("timestamp"))
        if ts is None:
            warnings.append(f"id={entry.get('id')} step {step.get('order')}: timestamp 없음/형식 오류")
            continue
        if not (clip_start <= ts <= clip_end):
            warnings.append(
                f"id={entry.get('id')} step {step.get('order')}: timestamp {step.get('timestamp')}가 "
                f"clip 범위({source.get('clip_start')}~{source.get('clip_end')}) 밖"
            )
    return warnings


def build_document(entry: dict, mode: str) -> str:
    """임베딩 대상 텍스트 구성. mode에 따라 실험 비교(서시은 ingest.py 그대로)."""
    k = entry["knowledge"]
    m = entry["metadata"]
    if mode == "insight":
        return k["tacit_insight"]
    steps = " / ".join(s["action"] for s in k.get("diagnostic_steps", []))
    return (
        f"작업: {m['task']}. 상황: {k['situation']}. "
        f"노하우: {k['tacit_insight']} 이유: {k.get('reasoning', '')} "
        f"절차: {steps}. 키워드: {', '.join(m.get('keywords', []))}"
    )


def load_entries(input_path: Path) -> list[dict]:
    """input_path 안의 JSON을 전부 읽는다. 스키마가 안 맞는 파일은 건너뛰고 경고만
    남긴다 — 하나 잘못됐다고 나머지까지 막히면 안 되므로(STEP3/STEP6/STEP7_DB와
    동일한 "관용적 실패" 원칙)."""
    files = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    entries = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            candidates = data if isinstance(data, list) else [data]
            for entry in candidates:
                # 필수 필드 존재 확인(없으면 KeyError로 이 파일만 건너뜀)
                _ = entry["id"], entry["knowledge"]["tacit_insight"], entry["metadata"]["task"]
                warnings = validate_timestamps(entry)
                for w in warnings:
                    print(f"      [WARN][timestamp_validity] {w}")
                entries.append(entry)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"      [WARN] '{f}' 건너뜀 — 스키마 불일치: {e}")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="암묵지 JSON 폴더(기본: config.py의 STEP7_DB/gold_records)")
    ap.add_argument("--db-path", default=None, help="Qdrant 경로(기본: config.py의 qdrant_path)")
    ap.add_argument("--embed-mode", choices=["full", "insight"], default=None,
                     help="기본: config.py/EMBED_MODE 환경변수")
    args = ap.parse_args()

    run_preflight()

    from config import CONFIG
    from sentence_transformers import SentenceTransformer

    input_dir = Path(args.input) if args.input else Path(CONFIG.input_dir)
    embed_mode = args.embed_mode or CONFIG.embed_mode
    coll_name = f"tacit_knowledge_{embed_mode}"

    entries = load_entries(input_dir)
    print(f"[1/3] 암묵지 {len(entries)}건 로드 (입력: {input_dir})")
    if not entries:
        raise FileNotFoundError(f"'{input_dir}'에서 유효한 암묵지 JSON을 하나도 못 읽었습니다.")

    print(f"[2/3] 임베딩 모델 로드: {CONFIG.embedding_model_name} (device={CONFIG.device}, embed_mode={embed_mode})")
    model = SentenceTransformer(CONFIG.embedding_model_name, device=CONFIG.device)
    docs = [build_document(e, embed_mode) for e in entries]
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)

    def _payload(i, e):
        # STEP7_DB의 verification.routing/confidence는 accept_only 필터링에 쓰므로
        # (raw_json 안에 묻히면 DB 필터로 못 거르니) 별도 키로도 저장해둔다.
        p = {
            "doc_id": e["id"],
            "document": docs[i],
            "task": e["metadata"]["task"],
            "video_id": e["metadata"]["source"]["video_id"],
            "clip_start": e["metadata"]["source"]["clip_start"],
            "clip_end": e["metadata"]["source"]["clip_end"],
            "routing": e.get("verification", {}).get("routing"),
            "confidence": (e.get("verification", {}).get("confidence") or {}).get("score"),
            "raw_json": json.dumps(e, ensure_ascii=False),  # 원본 전체 보존 → LLM 컨텍스트용
        }
        return p

    # 2026-07-12 백엔드 분기(config.vector_backend — chroma 기본 / qdrant 롤백 보존)
    if CONFIG.vector_backend == "chroma":
        import chromadb
        from chromadb.config import Settings

        db_path = args.db_path or CONFIG.chroma_path
        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False))
        try:
            client.delete_collection(coll_name)  # 재적재 = 전체 교체(기존 qdrant 동작과 동일)
        except Exception:
            pass
        collection = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
        collection.upsert(
            ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, e["id"])) for e in entries],
            embeddings=[embeddings[i].tolist() for i in range(len(entries))],
            # chroma metadata는 스칼라만 + None 불가 → None 값 키는 제외(search의 .get이 None 처리)
            metadatas=[{k: v for k, v in _payload(i, e).items() if v is not None}
                       for i, e in enumerate(entries)],
        )
        count = collection.count()
    else:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        db_path = args.db_path or CONFIG.qdrant_path
        client = QdrantClient(path=db_path)
        if client.collection_exists(coll_name):
            client.delete_collection(coll_name)
        client.create_collection(
            coll_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
        )
        client.upsert(
            collection_name=coll_name,
            points=[
                PointStruct(
                    # Qdrant는 문자열 ID 불가(정수/UUID만) → 원본 id에서 UUID 파생
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, e["id"])),
                    vector=embeddings[i].tolist(),
                    payload=_payload(i, e),
                )
                for i, e in enumerate(entries)
            ],
        )
        count = client.count(collection_name=coll_name, exact=True).count

    # 스모크 테스트: "에러 없이 끝남"만으로 통과 처리하지 않는다 — 실제 개수까지 확인한다
    # (docs/실행전_방어_체크리스트.md 5번 원칙).
    print(f"[3/3] 적재 완료({CONFIG.vector_backend}) → 컬렉션 '{coll_name}' ({db_path}/) — "
          f"이번 적재 {len(entries)}건 / 컬렉션 전체 {count}건")
    assert count > 0, "적재 후 컬렉션이 비어있음 — 적재 실패"


if __name__ == "__main__":
    main()
