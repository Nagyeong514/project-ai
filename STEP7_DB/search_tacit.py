# -*- coding: utf-8 -*-
"""
암묵지 검색(RAG) 파이프라인 — 신입 작업자의 증상 질의를 받아 accept된 암묵지를
검색하고 payload를 [핵심/이유/절차/출처] 템플릿으로 조립한다.

이번 단계는 템플릿 조립까지만 한다 — LLM 재작성(명장 말투 → 초보 눈높이)은
다음 단계로 미룸(코딩지시 4절 "하지 말 것"). 환각 금지 원칙: 답변은 검색된
payload 내용만 쓰고, 관련도가 낮으면 지어내지 않고 "못 찾았다"고 답한다.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")

NOT_FOUND_MESSAGE = "관련 노하우를 찾지 못했습니다. 다른 표현으로 다시 질문해보세요."


def parse_timestamp(ts: Optional[str]) -> Optional[int]:
    """'HH:MM:SS' -> 총 초. 형식이 잘못되면 None. (STEP6 rules.py의 동명 함수와 동일 규칙)"""
    if not ts:
        return None
    m = TIME_RE.match(ts.strip())
    if not m:
        return None
    h, mi, s = map(int, m.groups())
    return h * 3600 + mi * 60 + s


def _split_sentences(text: str) -> List[str]:
    """마침표 뒤 공백 기준 문장 분리(한국어 간이 분리 — 완벽할 필요 없음, 초보자 함정 문구
    추출용으로만 쓰인다)."""
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]


def extract_beginner_trap(*texts: str) -> Optional[str]:
    """tacit_insight/reasoning 중 '초보자'가 들어간 문장을 찾아 반환. 없으면 None."""
    for text in texts:
        for sentence in _split_sentences(text or ""):
            if "초보자" in sentence:
                return sentence
    return None


def validate_timestamps(payload: Dict[str, Any]) -> List[str]:
    """D4: source.clip_start <= 각 step.timestamp <= source.clip_end 검증.
    범위 밖이면 경고 문자열을 반환(STEP6 timestamp_validity와 같은 규칙 — 여기선
    탈락시키지 않고 경고만 남긴다, 검색 결과라 후보 자체를 폐기할 권한은 없음)."""
    warnings: List[str] = []
    clip_start = parse_timestamp(payload.get("clip_start"))
    clip_end = parse_timestamp(payload.get("clip_end"))
    if clip_start is None or clip_end is None:
        warnings.append(f"id={payload.get('id')}: clip_start/clip_end 형식 오류 — 검증 불가")
        return warnings

    for step in payload.get("diagnostic_steps", []):
        ts = parse_timestamp(step.get("timestamp"))
        if ts is None:
            # action_only여도 timestamp는 VLM ts로 채워지는 게 정상(설계 결정) — 없으면 경고.
            warnings.append(f"id={payload.get('id')} step {step.get('order')}: timestamp 없음/형식 오류")
            continue
        if not (clip_start <= ts <= clip_end):
            warnings.append(
                f"id={payload.get('id')} step {step.get('order')}: timestamp {step.get('timestamp')}가 "
                f"clip 범위({payload.get('clip_start')}~{payload.get('clip_end')}) 밖"
            )
    return warnings


def format_answer(payload: Dict[str, Any], score: float) -> str:
    """D3: payload를 정확히 [핵심/이유/절차/출처] 템플릿으로 조립. LLM 재작성 없음.

    action_only 스텝은 source_utterance가 null이므로 절차엔 action만 쓴다(코딩지시 5절).
    """
    lines = [f"[유사도 {score:.3f}]"]
    lines.append(f"핵심: {payload.get('tacit_insight', '')}")
    lines.append(f"이유: {payload.get('reasoning', '')}")
    lines.append("절차:")
    steps = sorted(payload.get("diagnostic_steps", []), key=lambda s: s.get("order", 0))
    for i, step in enumerate(steps, start=1):
        lines.append(f"  {i}. {step.get('action', '')}")

    video_id = payload.get("video_id", "?")
    clip_start = payload.get("clip_start", "?")
    clip_end = payload.get("clip_end", "?")
    lines.append(f"출처: {video_id} ({clip_start}–{clip_end})")

    trap = extract_beginner_trap(payload.get("tacit_insight", ""), payload.get("reasoning", ""))
    if trap:
        lines.append(f"⚠ 초보자 함정: {trap}")

    return "\n".join(lines)


def _build_accept_filter():
    """D5: routing==accept 필터 훅. langchain_qdrant는 payload를
    {"metadata": {...}, "page_content": ...} 구조로 저장하므로 metadata.routing을 본다."""
    from qdrant_client.http import models as qmodels

    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.routing", match=qmodels.MatchValue(value="accept"))]
    )


def search_tacit(
    vectorstore,
    query: str,
    top_k: int = 3,
    accept_only: bool = False,
    min_similarity: float = 0.30,
) -> List[Dict[str, Any]]:
    """신입 증상 질의 -> bge-m3 인코딩(vectorstore 내부) -> Qdrant 검색 -> 답변 조립.

    accept_only=False가 기본값이다(D5: 지금은 필터 끔 — STEP3 VLM 버그 수정 후 켤 자리).
    반환: [{"id","score","routing","confidence","answer","timestamp_warnings"}], 유사순 정렬.
    관련 결과가 min_similarity 미만이면 그 자리에 조립하지 않고, 전부 미만이면
    단일 NOT_FOUND 항목 하나만 반환한다(환각 금지 — 코딩지시 5절).
    """
    search_kwargs: Dict[str, Any] = {"k": top_k}
    if accept_only:
        search_kwargs["filter"] = _build_accept_filter()

    raw_results = vectorstore.similarity_search_with_score(query, **search_kwargs)

    results: List[Dict[str, Any]] = []
    for doc, score in raw_results:
        similarity = score if score <= 1.0 else 1.0 / (1.0 + score)
        if similarity < min_similarity:
            continue
        payload = doc.metadata
        warnings = validate_timestamps(payload)
        for w in warnings:
            print(f"      [WARN][timestamp_validity] {w}")
        results.append(
            {
                "id": payload.get("id"),
                "score": float(similarity),
                "routing": payload.get("routing"),
                "confidence": payload.get("confidence"),
                "answer": format_answer(payload, similarity),
                "timestamp_warnings": warnings,
            }
        )

    if not results:
        return [{"id": None, "score": 0.0, "routing": None, "confidence": None,
                  "answer": NOT_FOUND_MESSAGE, "timestamp_warnings": []}]
    return results
