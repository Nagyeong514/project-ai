# -*- coding: utf-8 -*-
"""
embedding_model_eval 전용 평가 데이터 정의.

원본 파일(STEP7_DB/gold_records/*.json, STEP8_RAG서비스/ingest.py)은 절대 수정하지 않고
읽기만 한다. chunk 구성 방식은 STEP8_RAG서비스/ingest.py의 build_document(mode="full")을
그대로 재사용한다 — 실제 배포된 RAG 서비스(app.py)가 쓰는 chunk 형식과 동일해야
embedding model 비교 결과가 실제 서비스에 의미가 있기 때문이다.
  "작업: {task}. 상황: {situation}. 노하우: {tacit_insight} 이유: {reasoning} 절차: {steps}. 키워드: {keywords}"

QUERIES: 각 항목 (query, gold_id, qtype) 튜플.
qtype 코드:
  V  = 원문과 표현이 비슷한 질문(near-verbatim)
  S  = 원문과 표현이 다른 의미 기반 질문(semantic paraphrase)
  T  = STT 결과처럼 구어체인 질문
  I  = 약간 불완전하거나 짧은 질문
  M  = 전문용어 + 일반 표현이 섞인 질문
한 query가 여러 성격을 겸하면 복수 코드를 붙인다(예: "T+I").
"""
from __future__ import annotations

import glob
import json
import os
import sys

# 2026-07-15: embedding_eval/ → STEP1_모델연구/08_RAG/ 로 이동하며 repoint.
# 옛날엔 옆에 둔 STEP7/8 사본(../STEP7_DB, ../STEP8_RAG서비스)을 읽었으나,
# gold_records와 ingest.build_document이 project-ai 실제 폴더와 완전히 동일함을 확인해
# 1.4GB 사본을 폐기하고 project-ai 실제 STEP7/8을 직접 참조한다(중복 제거).
# 여기(08_RAG/embedding_model_eval)에서 project-ai 루트는 3단계 위.
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
STEP7_GOLD_DIR = os.path.join(PROJECT_ROOT, "STEP7_DB", "gold_records")
STEP8_DIR = os.path.join(PROJECT_ROOT, "STEP8_RAG서비스")


def _load_build_document():
    """STEP8_RAG서비스/ingest.py의 build_document()를 원본 수정 없이 그대로 import."""
    sys.path.insert(0, STEP8_DIR)
    import ingest  # STEP8_RAG서비스/ingest.py (읽기 전용 import, 실행하지 않음)
    return ingest.build_document


def load_chunks() -> list[dict]:
    """STEP7_DB/gold_records/*.json -> [{"id":..., "text":..., "task":..., "keywords":[...]}]
    STEP8 ingest.py와 동일한 embed_mode="full" 텍스트 구성을 그대로 재사용한다."""
    build_document = _load_build_document()
    chunks = []
    for path in sorted(glob.glob(os.path.join(STEP7_GOLD_DIR, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        text = build_document(entry, "full")
        chunks.append(
            {
                "id": entry["id"],
                "text": text,
                "task": entry["metadata"]["task"],
                "tacit_insight": entry["knowledge"]["tacit_insight"],
                "keywords": entry["metadata"].get("keywords", []),
            }
        )
    return chunks


# (query, gold_chunk_id, qtype) — 최소 20개 요구사항 대비 28개, 9개 chunk 전체 커버,
# 정성평가 필수 6유형(LED/RAM-RDIMM/부팅-POST실패/케이블-커넥터/BIOS메모리인식/STT구어체) 포함.
QUERIES: list[tuple[str, str, str]] = [
    # --- tk_dell7920_ch_preread_001 (RDIMM 채널 사전 확인) ---
    ("RDIMM 끼우기 전에 라벨하고 색깔로 채널부터 봐야 돼요?", "tk_dell7920_ch_preread_001", "V"),
    ("램 아무 슬롯에나 꽂아도 되나요?", "tk_dell7920_ch_preread_001", "S"),

    # --- tk_dell7920_tactile_conn_002 (전원 커넥터 촉각 확인) ---
    ("커넥터를 손으로 눌러서 체결 상태를 확인해야 하나요?", "tk_dell7920_tactile_conn_002", "V"),
    ("커넥터 눈으로 보기엔 잘 꽂힌 것 같은데 왜 안 켜지죠?", "tk_dell7920_tactile_conn_002", "S"),
    ("선 연결 확인 어떻게 해요", "tk_dell7920_tactile_conn_002", "I+M"),

    # --- tk_dell7920_led_read_003 (LED 패턴 판독) ---
    ("전원 LED 점멸 패턴으로 원인을 어떻게 좁혀요?", "tk_dell7920_led_read_003", "V"),
    ("전원 LED가 깜빡거리는데 무슨 뜻이에요?", "tk_dell7920_led_read_003", "S"),
    ("불이 막 깜빡깜빡 하는데 이게 뭔 뜻이에요", "tk_dell7920_led_read_003", "T"),
    ("LED 패턴", "tk_dell7920_led_read_003", "I"),

    # --- tk_dell7920_systematic_check_004 (체계적 케이블 점검 / 부팅 실패 진단) ---
    ("부품 하나 의심하기 전에 전체 연결 상태부터 봐야 하나요?", "tk_dell7920_systematic_check_004", "V"),
    ("어떤 부품이 문제인지 모르겠어요, 뭐부터 봐야 하죠?", "tk_dell7920_systematic_check_004", "S"),
    ("부팅이 안 되는데 뭐부터 확인해야 되냐면요", "tk_dell7920_systematic_check_004", "T+I"),

    # --- tk_dell7920_latch_removal_005 (RDIMM 탈거 시 양쪽 래치) ---
    ("RDIMM 탈거할 때 양쪽 래치를 동시에 눌러야 하나요?", "tk_dell7920_latch_removal_005", "V"),
    ("램 뺄 때 한쪽 래치만 눌러도 되나요?", "tk_dell7920_latch_removal_005", "S"),
    ("메모리 뺄 때 래치 하나만 눌러도 되나", "tk_dell7920_latch_removal_005", "T"),

    # --- tk_dell7920_terminal_clean_006 (RAM 단자 세척) ---
    ("새 부품이라도 RAM 단자 산화막을 먼저 제거해야 하나요?", "tk_dell7920_terminal_clean_006", "V"),
    ("새 램인데도 인식이 안 돼요", "tk_dell7920_terminal_clean_006", "S+I"),
    ("메모리 새거 꽂았는데 왜 인식을 못 하지", "tk_dell7920_terminal_clean_006", "T"),

    # --- tk_dell7920_latch_confirm_007 (래치 잠금 이중 확인) ---
    ("래치 잠금을 클릭 소리와 촉각으로 이중 확인해야 하나요?", "tk_dell7920_latch_confirm_007", "V"),
    ("램이 제대로 꽂혔는지 어떻게 확인해요?", "tk_dell7920_latch_confirm_007", "S"),
    ("메모리 제대로 끼워졌는지 확인", "tk_dell7920_latch_confirm_007", "I"),

    # --- tk_dell7920_channel_id_008 (슬롯 색상/라벨로 채널 식별) ---
    ("슬롯 색상과 라벨로 채널 구성을 즉시 파악할 수 있나요?", "tk_dell7920_channel_id_008", "V"),
    ("이 슬롯이 몇 채널인지 어떻게 알아요?", "tk_dell7920_channel_id_008", "S"),
    ("이거 A채널이야 B채널이야", "tk_dell7920_channel_id_008", "T+I"),

    # --- tk_dell7920_bios_verify_009 (BIOS 메모리 최종 검증) ---
    ("BIOS에서 메모리 용량과 채널 구성을 확인해야 완료된 건가요?", "tk_dell7920_bios_verify_009", "V"),
    ("부팅 됐으면 다 끝난 거 아닌가요?", "tk_dell7920_bios_verify_009", "S"),
    ("바이오스에서 램 용량 확인하는 거", "tk_dell7920_bios_verify_009", "I+M"),
    ("포스트 다 뜨고 부팅 됐는데 램 용량 맞는지는 바이오스에서 봐야 돼요?", "tk_dell7920_bios_verify_009", "T+M"),
]


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"chunks: {len(chunks)}건")
    for c in chunks:
        print(f"  - {c['id']}: {c['text'][:60]}...")
    print(f"\nqueries: {len(QUERIES)}건")
    ids_in_chunks = {c["id"] for c in chunks}
    missing = [q for q in QUERIES if q[1] not in ids_in_chunks]
    assert not missing, f"gold_id가 chunk에 없음: {missing}"
    print("모든 query의 gold_id가 chunk 집합 안에 있음 — OK")
