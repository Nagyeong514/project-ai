# -*- coding: utf-8 -*-
"""
하이퍼파라미터 비교 실험용 평가 스크립트 (서시은 evaluate.py 베이스).
신입 작업자 스타일 증상 질의 -> 정답 문서 id를 정의해두고, 검색이 top-k 안에
정답을 넣는 비율(Recall@k)을 측정한다.

EVAL_SET을 STEP7_DB/gold_records 9건 기준으로 교체(서시은 원본은 자기 sample_knowledge.json
3건 기준이었음). 질문은 tacit_insight를 그대로 베끼지 않고 신입이 실제로 물어볼 법한
증상/질문 표현으로 작성 — 그래야 "임베딩이 증상 언어와도 붙는지"를 재는 원래 목적에 맞다.

사용법:
    .venv/bin/python3 evaluate.py --top-k 3
    EMBED_MODE=insight .venv/bin/python3 evaluate.py --collection tacit_knowledge_insight --top-k 3
"""
from __future__ import annotations

import argparse

from preflight import run_preflight

# 무거운 임베딩 모델/DB 클라이언트는 프리플라이트 통과 후에만 import한다.

EVAL_SET = [
    ("램 아무 슬롯에나 꽂아도 되나요?", "tk_dell7920_ch_preread_001"),
    ("커넥터 눈으로 보기엔 잘 꽂힌 것 같은데 왜 안 켜지죠?", "tk_dell7920_tactile_conn_002"),
    ("전원 LED가 깜빡거리는데 무슨 뜻이에요?", "tk_dell7920_led_read_003"),
    ("어떤 부품이 문제인지 모르겠어요, 뭐부터 봐야 하죠?", "tk_dell7920_systematic_check_004"),
    ("램 뺄 때 한쪽 래치만 눌러도 되나요?", "tk_dell7920_latch_removal_005"),
    ("새 램인데도 인식이 안 돼요", "tk_dell7920_terminal_clean_006"),
    ("램이 제대로 꽂혔는지 어떻게 확인해요?", "tk_dell7920_latch_confirm_007"),
    ("이 슬롯이 몇 채널인지 어떻게 알아요?", "tk_dell7920_channel_id_008"),
    ("부팅 됐으면 다 끝난 거 아닌가요?", "tk_dell7920_bios_verify_009"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None, help="기본: config.py/COLLECTION 환경변수")
    ap.add_argument("--top-k", type=int, default=None, help="기본: config.py/TOP_K 환경변수")
    args = ap.parse_args()

    run_preflight()

    from config import CONFIG
    import search as se

    collection = args.collection or CONFIG.qdrant_collection
    top_k = args.top_k or CONFIG.top_k_default

    embedder, client = se.load_backend(CONFIG)
    eval_config = CONFIG.model_copy(update={"qdrant_collection": collection, "min_similarity_threshold": 0.0})

    hits = 0
    for question, gold_id in EVAL_SET:
        results = se.search(embedder, client, eval_config, question, top_k=top_k)
        ids = [r["id"] for r in results]
        ok = gold_id in ids
        hits += ok
        top_sim = results[0]["similarity"] if results else 0.0
        print(f"{'O' if ok else 'X'}  top1_sim={top_sim}  Q: {question}")

    print(f"\nRecall@{top_k} = {hits}/{len(EVAL_SET)} = {hits/len(EVAL_SET):.2f}  (collection={collection})")


if __name__ == "__main__":
    main()
