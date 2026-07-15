# -*- coding: utf-8 -*-
"""
RAG Embedding Model 3종 실측 비교 스크립트

비교 모델:
  1) BAAI/bge-m3
  2) intfloat/multilingual-e5-base
  3) sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

기능:
  - STEP7_DB/gold_records JSON을 읽어 STEP8 RAG 서비스 방식으로 chunk 생성
  - 동일 query set / 동일 gold label / 동일 top-k 조건으로 3개 모델 평가
  - Hit@1/3/5, Recall@3/5, MRR, nDCG@3/5, latency 측정
  - 결과 CSV/JSON/Markdown/HTML 보고서 생성
  - 전체 결과를 zip으로 묶어 다운로드 가능하게 저장

권장 실행 위치:
  project-ai/
    STEP7_DB/
    STEP8_RAG서비스/
    embedding_model_eval/
      run_embedding_model_eval.py
      queries.json

예시:
  python3 run_embedding_model_eval.py --project-root .. --device cuda
  python3 run_embedding_model_eval.py --gold-dir ../STEP7_DB/gold_records --chunk-mode step8_full
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MODEL_SPECS = [
    {
        "key": "bge_m3",
        "name": "BAAI/bge-m3",
        "display": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "profile": "가장 무겁지만 한국어/다국어 retrieval과 구어체 표현 변이에 강한 현행 후보",
    },
    {
        "key": "multilingual_e5_base",
        "name": "intfloat/multilingual-e5-base",
        "display": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "profile": "품질과 속도 균형형 후보. query:/passage: prefix 적용이 필수",
    },
    {
        "key": "minilm_l12",
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "display": "paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "profile": "가장 가볍고 빠른 후보. 짧은 증상 질문→긴 지식 chunk 검색에서는 불리할 수 있음",
    },
]

NEGATIVE_QUERIES = [
    "오늘 점심 뭐 먹지?",
    "파이썬 리스트 정렬하는 법 알려줘",
    "내일 부산 날씨 어때?",
    "헬스할 때 단백질 얼마나 먹어야 해?",
    "노트북 배터리가 빨리 닳아요",
]


@dataclass
class Entry:
    doc_id: str
    source_file: str
    raw: Dict[str, Any]
    text: str


# -----------------------------------------------------------------------------
# 파일/데이터 로딩
# -----------------------------------------------------------------------------


def find_gold_dir(args: argparse.Namespace) -> Path:
    """gold_records 위치를 자동 탐색한다."""
    candidates: List[Path] = []

    if args.gold_dir:
        candidates.append(Path(args.gold_dir))

    if args.project_root:
        root = Path(args.project_root)
        candidates.append(root / "STEP7_DB" / "gold_records")

    here = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    for base in [here, here.parent, script_dir, script_dir.parent]:
        candidates.extend([
            base / "STEP7_DB" / "gold_records",
            base.parent / "STEP7_DB" / "gold_records",
        ])

    seen = set()
    unique_candidates = []
    for c in candidates:
        rc = c.expanduser().resolve()
        if str(rc) not in seen:
            seen.add(str(rc))
            unique_candidates.append(rc)

    for c in unique_candidates:
        if c.exists() and c.is_dir() and list(c.glob("*.json")):
            return c

    msg = "\n".join(f"  - {c}" for c in unique_candidates)
    raise FileNotFoundError(
        "STEP7_DB/gold_records 폴더를 찾지 못했습니다.\n"
        "--gold-dir 로 직접 지정하세요. 탐색한 위치:\n" + msg
    )


def load_queries(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("queries.json은 list 형태여야 합니다.")
    required = {"qid", "type", "query", "gold", "relevant"}
    for q in data:
        missing = required - set(q)
        if missing:
            raise ValueError(f"queries.json 항목에 필수 키가 없습니다: {missing}, 항목={q}")
        if q["gold"] not in q["relevant"]:
            q["relevant"] = [q["gold"], *q["relevant"]]
    return data


def build_document(entry: Dict[str, Any], mode: str) -> str:
    """STEP7/STEP8 코드에 맞춘 임베딩 대상 텍스트 생성."""
    k = entry["knowledge"]
    m = entry["metadata"]
    situation = k.get("situation", "")
    insight = k.get("tacit_insight", "")
    reasoning = k.get("reasoning", "")

    if mode == "step7":
        # STEP7_DB/vectordb_utils.py 방식
        return f"[상황] {situation}\n[노하우] {insight}\n[이유] {reasoning}"

    if mode == "step8_insight":
        # STEP8_RAG서비스/ingest.py의 insight 모드
        return insight

    if mode == "step8_full":
        # STEP8_RAG서비스/ingest.py의 full 모드
        steps = " / ".join(s.get("action", "") for s in k.get("diagnostic_steps", []))
        return (
            f"작업: {m.get('task', '')}. 상황: {situation}. "
            f"노하우: {insight} 이유: {reasoning} "
            f"절차: {steps}. 키워드: {', '.join(m.get('keywords', []))}"
        )

    raise ValueError(f"지원하지 않는 chunk mode입니다: {mode}")


def load_entries(gold_dir: Path, chunk_mode: str) -> List[Entry]:
    entries: List[Entry] = []
    for path in sorted(gold_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            doc_id = raw["id"]
            _ = raw["knowledge"]["tacit_insight"]
            _ = raw["metadata"]["task"]
            text = build_document(raw, chunk_mode)
            entries.append(Entry(doc_id=doc_id, source_file=path.name, raw=raw, text=text))
        except Exception as e:
            print(f"[WARN] {path.name} 건너뜀: {e}")
    if not entries:
        raise FileNotFoundError(f"유효한 JSON을 읽지 못했습니다: {gold_dir}")
    return entries


# -----------------------------------------------------------------------------
# 평가 지표
# -----------------------------------------------------------------------------


def cosine_topk(query_vec: np.ndarray, doc_mat: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """정규화된 벡터 기준 cosine similarity top-k."""
    sims = doc_mat @ query_vec
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    rel_set = set(relevant_ids)
    gains = [1 if doc_id in rel_set else 0 for doc_id in ranked_ids]
    dcg = dcg_at_k(gains, k)
    ideal_count = min(len(rel_set), k)
    ideal = dcg_at_k([1] * ideal_count, k)
    return 0.0 if ideal == 0 else dcg / ideal


def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), 95))


# -----------------------------------------------------------------------------
# 모델 실행
# -----------------------------------------------------------------------------


def import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers를 import하지 못했습니다.\n"
            "먼저 `pip install -r requirements_eval.txt`를 실행하세요.\n"
            f"원인: {e}"
        )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def encode_texts(
    model: Any,
    texts: List[str],
    prefix: str = "",
    batch_size: int = 16,
    show_progress_bar: bool = True,
) -> np.ndarray:
    prefixed = [prefix + t for t in texts]
    emb = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(emb, dtype=np.float32)


def evaluate_one_model(
    spec: Dict[str, str],
    entries: List[Entry],
    queries: List[Dict[str, Any]],
    out_dir: Path,
    device: str,
    top_k: int,
    batch_size: int,
    cache_dir: Optional[str],
    local_files_only: bool,
) -> Dict[str, Any]:
    SentenceTransformer = import_sentence_transformers()
    model_name = spec["name"]
    key = spec["key"]
    model_out = out_dir / key
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"\n==============================")
    print(f"[MODEL] {model_name}")
    print(f"==============================")

    load_start = time.perf_counter()
    kwargs: Dict[str, Any] = {"device": device}
    if cache_dir:
        kwargs["cache_folder"] = cache_dir
    if local_files_only:
        kwargs["local_files_only"] = True

    model = SentenceTransformer(model_name, **kwargs)
    load_sec = time.perf_counter() - load_start

    docs = [e.text for e in entries]
    doc_ids = [e.doc_id for e in entries]

    print(f"[1/3] 문서 임베딩: {len(docs)}건")
    doc_start = time.perf_counter()
    doc_mat = encode_texts(
        model,
        docs,
        prefix=spec.get("passage_prefix", ""),
        batch_size=batch_size,
        show_progress_bar=True,
    )
    doc_embed_sec = time.perf_counter() - doc_start
    dim = int(doc_mat.shape[1])

    per_query_rows: List[Dict[str, Any]] = []
    hit1 = hit3 = hit5 = 0
    reciprocal_ranks: List[float] = []
    recall3_values: List[float] = []
    recall5_values: List[float] = []
    ndcg3_values: List[float] = []
    ndcg5_values: List[float] = []
    query_latencies_ms: List[float] = []

    print(f"[2/3] Query 평가: {len(queries)}개")
    for q in queries:
        q_start = time.perf_counter()
        q_vec = encode_texts(
            model,
            [q["query"]],
            prefix=spec.get("query_prefix", ""),
            batch_size=1,
            show_progress_bar=False,
        )[0]
        ranked = cosine_topk(q_vec, doc_mat, k=min(top_k, len(entries)))
        latency_ms = (time.perf_counter() - q_start) * 1000.0
        query_latencies_ms.append(latency_ms)

        ranked_ids = [doc_ids[i] for i, _ in ranked]
        ranked_scores = [score for _, score in ranked]
        gold = q["gold"]
        relevant = q["relevant"]

        gold_rank: Optional[int] = None
        for rank, rid in enumerate(ranked_ids, start=1):
            if rid == gold:
                gold_rank = rank
                break

        h1 = gold_rank == 1
        h3 = gold_rank is not None and gold_rank <= 3
        h5 = gold_rank is not None and gold_rank <= 5
        hit1 += int(h1)
        hit3 += int(h3)
        hit5 += int(h5)
        reciprocal_ranks.append(0.0 if gold_rank is None else 1.0 / gold_rank)

        rel_set = set(relevant)
        recall3 = len(set(ranked_ids[:3]) & rel_set) / max(len(rel_set), 1)
        recall5 = len(set(ranked_ids[:5]) & rel_set) / max(len(rel_set), 1)
        recall3_values.append(recall3)
        recall5_values.append(recall5)
        ndcg3_values.append(ndcg_at_k(ranked_ids, relevant, 3))
        ndcg5_values.append(ndcg_at_k(ranked_ids, relevant, 5))

        per_query_rows.append({
            "qid": q["qid"],
            "type": q["type"],
            "query": q["query"],
            "gold": gold,
            "relevant": "|".join(relevant),
            "gold_rank": gold_rank if gold_rank is not None else "not_in_top_k",
            "hit_at_1": int(h1),
            "hit_at_3": int(h3),
            "hit_at_5": int(h5),
            "top1_id": ranked_ids[0] if ranked_ids else "",
            "top1_score": round(ranked_scores[0], 6) if ranked_scores else "",
            "top3_ids": "|".join(ranked_ids[:3]),
            "top3_scores": "|".join(f"{s:.6f}" for s in ranked_scores[:3]),
            "top5_ids": "|".join(ranked_ids[:5]),
            "top5_scores": "|".join(f"{s:.6f}" for s in ranked_scores[:5]),
            "latency_ms": round(latency_ms, 3),
        })

    n = len(queries)
    summary = {
        "model_key": key,
        "model_name": model_name,
        "display": spec["display"],
        "profile": spec["profile"],
        "device": device,
        "dim": dim,
        "load_sec": round(load_sec, 3),
        "doc_embed_sec": round(doc_embed_sec, 3),
        "avg_query_latency_ms": round(safe_mean(query_latencies_ms), 3),
        "p95_query_latency_ms": round(safe_p95(query_latencies_ms), 3),
        "hit_at_1": round(hit1 / n, 6),
        "hit_at_3": round(hit3 / n, 6),
        "hit_at_5": round(hit5 / n, 6),
        "hit_at_1_count": hit1,
        "hit_at_3_count": hit3,
        "hit_at_5_count": hit5,
        "num_queries": n,
        "recall_at_3": round(safe_mean(recall3_values), 6),
        "recall_at_5": round(safe_mean(recall5_values), 6),
        "mrr": round(safe_mean(reciprocal_ranks), 6),
        "ndcg_at_3": round(safe_mean(ndcg3_values), 6),
        "ndcg_at_5": round(safe_mean(ndcg5_values), 6),
    }

    print(f"[3/3] 완료: Hit@1={summary['hit_at_1']:.3f}, Hit@3={summary['hit_at_3']:.3f}, MRR={summary['mrr']:.3f}, avg latency={summary['avg_query_latency_ms']}ms")

    # negative query separation: 실제 정답 없는 질문에 top1 similarity가 얼마나 높은지 확인
    neg_rows: List[Dict[str, Any]] = []
    neg_top1_scores: List[float] = []
    for nq in NEGATIVE_QUERIES:
        q_vec = encode_texts(model, [nq], prefix=spec.get("query_prefix", ""), batch_size=1, show_progress_bar=False)[0]
        ranked = cosine_topk(q_vec, doc_mat, k=3)
        ids = [doc_ids[i] for i, _ in ranked]
        scores = [score for _, score in ranked]
        neg_top1_scores.append(scores[0])
        neg_rows.append({
            "query": nq,
            "top1_id": ids[0],
            "top1_score": round(scores[0], 6),
            "top3_ids": "|".join(ids),
            "top3_scores": "|".join(f"{s:.6f}" for s in scores),
        })
    summary["negative_top1_avg"] = round(safe_mean(neg_top1_scores), 6)
    summary["negative_top1_max"] = round(max(neg_top1_scores), 6) if neg_top1_scores else 0.0

    write_csv(model_out / "per_query_results.csv", per_query_rows)
    write_csv(model_out / "negative_query_results.csv", neg_rows)
    (model_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_out / "per_query_results.json").write_text(json.dumps(per_query_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 메모리 정리
    try:
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"summary": summary, "per_query": per_query_rows, "negative": neg_rows, "error": None}


# -----------------------------------------------------------------------------
# 결과 저장/보고서 생성
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def comparison_sort_key(summary: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    # 큰 값 우선: Hit@1, MRR, nDCG@3, Hit@3 / 작은 값 우선: latency → 음수 처리
    return (
        float(summary.get("hit_at_1", 0.0)),
        float(summary.get("mrr", 0.0)),
        float(summary.get("ndcg_at_3", 0.0)),
        float(summary.get("hit_at_3", 0.0)),
        -float(summary.get("avg_query_latency_ms", 999999.0)),
    )


def choose_winner(successful: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not successful:
        return {
            "winner": None,
            "reason": "성공적으로 실행된 모델이 없어 최종 선택 불가",
            "note": "HuggingFace 접속, 패키지 설치, GPU/CPU 메모리를 확인해야 합니다.",
        }

    sorted_models = sorted(successful, key=comparison_sort_key, reverse=True)
    best = sorted_models[0]

    # 품질이 거의 동률이면 latency도 해석에 반영
    near = [s for s in sorted_models if abs(s["hit_at_1"] - best["hit_at_1"]) <= 0.000001 and abs(s["mrr"] - best["mrr"]) <= 0.000001]
    if len(near) > 1:
        fastest = min(near, key=lambda s: s["avg_query_latency_ms"])
        if fastest["model_key"] != best["model_key"]:
            return {
                "winner": fastest["model_key"],
                "winner_display": fastest["display"],
                "reason": (
                    "Hit@1과 MRR이 동률이라 실제 서비스에서는 평균 query latency가 더 낮은 모델을 선택하는 것이 합리적입니다. "
                    f"동률 후보 중 가장 빠른 모델은 {fastest['display']}입니다."
                ),
                "note": "단, 데이터가 9건이라 동률이 쉽게 발생합니다. 문서가 늘어나면 반드시 재평가해야 합니다.",
            }

    return {
        "winner": best["model_key"],
        "winner_display": best["display"],
        "reason": (
            f"{best['display']}가 Hit@1={best['hit_at_1']:.3f}, MRR={best['mrr']:.3f}, "
            f"nDCG@3={best['ndcg_at_3']:.3f} 기준으로 가장 앞섭니다. "
            "RAG에서는 정답 chunk가 앞순위에 들어오는 것이 가장 중요하므로 이 모델을 1순위로 선택합니다."
        ),
        "note": "속도보다 검색 정확도를 우선한 선택입니다. 실제 배포에서 latency가 문제라면 e5-base를 대안으로 검토하세요.",
    }


def explain_metric(metric: str) -> str:
    explanations = {
        "Hit@1": "정답 chunk가 검색 결과 1등으로 나온 비율입니다. 가장 직관적인 품질 지표입니다.",
        "Hit@3": "정답 chunk가 top-3 안에 들어온 비율입니다. RAG에서 top-3를 LLM에 넣는다면 매우 중요합니다.",
        "MRR": "정답 chunk가 앞순위에 나올수록 높아지는 점수입니다. 1등이면 1점, 2등이면 0.5점입니다.",
        "nDCG@3": "정답뿐 아니라 관련 chunk들이 top-3 안에서 얼마나 좋은 순서로 나왔는지 보는 지표입니다.",
        "latency": "질문 1개를 임베딩하고 검색하는 데 걸린 평균 시간입니다. 낮을수록 빠릅니다.",
    }
    return explanations.get(metric, "")


def make_markdown_report(
    output_dir: Path,
    args: argparse.Namespace,
    gold_dir: Path,
    entries: List[Entry],
    queries: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    decision: Dict[str, Any],
    failed: List[Dict[str, Any]],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# RAG Embedding Model 3종 실측 비교 보고서")
    lines.append("")
    lines.append(f"- 생성 시각: {now}")
    lines.append(f"- gold_records 위치: `{gold_dir}`")
    lines.append(f"- chunk mode: `{args.chunk_mode}`")
    lines.append(f"- 평가 문서 수: {len(entries)}건")
    lines.append(f"- 평가 query 수: {len(queries)}개")
    lines.append(f"- device: `{resolve_device(args.device)}`")
    lines.append("")
    lines.append("## 1. 이 평가가 보는 것")
    lines.append("")
    lines.append("이 평가는 LLM 답변 자체를 비교하는 것이 아닙니다. RAG에서 사용자의 질문이 들어왔을 때, 벡터DB에서 정답 암묵지 chunk를 얼마나 앞순위로 잘 찾아오는지 비교합니다.")
    lines.append("")
    lines.append("검색 단계에서 정답 chunk를 못 찾으면 LLM은 올바른 근거를 받지 못합니다. 그래서 여기서는 similarity score의 절대값보다 `정답 chunk의 순위`를 더 중요하게 봅니다.")
    lines.append("")
    lines.append("## 2. 지표 설명")
    lines.append("")
    lines.append("| 지표 | 쉬운 의미 |")
    lines.append("|---|---|")
    lines.append(f"| Hit@1 | {explain_metric('Hit@1')} |")
    lines.append(f"| Hit@3 | {explain_metric('Hit@3')} |")
    lines.append(f"| MRR | {explain_metric('MRR')} |")
    lines.append(f"| nDCG@3 | {explain_metric('nDCG@3')} |")
    lines.append(f"| latency | {explain_metric('latency')} |")
    lines.append("")
    lines.append("## 3. 전체 비교 결과")
    lines.append("")
    lines.append("| 모델 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@3 | Recall@3 | 평균 latency(ms) | dim |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in comparison_rows:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('display', r.get('model_key'))} | 실패 | 실패 | 실패 | 실패 | 실패 | 실패 | 실패 | - |")
            continue
        lines.append(
            f"| {r['display']} | {r['hit_at_1']:.3f} | {r['hit_at_3']:.3f} | {r['hit_at_5']:.3f} | "
            f"{r['mrr']:.3f} | {r['ndcg_at_3']:.3f} | {r['recall_at_3']:.3f} | "
            f"{r['avg_query_latency_ms']:.3f} | {r['dim']} |"
        )
    lines.append("")

    if failed:
        lines.append("## 4. 실행 실패 모델")
        lines.append("")
        for f in failed:
            lines.append(f"- **{f['display']}**: `{f['error']}`")
        lines.append("")
        lines.append("실패한 모델이 있으면 3개 모델의 최종 비교는 아직 완성된 것이 아닙니다. HuggingFace 접속 가능 여부, 모델 캐시, 패키지 설치, GPU/CPU 메모리를 먼저 확인해야 합니다.")
        lines.append("")

    lines.append("## 4. 최종 선택")
    lines.append("")
    if decision.get("winner"):
        lines.append(f"**선택 모델: `{decision['winner_display']}`**")
        lines.append("")
        lines.append(decision["reason"])
        lines.append("")
        lines.append(decision["note"])
    else:
        lines.append("**최종 선택 불가**")
        lines.append("")
        lines.append(decision["reason"])
        lines.append("")
        lines.append(decision["note"])
    lines.append("")

    lines.append("## 5. 모델별 해석")
    lines.append("")
    for spec in MODEL_SPECS:
        key = spec["key"]
        result = results.get(key)
        lines.append(f"### {spec['display']}")
        lines.append("")
        lines.append(f"- 모델 성격: {spec['profile']}")
        if not result or result.get("error"):
            lines.append("- 이번 실행에서는 실패했기 때문에 성능 해석에서 제외합니다.")
            lines.append("")
            continue
        s = result["summary"]
        lines.append(f"- Hit@1={s['hit_at_1']:.3f}, Hit@3={s['hit_at_3']:.3f}, MRR={s['mrr']:.3f}")
        lines.append(f"- 평균 query latency={s['avg_query_latency_ms']:.3f}ms, dim={s['dim']}")
        lines.append(f"- 무관 질문 top1 similarity 평균={s['negative_top1_avg']:.3f}, 최대={s['negative_top1_max']:.3f}")

        failures = [row for row in result["per_query"] if row["hit_at_1"] == 0]
        if not failures:
            lines.append("- 모든 평가 query에서 정답 chunk를 1등으로 찾았습니다.")
        else:
            lines.append(f"- 정답을 1등으로 못 찾은 query가 {len(failures)}개 있습니다. 대표 사례:")
            for row in failures[:5]:
                lines.append(
                    f"  - `{row['qid']}` {row['query']} → 정답 `{row['gold']}`, "
                    f"top1 `{row['top1_id']}`, gold_rank `{row['gold_rank']}`"
                )
        lines.append("")

    lines.append("## 6. 비전공자용 결론")
    lines.append("")
    lines.append("이 평가는 '질문을 보고 맞는 암묵지 문서를 잘 찾는가'를 본 것입니다. `Hit@1`이 높다는 것은 사용자의 질문에 대해 정답 문서를 바로 1등으로 가져온다는 뜻입니다. `Hit@3`이 높다는 것은 LLM에게 넘기는 후보 3개 안에는 정답이 들어간다는 뜻입니다.")
    lines.append("")
    lines.append("따라서 실제 RAG 서비스에서는 보통 `Hit@1`, `MRR`, `Hit@3` 순서로 중요하게 보면 됩니다. 속도는 그 다음입니다. 단, 품질이 거의 같다면 더 빠르고 가벼운 모델을 고르는 것이 실용적입니다.")
    lines.append("")
    lines.append("## 7. 생성 파일")
    lines.append("")
    lines.append("- `embedding_model_comparison.csv`: 모델별 요약 비교표")
    lines.append("- `report.md`: 사람이 읽는 해석 보고서")
    lines.append("- `report.html`: 브라우저에서 볼 수 있는 해석 보고서")
    lines.append("- `results/<model>/summary.json`: 모델별 요약 지표")
    lines.append("- `results/<model>/per_query_results.csv`: query별 검색 순위")
    lines.append("- `embedding_eval_results.zip`: 위 결과물을 하나로 묶은 다운로드용 파일")
    lines.append("")

    return "\n".join(lines)


def markdown_to_simple_html(md: str, title: str = "RAG Embedding Model 비교 보고서") -> str:
    """외부 패키지 없이 충분히 읽기 쉬운 HTML로 변환."""
    lines = md.splitlines()
    body: List[str] = []
    in_table = False
    table_rows: List[str] = []
    in_ul = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        rows = table_rows
        table_rows = []
        in_table = False
        if len(rows) >= 2:
            header = [c.strip() for c in rows[0].strip("|").split("|")]
            body.append("<table>")
            body.append("<thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in header) + "</tr></thead>")
            body.append("<tbody>")
            for row in rows[2:]:
                cells = [c.strip() for c in row.strip("|").split("|")]
                body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            body.append("</tbody></table>")

    def flush_ul() -> None:
        nonlocal in_ul
        if in_ul:
            body.append("</ul>")
            in_ul = False

    for line in lines:
        if line.startswith("|") and line.endswith("|"):
            flush_ul()
            in_table = True
            table_rows.append(line)
            continue
        flush_table()

        if line.startswith("# "):
            flush_ul(); body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_ul(); body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush_ul(); body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            if not in_ul:
                body.append("<ul>"); in_ul = True
            body.append(f"<li>{inline_md_to_html(line[2:])}</li>")
        elif line.strip() == "":
            flush_ul()
        else:
            flush_ul(); body.append(f"<p>{inline_md_to_html(line)}</p>")
    flush_table(); flush_ul()

    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',Arial,sans-serif;max-width:980px;margin:36px auto;padding:0 24px;line-height:1.72;color:#1f2937;background:#ffffff}
    h1{border-bottom:3px solid #2563eb;padding-bottom:8px} h2{margin-top:32px;border-bottom:1px solid #d1d5db;padding-bottom:4px;color:#1d4ed8} h3{margin-top:24px;color:#374151}
    table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px} th,td{border:1px solid #d1d5db;padding:8px 10px;text-align:left;vertical-align:top} th{background:#eff6ff} tr:nth-child(even){background:#f9fafb}
    code{background:#f3f4f6;padding:2px 5px;border-radius:4px} strong{font-weight:700}
    """
    return f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


def inline_md_to_html(text: str) -> str:
    # 아주 단순한 inline 처리: `code`, **bold** 정도만 처리
    escaped = html.escape(text)
    import re
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def make_zip(output_dir: Path) -> Path:
    zip_path = output_dir / "embedding_eval_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in output_dir.rglob("*"):
            if path == zip_path or path.is_dir():
                continue
            z.write(path, path.relative_to(output_dir))
    return zip_path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RAG embedding model 3종 실측 비교")
    ap.add_argument("--project-root", default=None, help="STEP7_DB와 STEP8_RAG서비스가 들어있는 프로젝트 루트")
    ap.add_argument("--gold-dir", default=None, help="STEP7_DB/gold_records 직접 경로")
    ap.add_argument("--queries", default=None, help="queries.json 경로. 기본: 스크립트와 같은 폴더의 queries.json")
    ap.add_argument("--output-dir", default="embedding_eval_outputs", help="결과 저장 폴더")
    ap.add_argument("--chunk-mode", choices=["step7", "step8_full", "step8_insight"], default="step8_full",
                    help="임베딩할 chunk 구성 방식. 기본 step8_full = 현재 STEP8 서비스 방식")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="모델 실행 장치")
    ap.add_argument("--top-k", type=int, default=9, help="검색 후보 수. 문서가 9건이므로 기본 9")
    ap.add_argument("--batch-size", type=int, default=16, help="임베딩 batch size")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace/SentenceTransformer 캐시 폴더")
    ap.add_argument("--local-files-only", action="store_true", help="인터넷 다운로드 없이 로컬 캐시 모델만 사용")
    ap.add_argument("--models", nargs="*", default=[m["key"] for m in MODEL_SPECS],
                    help="실행할 모델 key. 기본: 3개 모두. 예: --models bge_m3 multilingual_e5_base")
    ap.add_argument("--stop-on-error", action="store_true", help="한 모델 실패 시 전체 중단")
    ap.add_argument("--colab-download", action="store_true", help="Colab에서 실행 시 결과 zip 자동 다운로드")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_dir = find_gold_dir(args)
    query_path = Path(args.queries).expanduser().resolve() if args.queries else Path(__file__).resolve().parent / "queries.json"
    if not query_path.exists():
        raise FileNotFoundError(f"queries.json을 찾지 못했습니다: {query_path}")

    device = resolve_device(args.device)
    print(f"[SETUP] gold_dir={gold_dir}")
    print(f"[SETUP] queries={query_path}")
    print(f"[SETUP] output_dir={output_dir}")
    print(f"[SETUP] chunk_mode={args.chunk_mode}")
    print(f"[SETUP] device={device}")

    entries = load_entries(gold_dir, args.chunk_mode)
    queries = load_queries(query_path)

    # 평가 데이터 정보 저장
    docs_dump = [
        {
            "doc_id": e.doc_id,
            "source_file": e.source_file,
            "embedding_text": e.text,
            "task": e.raw.get("metadata", {}).get("task"),
            "keywords": e.raw.get("metadata", {}).get("keywords", []),
        }
        for e in entries
    ]
    (output_dir / "evaluation_documents.json").write_text(json.dumps(docs_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluation_queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_specs = [s for s in MODEL_SPECS if s["key"] in set(args.models)]
    if not selected_specs:
        raise ValueError(f"실행할 모델이 없습니다. 가능한 key: {[m['key'] for m in MODEL_SPECS]}")

    results: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for spec in selected_specs:
        try:
            r = evaluate_one_model(
                spec=spec,
                entries=entries,
                queries=queries,
                out_dir=output_dir / "results",
                device=device,
                top_k=args.top_k,
                batch_size=args.batch_size,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            )
            results[spec["key"]] = r
            row = dict(r["summary"])
            row["status"] = "ok"
            comparison_rows.append(row)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[ERROR] {spec['display']} 실패: {err}", file=sys.stderr)
            failure = {
                "model_key": spec["key"],
                "model_name": spec["name"],
                "display": spec["display"],
                "status": "failed",
                "error": err,
            }
            results[spec["key"]] = {"summary": None, "per_query": [], "negative": [], "error": err}
            comparison_rows.append(failure)
            failed.append(failure)
            if args.stop_on_error:
                raise

    successful = [r for r in comparison_rows if r.get("status") == "ok"]
    successful_sorted = sorted(successful, key=comparison_sort_key, reverse=True)
    failed_sorted = [r for r in comparison_rows if r.get("status") != "ok"]
    comparison_rows_final = successful_sorted + failed_sorted

    write_csv(output_dir / "embedding_model_comparison.csv", comparison_rows_final)
    (output_dir / "embedding_model_comparison.json").write_text(
        json.dumps(comparison_rows_final, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    decision = choose_winner(successful_sorted)
    (output_dir / "final_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = make_markdown_report(
        output_dir=output_dir,
        args=args,
        gold_dir=gold_dir,
        entries=entries,
        queries=queries,
        results=results,
        comparison_rows=comparison_rows_final,
        decision=decision,
        failed=failed,
    )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "report.html").write_text(markdown_to_simple_html(report_md), encoding="utf-8")

    zip_path = make_zip(output_dir)

    print("\n==============================")
    print("평가 완료")
    print("==============================")
    print(f"비교표: {output_dir / 'embedding_model_comparison.csv'}")
    print(f"보고서: {output_dir / 'report.html'}")
    print(f"다운로드용 zip: {zip_path}")
    if decision.get("winner"):
        print(f"최종 선택: {decision['winner_display']}")
    else:
        print("최종 선택: 불가")

    if args.colab_download:
        try:
            from google.colab import files  # type: ignore
            files.download(str(zip_path))
        except Exception as e:
            print(f"[WARN] Colab 자동 다운로드 실패: {e}")


if __name__ == "__main__":
    main()
