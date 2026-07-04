# -*- coding: utf-8 -*-
"""
문서 6-4 (실행 순서) 를 LangGraph StateGraph로 구현.

[0단계] timestamp_validity (Rule, 탈락조건) -> 실패시 STOP -> reject
[1단계] Gate A: Manual RAG 검색 (유사도 임계 미만시 키워드 재구성 후 1회 재검색)
[2단계] Gate A: LLM Judge same/delta/novel (탈락조건) -> same시 STOP -> reject
[3단계] Gate B,C 점수항목: LLM Judge
[4단계] confidence 가중합 -> accept/hold/reject 라우팅
"""
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

import rules
import llm_utils
import embedding_utils


class PipelineState(TypedDict, total=False):
    candidate: dict
    transcript: List[dict]
    config: Any
    llm: Any
    vectorstore: Any

    log: List[str]

    timestamp_valid: bool
    timestamp_detail: dict

    manual_query_used: str
    manual_hits: List[Dict[str, Any]]
    manual_top_score: float
    manual_retry_used: bool

    gate_a_relation: str
    gate_a_justification: str
    gate_a_raw: dict

    action_reason_consistency: float
    action_reason_justification: str

    reasoning_grounding: float
    reasoning_grounding_justification: str

    step_grounding_ratio: float
    step_grounding_detail: list

    utterance_signal: float
    utterance_signal_rule_detail: dict
    utterance_signal_justification: str

    confidence: float
    decision: str
    reject_reason: str


def _log(state: PipelineState, msg: str):
    state.setdefault("log", [])
    state["log"].append(msg)


# --------------------------------------------------------------------------
# [0단계] timestamp_validity
# --------------------------------------------------------------------------
def node_timestamp_check(state: PipelineState) -> PipelineState:
    candidate = state["candidate"]
    transcript = state.get("transcript", [])
    valid, detail = rules.check_timestamp_validity(candidate, transcript)
    state["timestamp_valid"] = valid
    state["timestamp_detail"] = detail
    _log(state, "=" * 70)
    _log(state, "[0단계] timestamp_validity (Rule 기반, 탈락조건)")
    for c in detail["checks"]:
        _log(state, f"  - {c['item']} @ {c.get('timestamp')}: {'OK' if c['ok'] else 'FAIL'} ({c['note']})")
    _log(state, f"  => 결과: {'PASS' if valid else 'FAIL'}")
    if not valid:
        state["decision"] = "reject"
        state["reject_reason"] = "timestamp_validity 실패 (0단계에서 STOP)"
        _log(state, f"  => STOP: reject ({state['reject_reason']})")
    return state


def route_after_timestamp(state: PipelineState) -> str:
    return "reject" if not state["timestamp_valid"] else "gate_a_search"


# --------------------------------------------------------------------------
# [1단계] Gate A: Manual RAG 검색
# --------------------------------------------------------------------------
def node_gate_a_search(state: PipelineState) -> PipelineState:
    candidate = state["candidate"]
    config = state["config"]
    vectorstore = state["vectorstore"]
    meta = candidate["metadata"]

    def _search(query: str):
        hits = embedding_utils.search_manual(vectorstore, query, config.manual_top_k)
        return [{"text": t, "score": s, "metadata": m} for t, s, m in hits]

    primary_query = f"{meta['equipment']} {meta['task']} " + " ".join(meta.get("keywords", []))
    hits = _search(primary_query)
    top_score = max([h["score"] for h in hits], default=0.0)

    _log(state, "=" * 70)
    _log(state, "[1단계] Gate A - Manual RAG 검색")
    _log(state, f"  1차 쿼리: '{primary_query}'")
    for h in hits:
        _log(state, f"    - score={h['score']:.3f} source={h['metadata'].get('source')} text={h['text'][:80]}...")
    _log(state, f"  1차 top1 유사도: {top_score:.3f} (임계값 {config.manual_similarity_threshold})")

    retry_used = False
    if top_score < config.manual_similarity_threshold:
        retry_used = True
        # 키워드 재구성: tacit_insight 문장을 그대로 사용해 재검색 (문서 6-4 [1단계] "키워드를 바꿔 1회 재검색")
        fallback_query = candidate["knowledge"]["tacit_insight"]
        hits2 = _search(fallback_query)
        top_score2 = max([h["score"] for h in hits2], default=0.0)
        _log(state, f"  2차(재구성) 쿼리: '{fallback_query}'")
        for h in hits2:
            _log(state, f"    - score={h['score']:.3f} source={h['metadata'].get('source')} text={h['text'][:80]}...")
        _log(state, f"  2차 top1 유사도: {top_score2:.3f}")
        if top_score2 > top_score:
            hits, top_score, primary_query = hits2, top_score2, fallback_query

    state["manual_query_used"] = primary_query
    state["manual_hits"] = hits
    state["manual_top_score"] = top_score
    state["manual_retry_used"] = retry_used
    return state


# --------------------------------------------------------------------------
# [2단계] Gate A: LLM Judge same/delta/novel
# --------------------------------------------------------------------------
def node_gate_a_judge(state: PipelineState) -> PipelineState:
    candidate = state["candidate"]
    llm = state["llm"]
    tacit_insight = candidate["knowledge"]["tacit_insight"]
    hits = state.get("manual_hits", [])

    result = llm_utils.judge_manual_relation(llm, tacit_insight, hits)
    state["gate_a_relation"] = result.get("relation", "novel")
    state["gate_a_justification"] = result.get("justification", "")
    state["gate_a_raw"] = result

    _log(state, "=" * 70)
    _log(state, "[2단계] Gate A - LLM Judge (same/delta/novel, 탈락조건)")
    _log(state, f"  relation = {state['gate_a_relation']}")
    _log(state, f"  justification = {state['gate_a_justification']}")

    if state["gate_a_relation"] == "same":
        state["decision"] = "reject"
        state["reject_reason"] = "Gate A: 매뉴얼과 동일(same) 판정 (2단계에서 STOP)"
        _log(state, f"  => STOP: reject ({state['reject_reason']})")
    return state


def route_after_gate_a(state: PipelineState) -> str:
    return "reject" if state["gate_a_relation"] == "same" else "gate_bc_scores"


# --------------------------------------------------------------------------
# [3단계] Gate B, C 점수 항목
# --------------------------------------------------------------------------
def _collect_transcript_snippets(transcript: List[dict], timestamps: List[str]) -> List[str]:
    snippets = []
    for ts in timestamps:
        ts_sec = rules.parse_timestamp(ts)
        if ts_sec is None:
            continue
        seg = rules.find_transcript_segment(ts_sec, transcript)
        if seg:
            snippets.append(f"[{ts}] {seg['text']}")
    return snippets


def node_gate_bc_scores(state: PipelineState) -> PipelineState:
    candidate = state["candidate"]
    llm = state["llm"]
    transcript = state.get("transcript", [])
    config = state["config"]

    _log(state, "=" * 70)
    _log(state, "[3단계] Gate B, C - LLM Judge 점수 항목 (서로 독립, 병렬 처리 가능)")

    # --- Gate B: action_reason_consistency ---
    b_result = llm_utils.score_action_reason_consistency(llm, candidate)
    state["action_reason_consistency"] = float(b_result.get("score", 0.0))
    state["action_reason_justification"] = b_result.get("justification", "")
    _log(state, f"  [Gate B] action_reason_consistency = {state['action_reason_consistency']:.3f}")
    _log(state, f"    justification: {state['action_reason_justification']}")

    # --- Gate C-1: reasoning_grounding ---
    reasoning_snippets = _collect_transcript_snippets(transcript, candidate["knowledge"].get("reasoning_source", []))
    c1_result = llm_utils.score_reasoning_grounding(llm, candidate, reasoning_snippets)
    state["reasoning_grounding"] = float(c1_result.get("score", 0.0))
    state["reasoning_grounding_justification"] = c1_result.get("justification", "")
    _log(state, f"  [Gate C-1] reasoning_grounding = {state['reasoning_grounding']:.3f}")
    _log(state, f"    참고 발화: {reasoning_snippets}")
    _log(state, f"    justification: {state['reasoning_grounding_justification']}")

    # --- Gate C-2: step_grounding_ratio ---
    steps = candidate["knowledge"].get("diagnostic_steps", [])
    utterance_steps = [s for s in steps if s.get("evidence") == "utterance"]
    step_details = []
    grounded_count = 0
    for s in utterance_steps:
        r = llm_utils.judge_step_grounded(llm, s)
        grounded = bool(r.get("grounded", False))
        if grounded:
            grounded_count += 1
        step_details.append({
            "order": s.get("order"), "action": s.get("action"), "grounded": grounded,
            "justification": r.get("justification", "")
        })
    ratio = grounded_count / len(utterance_steps) if utterance_steps else 1.0
    state["step_grounding_ratio"] = ratio
    state["step_grounding_detail"] = step_details
    _log(state, f"  [Gate C-2] step_grounding_ratio = {ratio:.3f} ({grounded_count}/{len(utterance_steps)} grounded)")
    for d in step_details:
        _log(state, f"    - step {d['order']}: grounded={d['grounded']} ({d['justification']})")

    # --- Gate C-3: utterance_signal (규칙기반 + LLM 보정) ---
    rule_score, rule_detail = rules.utterance_signal_rule_score(candidate, config)
    c3_result = llm_utils.score_utterance_signal(llm, candidate, rule_score, rule_detail)
    state["utterance_signal"] = float(c3_result.get("score", rule_score))
    state["utterance_signal_rule_detail"] = rule_detail
    state["utterance_signal_justification"] = c3_result.get("justification", "")
    _log(state, f"  [Gate C-3] utterance_signal (rule={rule_score:.3f}) -> final={state['utterance_signal']:.3f}")
    _log(state, f"    justification: {state['utterance_signal_justification']}")

    return state


# --------------------------------------------------------------------------
# [4단계] confidence 가중합 및 라우팅
# --------------------------------------------------------------------------
def node_confidence_route(state: PipelineState) -> PipelineState:
    config = state["config"]
    w = config.weights()
    t_high, t_low = config.thresholds()

    confidence = (
        w["reasoning_grounding"] * state["reasoning_grounding"]
        + w["step_grounding_ratio"] * state["step_grounding_ratio"]
        + w["action_reason_consistency"] * state["action_reason_consistency"]
        + w["utterance_signal"] * state["utterance_signal"]
    )
    state["confidence"] = confidence

    if confidence >= t_high:
        decision = "accept"
    elif confidence >= t_low:
        decision = "hold"
    else:
        decision = "reject"
    state["decision"] = decision

    _log(state, "=" * 70)
    _log(state, f"[4단계] confidence 가중합 (track={config.track})")
    _log(state, f"  weights = {w}")
    _log(state, f"  confidence = {w['reasoning_grounding']}*{state['reasoning_grounding']:.3f}"
                f" + {w['step_grounding_ratio']}*{state['step_grounding_ratio']:.3f}"
                f" + {w['action_reason_consistency']}*{state['action_reason_consistency']:.3f}"
                f" + {w['utterance_signal']}*{state['utterance_signal']:.3f}"
                f" = {confidence:.4f}")
    _log(state, f"  threshold: T_high={t_high}, T_low={t_low}")
    _log(state, f"  => 최종 결정: {decision.upper()}")
    return state


def node_reject_terminal(state: PipelineState) -> PipelineState:
    """0/2단계에서 조기 종료된 경우의 종착 노드 (confidence 계산 없이 reject로 확정)."""
    state["decision"] = "reject"
    state.setdefault("confidence", 0.0)
    return state


# --------------------------------------------------------------------------
# 그래프 빌드
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("timestamp_check", node_timestamp_check)
    graph.add_node("gate_a_search", node_gate_a_search)
    graph.add_node("gate_a_judge", node_gate_a_judge)
    graph.add_node("gate_bc_scores", node_gate_bc_scores)
    graph.add_node("confidence_route", node_confidence_route)
    graph.add_node("reject_terminal", node_reject_terminal)

    graph.set_entry_point("timestamp_check")

    graph.add_conditional_edges(
        "timestamp_check", route_after_timestamp,
        {"reject": "reject_terminal", "gate_a_search": "gate_a_search"},
    )
    graph.add_edge("gate_a_search", "gate_a_judge")
    graph.add_conditional_edges(
        "gate_a_judge", route_after_gate_a,
        {"reject": "reject_terminal", "gate_bc_scores": "gate_bc_scores"},
    )
    graph.add_edge("gate_bc_scores", "confidence_route")
    graph.add_edge("confidence_route", END)
    graph.add_edge("reject_terminal", END)

    return graph.compile()
