# -*- coding: utf-8 -*-
"""
6단계 품질 검증 파이프라인 실행 스크립트.

사용법
------
1) config.py 의 input_dir(기본: ./input) 에 암묵지 후보 JSON 파일들을 넣는다.
   각 JSON이 참조하는 transcript_ref 파일도 실제 존재해야 한다 (상대경로/절대경로 모두 허용,
   상대경로면 candidate JSON 파일이 있는 폴더 기준으로 찾는다).
2) manuals/ 폴더에 관련 매뉴얼 .txt(.md) 파일을 넣는다 (Gate A Manual RAG 용, 최소 1개 필요).
3) 그대로 실행:
       python main.py
   또는 특정 파일/폴더 지정:
       python main.py --input /path/to/candidate.json
       python main.py --input /path/to/dir --output_dir /path/to/out
   오프라인 구조 테스트(모델 다운로드 없이 파이프라인 흐름만 확인):
       python main.py --mock

출력
----
각 후보 JSON 하나당 output_dir 에 아래 2개 파일 생성:
  - <id>_result.txt   : 0~4단계 전체 중간 결과 + 최종 결과 상세 로그
  - <id>_verified.json: 원본 JSON + "verification" 블록 추가 (문서 6-8)
"""
import argparse
import glob
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from preflight import run_preflight
# embedding_utils / llm_utils / graph는 langgraph·langchain·torch·transformers를
# 끌고 온다. run_preflight()가 통과하기 전에는 import하지 않는다 — 2026-07-03
# 실사고: 여기서 무조건 import했다가 langgraph가 없어서 run() 진입도 못하고
# 죽었고, run() 안에 프리플라이트를 넣어도 이미 늦어서 무용지물이었다.


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_transcript_path(candidate_path: str, candidate: dict) -> str:
    ref = candidate["metadata"]["source"].get("transcript_ref")
    if not ref:
        return ""
    if os.path.isabs(ref) and os.path.exists(ref):
        return ref
    # candidate 파일 기준 상대경로
    base_dir = os.path.dirname(os.path.abspath(candidate_path))
    cand1 = os.path.join(base_dir, ref)
    if os.path.exists(cand1):
        return cand1
    # config.input_dir 기준
    cand2 = os.path.join(CONFIG.input_dir, ref)
    if os.path.exists(cand2):
        return cand2
    # 프로젝트 루트 기준
    cand3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), ref)
    if os.path.exists(cand3):
        return cand3
    return ""


def load_transcript(candidate_path: str, candidate: dict) -> list:
    path = resolve_transcript_path(candidate_path, candidate)
    if not path:
        return []
    data = load_json(path)
    # {"segments": [...]} 형태와 [...] 형태 둘 다 지원
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    return []


def find_candidate_files(input_path: str) -> list:
    if os.path.isfile(input_path):
        return [input_path]
    return sorted(glob.glob(os.path.join(input_path, "*.json")))


def format_final_summary(state: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("[최종 결과 요약]")
    lines.append(f"  decision        : {state.get('decision', '?').upper()}")
    if "confidence" in state:
        lines.append(f"  confidence      : {state['confidence']:.4f}")
    if state.get("reject_reason"):
        lines.append(f"  reject_reason   : {state['reject_reason']}")
    if "gate_a_relation" in state:
        lines.append(f"  gate_a_relation : {state['gate_a_relation']}")
    for key in ["action_reason_consistency", "reasoning_grounding", "step_grounding_ratio", "utterance_signal"]:
        if key in state:
            lines.append(f"  {key:<24}: {state[key]:.3f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def build_verification_block(state: dict) -> dict:
    """문서 6-8 '검증 내용 추가' 자리에 들어갈 구조화된 결과."""
    return {
        "decision": state.get("decision"),
        "confidence": state.get("confidence"),
        "reject_reason": state.get("reject_reason"),
        "timestamp_validity": {
            "passed": state.get("timestamp_valid"),
        },
        "gate_a_manual_comparison": {
            "relation": state.get("gate_a_relation"),
            "justification": state.get("gate_a_justification"),
            "manual_query_used": state.get("manual_query_used"),
            "manual_top_score": state.get("manual_top_score"),
            "manual_retry_used": state.get("manual_retry_used"),
        },
        "scores": {
            "action_reason_consistency": state.get("action_reason_consistency"),
            "reasoning_grounding": state.get("reasoning_grounding"),
            "step_grounding_ratio": state.get("step_grounding_ratio"),
            "utterance_signal": state.get("utterance_signal"),
        },
        # 침묵 트랙(2026-07-09): 실제 가중합에 쓰인 벡터를 기록. 발화 트랙은 기존 그대로.
        "weight_track": state.get("weight_track", "utterance"),
        "weights_used": (CONFIG.weights_silent()
                         if state.get("weight_track") == "silent" else CONFIG.weights()),
        "thresholds_used": {"t_high": CONFIG.thresholds()[0], "t_low": CONFIG.thresholds()[1]},
        "track": CONFIG.track,
    }


def run():
    parser = argparse.ArgumentParser(description="암묵지 후보 6단계 품질 검증 파이프라인")
    parser.add_argument("--input", default=CONFIG.input_dir, help="후보 JSON 파일 또는 폴더 경로")
    parser.add_argument("--output_dir", default=CONFIG.output_dir, help="결과 저장 폴더")
    parser.add_argument("--mock", action="store_true", help="LLM/임베딩을 더미로 대체해 오프라인으로 구조만 테스트")
    args = parser.parse_args()

    if args.mock:
        CONFIG.mock_mode = True
    CONFIG.output_dir = args.output_dir

    run_preflight(require_gpu=not CONFIG.mock_mode)

    # 프리플라이트 통과 후에만 무거운 모듈을 import한다(무거운 거 위/아래 경계).
    import embedding_utils
    import llm_utils
    from graph import build_graph

    os.makedirs(CONFIG.output_dir, exist_ok=True)

    candidate_files = find_candidate_files(args.input)
    if not candidate_files:
        print(f"[오류] 입력 경로 '{args.input}' 에서 JSON 파일을 찾지 못했습니다.")
        sys.exit(1)

    print(f"[초기화] 임베딩 모델 로딩: {CONFIG.embedding_model_name} (mock={CONFIG.mock_mode})")
    embeddings = embedding_utils.build_embeddings(CONFIG)
    print(f"[초기화] 매뉴얼 벡터스토어(Qdrant) 구축: {CONFIG.manual_dir}")
    vectorstore = embedding_utils.build_manual_vectorstore(CONFIG, embeddings)
    print(f"[초기화] LLM 로딩: {CONFIG.llm_model_name} (mock={CONFIG.mock_mode})")
    llm = llm_utils.build_llm(CONFIG)

    graph = build_graph()

    print(f"[실행] 후보 {len(candidate_files)}건 처리 시작")
    for path in candidate_files:
        print(f"  - {path}")
        try:
            candidate = load_json(path)
            transcript = load_transcript(path, candidate)
            if not transcript:
                print(f"    [경고] transcript_ref를 찾지 못했습니다. timestamp 실재 검사가 실패 처리될 수 있습니다.")

            init_state = {
                "candidate": candidate,
                "transcript": transcript,
                "config": CONFIG,
                "llm": llm,
                "vectorstore": vectorstore,
                "log": [f"후보 파일: {path}", f"후보 ID: {candidate.get('id')}"],
            }
            final_state = graph.invoke(init_state)

            cand_id = candidate.get("id", os.path.splitext(os.path.basename(path))[0])

            # 1) 상세 로그 txt 저장 (중간결과 + 최종결과 모두)
            log_text = "\n".join(final_state.get("log", [])) + "\n\n" + format_final_summary(final_state)
            txt_path = os.path.join(CONFIG.output_dir, f"{cand_id}_result.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(log_text)

            # 2) 검증 결과가 반영된 JSON 저장 (문서 6-8)
            verified = dict(candidate)
            verified["verification"] = build_verification_block(final_state)
            json_path = os.path.join(CONFIG.output_dir, f"{cand_id}_verified.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(verified, f, ensure_ascii=False, indent=2)

            print(f"    => decision={final_state.get('decision')} "
                  f"confidence={final_state.get('confidence', 0):.4f}  (저장: {txt_path})")

        except Exception as e:
            err_txt = os.path.join(
                CONFIG.output_dir,
                f"{os.path.splitext(os.path.basename(path))[0]}_ERROR.txt",
            )
            with open(err_txt, "w", encoding="utf-8") as f:
                f.write(f"처리 중 오류 발생: {e}\n\n")
                f.write(traceback.format_exc())
            print(f"    [오류] {e} (상세: {err_txt})")

    print(f"[완료] 결과 저장 위치: {CONFIG.output_dir}")


if __name__ == "__main__":
    run()
