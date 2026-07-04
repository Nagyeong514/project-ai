# -*- coding: utf-8 -*-
"""
LLM(Qwen2.5-14B-Instruct, 오픈소스) 로딩 및 게이트별 judge 프롬프트/파싱.

설계 원칙 (문서 6-2 #1): 에이전트(LLM)는 '탐색/판단'만 하고 사실을 새로 생성하지 않는다.
모든 judge 프롬프트는 "입력에 없는 내용을 추론해서 만들어내지 마라"를 명시적으로 강제한다.
"""
import json
import re
from typing import Dict, Any, List, Optional


# --------------------------------------------------------------------------
# LLM 래퍼
# --------------------------------------------------------------------------
class MockLLM:
    """오프라인/구조 테스트용 더미 LLM. 실제 판정에는 사용하지 말 것 (mock_mode=True 일 때만)."""

    def generate(self, system: str, user: str) -> str:
        # 매우 단순한 규칙 기반 더미 응답 (질문 텍스트에 포함된 키워드로 대충 흉내)
        if "same/delta/novel" in user or "relation" in user:
            return json.dumps({"relation": "delta", "justification": "[MOCK] 매뉴얼과 유사하나 접점 세척 등 차이가 있음"}, ensure_ascii=False)
        if "action_reason_consistency" in user or "인과적으로 맞는가" in user:
            return json.dumps({"score": 0.8, "justification": "[MOCK] 행동과 이유가 대체로 일치"}, ensure_ascii=False)
        if "reasoning_grounding" in user:
            return json.dumps({"score": 0.7, "justification": "[MOCK] 근거 발화와 reasoning이 유사"}, ensure_ascii=False)
        if "grounded" in user:
            return json.dumps({"grounded": True, "justification": "[MOCK] source_utterance가 action을 뒷받침함"}, ensure_ascii=False)
        return json.dumps({"score": 0.5, "justification": "[MOCK] 기본값"}, ensure_ascii=False)


class QwenLLM:
    """Qwen2.5-14B-Instruct 로컬(오픈소스) 추론. HuggingFace transformers 사용, API 미사용."""

    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 800, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # 2026-07-03: bfloat16 -> float16. 이 클러스터 GPU(RTX6000, Turing/sm75)는
            # bf16 텐서코어 미지원이라 STEP3에서 이미 "fp16/4bit만, bf16 금지"로 확정된 제약
            # (reference-cluster-specs). 원래 bf16이었던 건 다른(Ampere+) 환경에서 짠 흔적으로 추정.
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def build_llm(config):
    if config.mock_mode:
        return MockLLM()
    return QwenLLM(
        model_name=config.llm_model_name,
        device=config.device,
        max_new_tokens=config.llm_max_new_tokens,
        temperature=config.llm_temperature,
    )


# --------------------------------------------------------------------------
# 공통: LLM 출력에서 JSON만 뽑아내는 파서
# --------------------------------------------------------------------------
def _extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    # ```json ... ``` 코드블록 제거
    raw = re.sub(r"^```(json)?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    # 가장 바깥 { ... } 블록만 추출 시도
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = match.group(0) if match else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_output": raw}


HALLUCINATION_GUARD = (
    "당신은 엄격한 품질 검증관입니다. 반드시 아래 규칙을 지키세요:\n"
    "1) 입력(VLM 행동 정보, STT 발화, 매뉴얼 발췌)에 명시되지 않은 사실을 추론하거나 새로 만들어내지 마세요.\n"
    "2) 근거가 부족하면 낮은 점수를 주고, 그 이유를 명시하세요. 없는 근거를 지어내지 마세요.\n"
    "3) 출력은 반드시 지정된 JSON 형식 하나만 반환하세요. 다른 설명 문장을 절대 추가하지 마세요."
)


# --------------------------------------------------------------------------
# [게이트 A] Manual Comparison 판정 (same / delta / novel)
# --------------------------------------------------------------------------
def judge_manual_relation(llm, tacit_insight: str, manual_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    manual_text = "\n".join(
        f"- (유사도 {h['score']:.3f}, 출처 {h['metadata'].get('source','?')}) {h['text']}" for h in manual_hits
    ) or "(검색된 매뉴얼 내용 없음)"

    user = f"""다음은 숙련자의 암묵지 후보 한 문장과, 그와 관련해 검색된 매뉴얼 발췌입니다.

[암묵지 후보 - tacit_insight]
{tacit_insight}

[검색된 매뉴얼 발췌 (top-k)]
{manual_text}

이 tacit_insight가 매뉴얼과 어떤 relation/relation을 갖는지 판정하세요:
- "same": 매뉴얼 내용과 사실상 동일함 (암묵지로서 가치 없음)
- "delta": 매뉴얼에 있는 절차이지만 명장이 다르게/추가로 수행함
- "novel": 매뉴얼에 아예 없는 노하우

아래 JSON 형식으로만 답하세요:
{{"relation": "same|delta|novel", "justification": "판단 근거 (한국어, 1~2문장)"}}"""
    raw = llm.generate(HALLUCINATION_GUARD, user)
    parsed = _extract_json(raw)
    parsed.setdefault("relation", "novel")
    parsed.setdefault("justification", "")
    parsed["raw_output"] = raw
    return parsed


# --------------------------------------------------------------------------
# [게이트 B] Action-Reason Consistency (0~1 점수)
# --------------------------------------------------------------------------
def score_action_reason_consistency(llm, candidate: dict) -> Dict[str, Any]:
    know = candidate["knowledge"]
    steps_text = "\n".join(f"  {s['order']}. {s['action']}" for s in know.get("diagnostic_steps", []))
    user = f"""다음 진단 절차(action)와 reasoning(이유)이 action_reason_consistency(행동과 이유가 인과적으로 맞는가) 관점에서 얼마나 일치하는지 0~1 사이 점수로 평가하세요.
입력에 없는 내용을 추가로 상상해서 판단하지 마세요.

[situation]
{know['situation']}

[diagnostic_steps]
{steps_text}

[reasoning]
{know['reasoning']}

[tacit_insight]
{know['tacit_insight']}

행동들이 reasoning이 설명하는 인과관계와 논리적으로 이어지면 높은 점수(예: 0.8~1.0),
행동은 있지만 reasoning과 논리적 연결이 약하거나 비약이 있으면 낮은 점수(예: 0.0~0.4)를 주세요.

아래 JSON 형식으로만 답하세요:
{{"score": 0.0, "justification": "판단 근거 (한국어, 1~2문장)"}}"""
    raw = llm.generate(HALLUCINATION_GUARD, user)
    parsed = _extract_json(raw)
    parsed.setdefault("score", 0.0)
    parsed.setdefault("justification", "")
    parsed["raw_output"] = raw
    return parsed


# --------------------------------------------------------------------------
# [게이트 C-1] reasoning_grounding (0~1 점수): reasoning이 실제 발화에서 나왔는가
# --------------------------------------------------------------------------
def score_reasoning_grounding(llm, candidate: dict, transcript_snippets: List[str]) -> Dict[str, Any]:
    know = candidate["knowledge"]
    snippets_text = "\n".join(f"- {s}" for s in transcript_snippets) or "(reasoning_source 근처 transcript 없음)"
    user = f"""아래 reasoning 문장이, 주어진 실제 발화(transcript 발췌)들로부터 도출 가능한지 평가하세요.
reasoning_origin 필드값도 참고하되, 최종 판단은 실제 발화 내용과의 부합 정도로 내리세요.
발화에 없는 일반상식으로 그럴듯하게 채운 것이라면 낮은 점수를 주세요 (model_inferred 성격).

[reasoning]
{know['reasoning']}

[reasoning_origin 필드값]
{know['reasoning_origin']}

[reasoning_source 근처 실제 발화(transcript 발췌)]
{snippets_text}

아래 JSON 형식으로만 답하세요:
{{"score": 0.0, "justification": "판단 근거 (한국어, 1~2문장)"}}"""
    raw = llm.generate(HALLUCINATION_GUARD, user)
    parsed = _extract_json(raw)
    parsed.setdefault("score", 0.0)
    parsed.setdefault("justification", "")
    parsed["raw_output"] = raw
    return parsed


# --------------------------------------------------------------------------
# [게이트 C-2] step_grounding_ratio 를 위한 per-step grounded 여부 판정
# --------------------------------------------------------------------------
def judge_step_grounded(llm, step: dict) -> Dict[str, Any]:
    user = f"""다음 진단 스텝의 action이, 주어진 source_utterance(실제 발화)로부터 정당하게 근거되는지(grounded) 판정하세요.
발화에 없는 내용을 action에 추가했다면 grounded=false 로 판정하세요.

[action]
{step['action']}

[source_utterance]
{step.get('source_utterance')}

아래 JSON 형식으로만 답하세요:
{{"grounded": true/false, "justification": "판단 근거 (한국어, 1문장)"}}"""
    raw = llm.generate(HALLUCINATION_GUARD, user)
    parsed = _extract_json(raw)
    parsed.setdefault("grounded", False)
    parsed.setdefault("justification", "")
    parsed["raw_output"] = raw
    return parsed


# --------------------------------------------------------------------------
# [게이트 C-3] utterance_signal 의 LLM 판단 보정 (규칙기반 rule_score + action_only 비율 감안)
# --------------------------------------------------------------------------
def score_utterance_signal(llm, candidate: dict, rule_score: float, rule_detail: dict) -> Dict[str, Any]:
    user = f"""utterance_signal은 '근거 발화에 인과/이탈/주의/부정 마커가 실제로 담긴 정도'를 뜻합니다.
규칙기반 마커 탐지 결과와 전체 스텝 구성을 참고하여 최종 utterance_signal 점수(0~1)를 산정하세요.
action_only 스텝은 애초에 발화가 없으므로 페널티를 주지 말고, 발화가 있는 스텝들의 신호 품질만으로 판단하세요.

[규칙기반 탐지 결과]
- 전체 스텝 수: {rule_detail['total_steps']}
- 발화 있는(utterance) 스텝 수: {rule_detail['utterance_steps']}
- action_only 스텝 수: {rule_detail['action_only_steps']}
- 마커가 발견된 스텝 수: {rule_detail['marker_hit_steps']}
- 규칙기반 raw score: {rule_score:.3f}

[스텝별 발화 및 매칭된 마커]
{json.dumps(rule_detail['per_step'], ensure_ascii=False, indent=2)}

아래 JSON 형식으로만 답하세요:
{{"score": 0.0, "justification": "판단 근거 (한국어, 1~2문장)"}}"""
    raw = llm.generate(HALLUCINATION_GUARD, user)
    parsed = _extract_json(raw)
    parsed.setdefault("score", rule_score)
    parsed.setdefault("justification", "")
    parsed["raw_output"] = raw
    return parsed
