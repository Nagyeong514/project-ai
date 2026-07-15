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
    """Qwen 계열 로컬(오픈소스) judge 추론. HuggingFace transformers 사용, API 미사용.
    (2026-07-12: Qwen2.5-14B → Qwen3-14B 교체 — config.llm_model_name 주석 참고)"""

    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 800, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 2026-07-12: Qwen3 계열은 hybrid thinking이 기본 — <think> 블록이 judge JSON 파싱을
        # 오염시키므로 끈다(STEP5 llm_fusion._load와 동일한 주입 패턴, Qwen2.5 등엔 no-op).
        if "qwen3" in model_name.lower():
            _orig_act = self.tokenizer.apply_chat_template

            def _no_think(*a, **k):
                k.setdefault("enable_thinking", False)
                return _orig_act(*a, **k)

            self.tokenizer.apply_chat_template = _no_think
            print(f"[LLM] {model_name}: enable_thinking=False 주입")
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
def judge_manual_relation(llm, tacit_insight: str, manual_hits: List[Dict[str, Any]],
                          reasoning: str = "") -> Dict[str, Any]:
    # 2026-07-09 기준 개정(run5 오판 후속, 사용자 승인): 구버전은 "주제·항목이 매뉴얼에
    # 존재"만으로 same을 때렸다 — LED 코드표(황1백3/황2/백지속)가 매뉴얼에 있다는 이유로,
    # 표에 없는 패턴(황4백5)을 판독한 후보(tk_CLIP2_001, GT A3·A4 담지)를 same-reject
    # 0.0 처리. 개정 2가지: ① 구체 패턴·수치·절차가 '명시'된 경우만 same, 표에 없는
    # 패턴의 판독·해석은 delta/novel ② insight 한 문장에는 구체 패턴이 없을 수 있어
    # reasoning을 참고 입력으로 함께 준다(없으면 기존과 동일). 매뉴얼 원문은 무수정.
    # 2026-07-10 재개정(run3 오판 후속): 07-09 개정의 "완전 재구성" 문턱에 judge가
    # 앵커링해 same을 회피 — ESC·SysInfo 등 기본 조작 5건이 "조건을 강조" "더 포괄적"
    # 같은 무지목 논리로 delta 처리됨(run3 same 1/6건). 조치: ① "완전 재구성" 기준
    # 명시적 금지 ② "일반적/포괄적·표현 차이·자명한 이유"를 delta 금지 근거로 열거
    # ③ delta 판정 시 매뉴얼에 없는 새 정보를 justification에 지목 강제(못 찾으면 same)
    # ④ "애매하면 same 아닌 쪽" 문구 삭제. LED 표 밖 패턴 보호(07-09 ①)는 규칙 4로 존치.
    # 2026-07-10 2차 재개정(run4 오판 후속): 재개정이 과잉 교정 — same 남발로 진짜
    # 암묵지가 사살됨. ① tk_CLIP2_004(래치 동시 탈거, "안 그러면 슬롯 파손")를 매뉴얼
    # 2-3항(장착 시 래치)과 same 처리 — 작업 방향(장착↔탈거)과 비자명 위험 인과를 새
    # 정보로 안 침 ② 반대로 잡것 2건(케이블 정리, 전원 시 LED 확인)은 매뉴얼 2-4/2-2항에
    # 그대로 있는데 delta 생존. 조치: 규칙 5 신설(reasoning 속 '비자명한 구체 위험·결과'
    # 는 새 정보 — 막연한 우려는 제외), 규칙 6 신설(매뉴얼이 규정한 상태의 단순 확인은
    # same, 단 확인의 방법·감각 기준이 다르면 delta). delta 정의 예시에도 두 유형 반영.
    # 2026-07-10 3차(run5 핀포인트 2건, 최종): ① judge가 장착 절차의 "양쪽 래치" 문구로
    # 탈거 방법("동시에"+슬롯 파손)까지 규정된 것으로 재구성 → tk_CLIP2_004 same 오판.
    # 규칙 5에 "언급만 된 조작의 수행 방법(동시에/순서/부위/힘)은 delta" + 타 절차
    # 재구성 금지 추가. ② "결함이 있을 수 있다"류 막연 우려가 규칙 5의 구체 위험으로
    # 오인 → tk_CLIP1_009 delta 오판(0.76 accept 잔류). 막연 우려 배제 예시에 실제
    # 걸린 표현을 자구로 추가하고 구체 위험 대비 예시(슬롯 파손 등)를 병기.
    manual_text = "\n".join(
        f"- (유사도 {h['score']:.3f}, 출처 {h['metadata'].get('source','?')}) {h['text']}" for h in manual_hits
    ) or "(검색된 매뉴얼 내용 없음)"

    reasoning_block = (f"\n[후보의 reasoning (참고 — 구체 패턴·수치 대조용)]\n{reasoning}\n"
                       if reasoning else "")

    user = f"""다음은 숙련자의 암묵지 후보 한 문장과, 그와 관련해 검색된 매뉴얼 발췌입니다.

[암묵지 후보 - tacit_insight]
{tacit_insight}
{reasoning_block}
[검색된 매뉴얼 발췌 (top-k)]
{manual_text}

이 tacit_insight가 매뉴얼과 어떤 relation을 갖는지 판정하세요:
- "same": 후보가 말하는 조작·절차·내용이 매뉴얼에 이미 규정되어 있고, 후보가
  매뉴얼 대비 추가하는 고유 정보(새 패턴, 수치, 순서, 조건, 도구, 감각 기준,
  판독)가 없음. 표현이 다르거나 후보가 더 짧고 일반적이어도, 새 정보가 없으면
  same입니다.
- "delta": 매뉴얼에 관련 절차·항목은 있으나, 후보가 매뉴얼에 없는 구체 정보를
  실제로 추가함 (예: 매뉴얼 표에 없는 LED 패턴의 판독, 매뉴얼이 규정하지 않은
  도구나 감각 기준, 매뉴얼과 다른 순서, 매뉴얼이 다루지 않는 작업 방향이나
  상황에 대한 지식, 절차를 어겼을 때의 비자명한 구체적 위험·결과의 명시)
- "novel": 매뉴얼에 관련 내용 자체가 없는 노하우

판정 규칙:
1) same 판정에 "매뉴얼 문장만으로 후보를 완전히 재구성할 수 있는가"를 기준으로
   삼지 마세요. 볼 것은 하나입니다 — 후보에 매뉴얼 밖의 새 정보가 있는가.
   없으면 same입니다.
2) 다음은 delta의 근거가 될 수 없습니다:
   - 후보가 매뉴얼보다 "더 일반적이다", "더 포괄적이다" (일반적 = 추가 정보
     없음 = same)
   - 표현, 어순, 강조가 다르다 (강조는 내용 추가가 아님)
   - 후보가 그 조작의 자명한 이유나 필요성을 덧붙인다 ("누르지 않으면 이동할
     수 없다" 류는 새 정보가 아님)
3) delta로 판정하려면 justification에 "매뉴얼에 없는 새 정보가 정확히
   무엇인지"를 한 구절로 지목하세요 (예: "지우개를 이용한 접점 세척은 매뉴얼
   유의사항이 정식 절차가 아니라고 명시한 별도 노하우"). 지목할 새 정보를
   찾지 못하면 그것이 곧 same이라는 뜻입니다.
4) 예외 (보수적 보호, 유지): 매뉴얼에 관련 표나 코드가 있어도, 후보가 다루는
   특정 패턴·수치가 그 표에 없으면 same이 아니라 delta입니다. 이 경우
   justification에 해당 패턴이 표에 없음을 지목하세요.
5) "새 정보"는 tacit_insight 한 문장만이 아니라 [후보의 reasoning]에서도
   찾으세요. 매뉴얼이 절차만 규정하고, 그 절차를 지키지 않았을 때 생기는
   구체적 결과(특정 부품의 파손, 통전 불량, 성능 절반 저하 등)를 매뉴얼이
   명시하지 않았다면, 그 위험 지식을 담은 후보는 delta입니다. 매뉴얼이 어떤
   조작을 한 방향(예: 장착)에 대해서만 규정하고 후보는 다른 방향(예: 탈거)의
   주의점을 다루는 경우도 delta입니다. 또한 매뉴얼이 어떤 조작을 절차로
   언급만 하고(예: "슬롯에서 분리한다") 그 조작의 수행 방법 — 동시에, 특정
   순서로, 특정 부위를 잡고, 특정 힘으로 — 을 규정하지 않았다면, 그 방법과
   이를 어겼을 때의 위험을 담은 후보는 delta입니다. 매뉴얼의 다른 절차(예:
   장착)에 비슷한 단어가 나온다는 이유로 이 방법이 규정된 것으로 재구성하지
   마세요. 단, 다음 두 가지는 위험 지식이 아닙니다:
   - 조작의 정의에서 자명하게 따라오는 결과 ("누르지 않으면 이동할 수 없다",
     "켜지지 않으면 작업을 시작할 수 없다" 류 — 규칙 2 유지)
   - "문제가 발생할 수 있다", "불안정해질 수 있다", "결함이 있을 수 있다",
     "전원 공급 문제일 수 있다" 같은 막연하고 일반적인 우려 — 어느 부품이
     어떻게 손상·상실되는지의 지목(예: 슬롯 파손, 통전 불량, 성능 절반
     저하)이 없으면 새 정보가 아닙니다.
6) 후보가 매뉴얼이 이미 규정한 상태·현상·작업을 단순히 "확인한다/점검한다/
   유지한다"고 말하는 것은 새 정보가 아닙니다 (예: 매뉴얼에 "전원 인가 시
   LED가 점등된다"가 있으면 "전원을 누르고 LED가 반응하는지 확인한다"는 same,
   매뉴얼에 "남는 케이블은 정리한다"가 있으면 "케이블을 정리해 내부를 깔끔하게
   유지한다"는 same). 단, 확인하는 방법이나 감각 기준이 매뉴얼과 다르면
   (매뉴얼은 육안 확인만 규정하는데 손으로 당기고 흔들어 보는 촉각 점검,
   매뉴얼은 래치 잠금만 규정하는데 소리로 잠금을 판정) 그 방법·기준이 새
   정보이므로 delta입니다.

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