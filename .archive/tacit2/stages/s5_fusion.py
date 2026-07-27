"""S5: 정렬 + LLM 융합 + evidence 코드 부여.

v2 의 두 가지 구조 변경:
1) aligner 에 merge_cap_sec 상한 — 도미노 병합으로 클립 전체(-2~194초)가
   윈도우 1개로 뭉치던 버그(STEP5 OOM 의 진짜 원인) 차단. 프롬프트 크기도 자동으로 작아짐.
2) evidence/source_utterance/timestamp 는 LLM 이 아니라 코드가 채운다.
   LLM 은 source_hint(근거로 삼은 입력 원문)만 복사하고,
   코드가 그 hint 를 실제 관찰/발화와 대조해:
     - 관찰 매칭 → timestamp = 그 행동의 실제 시각
     - 그 시각 ±window_sec 안에 실제 발화 존재 → evidence=utterance + 그 발화 원문
     - 아니면 → evidence=action_only + source_utterance=null
   ("시작합니다"를 근거로 갖다 붙이던 오태깅의 구조적 해결 — 애초에 안 시킨다)
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from core.config import Cfg
from core.gpu import log_mem, release, set_alloc_conf
from core.paths import Contract, clip_id
from core.schema import (SCHEMA_VERSION, AlignedWindow, DiagnosticStep, EvidenceType,
                         Metadata, Observation, Source, TacitKnowledgeCandidate,
                         TacitKnowledgeDocument, Transcript, Utterance)
from core.timeline import sec_to_hhmmss
from prompts.fusion_prompt import build_fusion_messages


# ─────────────────────────── 정렬 (병합 상한) ───────────────────────────

def align(actions: List[Observation], utts: List[Utterance],
          window_sec: float, merge_cap: float) -> List[AlignedWindow]:
    windows: List[AlignedWindow] = []
    for a in sorted(actions, key=lambda o: o.timestamp):
        end = a.end_timestamp or a.timestamp
        w = AlignedWindow(window_start=max(0.0, a.timestamp - window_sec),
                          window_end=end + window_sec, actions=[a])
        if windows:
            last = windows[-1]
            overlap = w.window_start <= last.window_end
            capped = (max(w.window_end, last.window_end) - last.window_start) <= merge_cap
            if overlap and capped:  # ★ 겹쳐도 상한 넘으면 병합 안 함
                last.window_end = max(last.window_end, w.window_end)
                last.actions.append(a)
                continue
        windows.append(w)

    used = set()
    for w in windows:
        for i, u in enumerate(utts):
            if u.start < w.window_end and u.end > w.window_start:
                w.utterances.append(u)
                used.add(i)

    for i, u in enumerate(utts):  # 어느 윈도우에도 안 걸린 발화 → 독립 후보 재료
        if i not in used and not u.repeat_hallucination:
            windows.append(AlignedWindow(window_start=u.start, window_end=u.end,
                                         utterances=[u]))
    return sorted(windows, key=lambda w: w.window_start)


def serialize_windows(windows: List[AlignedWindow]) -> list:
    payload = []
    for w in windows:
        if w.case == "empty":
            continue
        payload.append({
            "case": w.case,
            "window_start": sec_to_hhmmss(w.window_start),
            "window_end": sec_to_hhmmss(w.window_end),
            "actions": [{
                "timestamp": sec_to_hhmmss(a.timestamp),
                "actor": a.actor, "action": a.action,
                "objects_visible": a.objects,
                **({"repeat_count": a.repeat_count,
                    "duration": f"{sec_to_hhmmss(a.timestamp)}~{sec_to_hhmmss(a.end_timestamp)}"}
                   if a.repeat_count > 1 and a.end_timestamp else {}),
            } for a in w.actions],
            "utterances": [{
                "start": sec_to_hhmmss(u.start),
                "text": u.normalized_text or u.raw_text,
                **({"repeat_hallucination": True} if u.repeat_hallucination else {}),
            } for u in w.utterances],
        })
    return payload


# ─────────────────────────── evidence 코드 부여 ───────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[\s\W]+", "", s or "").lower()


def _match(hint: str, text: str) -> bool:
    a, b = _norm(hint), _norm(text)
    if not a or not b:
        return False
    if len(a) < 8 or len(b) < 8:  # 짧은 문자열 부분매칭 오탐 방지
        return a == b
    return a in b or b in a


def ground_step(step_dict: dict, order: int, actions: List[Observation],
                utts: List[Utterance], window_sec: float) -> DiagnosticStep:
    hint = str(step_dict.get("source_hint", "") or "")
    action_text = str(step_dict.get("action", "") or "")

    ts: Optional[float] = None
    for a in actions:  # 1) hint 또는 action 이 실제 관찰과 매칭 → 그 행동의 진짜 시각
        if _match(hint, a.action) or _match(action_text, a.action):
            ts = a.timestamp
            break
    matched_utt: Optional[Utterance] = None
    for u in utts:  # 2) hint 가 실제 발화와 매칭
        if _match(hint, u.raw_text) or _match(hint, u.normalized_text):
            matched_utt = u
            if ts is None:
                ts = u.start
            break

    # 3) evidence 판정: 그 시각 ±window_sec 안의 '실제' 발화만 근거로 인정
    evidence, src = EvidenceType.action_only, None
    if matched_utt is not None and (ts is None or abs(matched_utt.start - ts) <= window_sec):
        evidence, src = EvidenceType.utterance, matched_utt.raw_text
        ts = ts if ts is not None else matched_utt.start
    elif ts is not None:
        near = [u for u in utts
                if abs(u.start - ts) <= window_sec and not u.repeat_hallucination]
        if near:
            evidence, src = EvidenceType.utterance, near[0].raw_text

    return DiagnosticStep(
        order=order, action=action_text, evidence=evidence,
        source_utterance=src,
        timestamp=sec_to_hhmmss(ts) if ts is not None else None,
    )


# ─────────────────────────── LLM ───────────────────────────

class FusionLLM:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.model = None
        self.tok = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        f = self.cfg.fusion
        kwargs = {"device_map": f.device}
        if f.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float16
        print(f"[S5] loading LLM {f.model_name} → {f.device}")
        self.tok = AutoTokenizer.from_pretrained(f.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(f.model_name, **kwargs)
        log_mem("S5 loaded")

    def infer(self, messages) -> str:
        import torch

        f = self.cfg.fusion
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=f.max_new_tokens,
                do_sample=f.temperature > 0, temperature=max(f.temperature, 1e-5))
        return self.tok.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]

    def unload(self):
        release(self.model, self.tok)
        self.model = self.tok = None
        log_mem("S5 released")


def _parse_json(raw: str) -> dict:
    text = re.sub(r"```(?:json)?", "", raw).strip()
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        raise ValueError("JSON 객체 없음")
    return json.loads(text[s:e + 1])


# ─────────────────────────── 스테이지 진입 ───────────────────────────

def run(cfg: Cfg, force: bool = False):
    set_alloc_conf()
    contract = Contract(cfg.paths.output_dir)
    contract.ensure_dirs()

    todo = []
    for v in cfg.paths.videos:
        cid = clip_id(v)
        if not force and contract.tacit(cid).exists():
            print(f"[S5] skip (exists): {cid}")
            continue
        if not (contract.transcript(cid).exists() and contract.observations(cid).exists()):
            print(f"[S5] WARN: {cid} 입력 없음(S1/S4 먼저). 건너뜀")
            continue
        todo.append(cid)
    if not todo:
        return

    llm = FusionLLM(cfg)
    llm.load()
    try:
        for cid in todo:
            tr = Transcript.model_validate(
                json.loads(contract.transcript(cid).read_text(encoding="utf-8")))
            obs_doc = json.loads(contract.observations(cid).read_text(encoding="utf-8"))
            actions = [Observation.model_validate(o) for o in obs_doc["observations"]]

            windows = align(actions, tr.utterances,
                            cfg.fusion.window_sec, cfg.fusion.merge_cap_sec)
            payload = serialize_windows(windows)
            print(f"[S5] {cid}: {len(actions)} actions + {len(tr.utterances)} utts "
                  f"→ {len(payload)} windows")

            messages = build_fusion_messages(cid, payload)
            obj, last_err = None, None
            for attempt in range(cfg.fusion.max_retries + 1):
                raw = llm.infer(messages)
                try:
                    obj = _parse_json(raw)
                    assert isinstance(obj.get("candidates"), list), "candidates 배열 없음"
                    break
                except Exception as e:
                    last_err = e
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user",
                         "content": f"JSON 파싱/구조 오류: {e}. 스키마를 지켜 JSON만 다시 출력하라."},
                    ]
            if obj is None:
                raise RuntimeError(f"[S5] {cid}: LLM {cfg.fusion.max_retries + 1}회 실패: {last_err}")

            # evidence/timestamp 는 코드가 부여, metadata 는 코드가 채움
            duration = actions[-1].end_timestamp or actions[-1].timestamp if actions else 0.0
            candidates = []
            stem = cid.rsplit(".", 1)[0].replace("-", "_")
            for i, c in enumerate(obj["candidates"], start=1):
                k = c.get("knowledge", {}) or {}
                steps = [ground_step(sd, j + 1, actions, tr.utterances, cfg.fusion.window_sec)
                         for j, sd in enumerate(k.get("diagnostic_steps", []) or [])]
                origin = k.get("reasoning_origin", "model_inferred")
                candidates.append(TacitKnowledgeCandidate(
                    id=f"tk_{stem}_{i:03d}",
                    metadata=Metadata(
                        scenario_id=stem,
                        source=Source(video_id=cid, clip_start="00:00:00",
                                      clip_end=sec_to_hhmmss(duration),
                                      transcript_ref=str(contract.transcript(cid)))),
                    knowledge={
                        "situation": str(k.get("situation", "")),
                        "situation_source": [s.timestamp for s in steps if s.timestamp][:1],
                        "tacit_insight": str(k.get("tacit_insight", "")),
                        "reasoning": str(k.get("reasoning", "")),
                        "reasoning_source": [],
                        "reasoning_origin": origin if origin in ("utterance", "model_inferred")
                                            else "model_inferred",
                        "diagnostic_steps": steps,
                        "conflict": bool(k.get("conflict", False)),
                        "conflict_detail": k.get("conflict_detail"),
                    }))

            doc = TacitKnowledgeDocument(schema_version=SCHEMA_VERSION,
                                         video_id=cid, candidates=candidates)
            out = contract.tacit(cid)
            out.write_text(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2),
                           encoding="utf-8")
            print(f"[S5] saved {out} ({len(candidates)} candidates)")
    finally:
        llm.unload()
