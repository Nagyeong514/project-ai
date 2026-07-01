"""
최종 출력 JSON 스키마 — **단일 진실 공급원(single source of truth)**.

파이프라인 어디서도 JSON 키 문자열을 하드코딩하지 마라. 키가 바뀌면 이 파일 한 곳만 고친다.
스키마는 아직 '최종 아님'(`SCHEMA_VERSION`)이고 키 이름이 바뀔 수 있다.

작업 지시서 4번의 목표 JSON을 그대로 Pydantic으로 옮긴 것이다.
주석의 [LLM]/[VLM]/[STT]/[자동] 은 각 필드를 누가 채우는지(출처)를 뜻한다.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# 스키마 버전. 키/구조가 바뀌면 여기서 올린다. 최종 출력 JSON에도 박혀 나간다.
SCHEMA_VERSION = "1.3"


# ──────────────────────────────────────────────────────────────────────────
# Enum — 자유 문자열 대신 닫힌 집합으로 둬서 오타/혼선을 막는다.
# ──────────────────────────────────────────────────────────────────────────
class EvidenceType(str, Enum):
    """한 항목의 근거가 무엇이냐.

    - UTTERANCE: 발화(말)로 뒷받침됨.
    - ACTION_ONLY: 말은 안 했지만 영상 행동으로 관측됨(말 안 한 중요한 행동).
    """

    UTTERANCE = "utterance"
    ACTION_ONLY = "action_only"


class ReasoningOrigin(str, Enum):
    """reasoning(원인 설명)이 어디서 왔냐 — 할루시네이션 규율의 핵심.

    - UTTERANCE: 숙련자 발화에서 직접 도출.
    - MODEL_INFERRED: LLM 일반지식으로 채움. **발화 근거인 척 위장 금지.**
    """

    UTTERANCE = "utterance"
    MODEL_INFERRED = "model_inferred"


# ──────────────────────────────────────────────────────────────────────────
# 하위 모델
# ──────────────────────────────────────────────────────────────────────────
class Source(BaseModel):
    """원본 추적 정보. clip_start/clip_end 는 'HH:MM:SS' 문자열(스펙 그대로)."""

    # 아래 3개는 우리가 이미 아는 결정적 값 → [자동] 시스템(orchestrator._finalize_metadata)이 채운다.
    # LLM이 추측해선 안 된다(예전엔 LLM이 "..." 리터럴을 뱉었음). 기본값을 둬서 LLM이 생략해도 통과.
    video_id: str = ""  # [자동]
    clip_start: Optional[str] = None  # [자동] "00:00:00"
    clip_end: Optional[str] = None  # [자동] 영상 길이 기반 "HH:MM:SS"
    transcript_ref: Optional[str] = None  # [자동] "transcripts/<video_id>.json"


class Metadata(BaseModel):
    scenario_id: str = ""  # [자동] 원본 파일명 기반 — 시스템이 채운다
    equipment: Optional[str] = None  # [LLM] 영상/메타에서 식별
    task: Optional[str] = None  # [LLM] 작업유형 자동분류
    keywords: List[str] = Field(default_factory=list)  # [LLM]
    scenario_title: Optional[str] = None  # [LLM] 자동 생성
    source: Source = Field(default_factory=Source)  # [자동] 시스템이 채운다


class DiagnosticStep(BaseModel):
    """진단 절차 한 스텝. evidence 로 발화근거/행동단독을 구분한다."""

    order: int
    action: str  # [VLM 행동] 무엇을 했는가
    evidence: EvidenceType
    # 발화 원문 그대로 보존(정제 단계에서 버리지 않는다). action_only면 None.
    source_utterance: Optional[str] = None
    timestamp: Optional[str] = None  # "HH:MM:SS" — action_only면 None일 수 있음

    # NOTE: source_utterance/ timestamp 가 None인데 evidence=UTTERANCE면 모순.
    # 교차검증은 validators에서(아래) 수행.


class Knowledge(BaseModel):
    # 본 것(VLM) vs 들은 것(STT) 충돌 보존(스펙 4·델타 #6). 어느 쪽이 맞는지 판단하지 않고
    # 둘 다 남긴 뒤 플래그만 — 진위 판별은 다음 단계(품질검증) 몫.
    conflict: bool = False
    conflict_detail: Optional[str] = None  # 예: "LED 횟수 불일치 — VLM:황1+백3 / STT:백5+황4"
    situation: str
    situation_source: List[str] = Field(default_factory=list)  # 근거 발화 timestamp들
    tacit_insight: str
    reasoning: Optional[str] = None
    reasoning_source: List[str] = Field(default_factory=list)
    reasoning_origin: ReasoningOrigin = ReasoningOrigin.MODEL_INFERRED
    diagnostic_steps: List[DiagnosticStep] = Field(default_factory=list)
    # TODO(decision): normal_procedure(매뉴얼 정상절차, 델타 판정 기준선) — 후속 단계용. 지금은 미포함.
    # normal_procedure: Optional[str] = None


class TacitKnowledgeCandidate(BaseModel):
    """LLM 융합 단계가 최종적으로 채우는 '암묵지 후보' 1건.

    이 파이프라인은 '후보 생성'까지만 책임진다. 진짜 암묵지인지 판별/검증은 다음 단계.
    """

    id: str = ""  # [자동] video_id + task + 일련번호  e.g. tk_dell7920_mem_boot_001
    schema_version: str = SCHEMA_VERSION
    metadata: Metadata = Field(default_factory=Metadata)  # [자동] LLM은 knowledge에만 집중
    knowledge: Knowledge

    # ── 교차 검증 ──────────────────────────────────────────────────────
    @staticmethod
    def _norm(s: Optional[str]) -> str:
        """대조용 정규화 — 공백/문장부호 제거(한글·영숫자만 남김). 한글은 \\w라 보존됨."""
        return re.sub(r"[\s\W]+", "", s).lower() if s else ""

    @staticmethod
    def _match(a: str, b: str) -> bool:
        """정규화된 두 문자열이 같거나 한쪽이 다른 쪽을 포함(짧은 문자열 오탐 방지 8자 하한)."""
        if not a or not b:
            return False
        if a == b:
            return True
        return len(a) >= 8 and (a in b or b in a)

    def cross_check(
        self,
        observation_texts: Optional[List[str]] = None,
        utterance_texts: Optional[List[str]] = None,
    ) -> List[str]:
        """할루시네이션 규율 위반 의심 지점을 문자열 리스트로 돌려준다(에러는 아님).

        오케스트레이터가 로그/재시도 판단에 쓴다. 강제 실패가 아니라 '경고'로 둔 이유:
        후보는 빠짐없이 살리는 게 우선이고, 판별은 다음 팀 몫이라서.

        observation_texts: 이 영상의 VLM 관찰문들(있으면 '관찰을 발화로 둔갑' 탐지).
        utterance_texts: 이 영상의 STT 발화문들(있으면 '입력에 없는 발화 지어냄' 탐지).
        """
        warnings: List[str] = []
        obs_norm = [n for n in (self._norm(t) for t in (observation_texts or [])) if n]
        utt_norm = [n for n in (self._norm(t) for t in (utterance_texts or [])) if n]

        for step in self.knowledge.diagnostic_steps:
            if step.evidence == EvidenceType.UTTERANCE and not step.source_utterance:
                warnings.append(
                    f"step {step.order}: evidence=utterance인데 source_utterance가 비었음(위장 의심)"
                )
            if step.evidence == EvidenceType.ACTION_ONLY and step.source_utterance:
                warnings.append(
                    f"step {step.order}: evidence=action_only인데 source_utterance가 있음(모순)"
                )
            # 신규: source_utterance가 VLM 관찰문/입력 발화와 맞는지 대조(둔갑·지어냄 탐지)
            if step.evidence == EvidenceType.UTTERANCE and step.source_utterance:
                su = self._norm(step.source_utterance)
                if su:
                    if any(self._match(su, o) for o in obs_norm):
                        warnings.append(
                            f"step {step.order}: source_utterance가 VLM 관찰문과 일치 "
                            "— 관찰을 발화로 둔갑시킨 것으로 의심"
                        )
                    elif utt_norm and not any(self._match(su, u) for u in utt_norm):
                        warnings.append(
                            f"step {step.order}: source_utterance가 입력 발화에 없음 "
                            "— 발화를 지어낸 것으로 의심"
                        )
        if (
            self.knowledge.reasoning_origin == ReasoningOrigin.UTTERANCE
            and not self.knowledge.reasoning_source
        ):
            warnings.append("reasoning_origin=utterance인데 reasoning_source 근거 timestamp가 없음")
        if self.knowledge.conflict and not self.knowledge.conflict_detail:
            warnings.append("conflict=true인데 conflict_detail이 비었음(충돌 내용 누락)")
        return warnings


class TacitKnowledgeDocument(BaseModel):
    """한 영상에서 나온 후보들의 묶음(파일 단위 출력).

    한 영상에서 여러 암묵지 후보가 나올 수 있으므로 리스트로 감싼다.
    """

    schema_version: str = SCHEMA_VERSION
    video_id: str
    candidates: List[TacitKnowledgeCandidate] = Field(default_factory=list)
