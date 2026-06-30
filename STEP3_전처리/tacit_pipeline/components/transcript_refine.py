"""
Transcript 정제 (스펙 5.5).

1) 영어 단어 정규화: 한글 STT 오인식 기술용어(RAM/POST/BIOS/RDIMM/ECC...) 표준화.
   - 정규화 사전은 외부 파일(resources/en_normalization.json).
   - **정렬 전에** 먼저 수행(이 모듈이 책임).
2) 근거성 발화 추출 — 유형 태깅(인과/매뉴얼차이/주의/부정).
   - 1차 정규식, 보강으로 LLM 분류기. 둘 다 교체 가능.
3) **비근거 발화도 버리지 않고 태깅만** 한다(원문 raw_text 보존):
   - 타임스탬프 정렬 보존 / 융합 시 맥락 / 'action_only' 판단을 위해.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..schema.intermediate import Transcript, Utterance, UtteranceTag

# 근거성 발화 패턴(스펙 5.5). 정규식 1차 필터. 필요시 외부화 가능.
_PATTERNS: Dict[UtteranceTag, List[str]] = {
    UtteranceTag.CAUSAL: [r"때문", r"하면\s*되", r"하면\s*돼", r"안\s*그러면", r"그래서", r"덕분"],
    UtteranceTag.MANUAL_DIFF: [r"원래는", r"보통은", r"사실", r"실은", r"원칙적으로"],
    UtteranceTag.CAUTION: [r"꼭", r"반드시", r"절대", r"조심", r"주의", r"신경"],
    UtteranceTag.NEGATION: [r"하면\s*안\s*[돼되]", r"하지\s*마", r"안\s*[돼되]요", r"금지"],
}


class RegexTranscriptRefiner:
    """정규식 기반 정제기. registry 키: 'regex_refiner'.

    use_llm_classifier=True 면 보강용 LLM 분류기를 추가로 호출(옵션, 교체 가능).
    """

    def __init__(
        self,
        normalization_dict_path: Optional[str] = None,
        use_llm_classifier: bool = False,
        llm_classifier: object | None = None,  # TODO(decision): LLM 분류기 인터페이스 확정
        **extra,
    ):
        self.use_llm_classifier = use_llm_classifier
        self.llm_classifier = llm_classifier
        self._norm = self._load_norm(normalization_dict_path)
        self._compiled = {
            tag: [re.compile(p) for p in pats] for tag, pats in _PATTERNS.items()
        }

    def _load_norm(self, path: Optional[str]) -> Dict[str, str]:
        if path is None:
            path = str(Path(__file__).parent.parent / "resources" / "en_normalization.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 메타 키(_로 시작) 제외
        return {k: v for k, v in raw.items() if not k.startswith("_")}

    def _normalize_english(self, text: str) -> str:
        """오인식 용어 치환. 긴 키부터(부분매칭 충돌 방지), 대소문자 무시."""
        out = text
        for key in sorted(self._norm, key=len, reverse=True):
            out = re.sub(re.escape(key), self._norm[key], out, flags=re.IGNORECASE)
        return out

    def _tag(self, text: str) -> List[UtteranceTag]:
        tags = [tag for tag, regs in self._compiled.items() if any(r.search(text) for r in regs)]
        if self.use_llm_classifier and self.llm_classifier is not None:
            # TODO(impl): LLM 분류기로 보강. 인터페이스 확정 후 연결.
            #   extra = self.llm_classifier.classify(text)  # -> List[UtteranceTag]
            pass
        return tags or [UtteranceTag.NONE]  # 비근거도 NONE으로 태깅(버리지 않음)

    def refine(self, transcript: Transcript) -> Transcript:
        refined: List[Utterance] = []
        for u in transcript.utterances:
            norm = self._normalize_english(u.raw_text)  # 정렬 전 정규화
            refined.append(
                Utterance(
                    start=u.start,
                    end=u.end,
                    raw_text=u.raw_text,  # 원문 보존(절대 손실 금지)
                    normalized_text=norm,
                    tags=self._tag(norm),
                )
            )
        return Transcript(
            video_id=transcript.video_id,
            language=transcript.language,
            model=transcript.model,
            utterances=refined,
        )
