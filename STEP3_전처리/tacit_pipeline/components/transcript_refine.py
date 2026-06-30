"""
Transcript 정제 (스펙 5.5 / 설계결정 b, 2026-06-30).

정제 stage는 **결정적(determimistic) 작업만** 한다:
  1) 영어/기술용어 정규화: 한글 STT 오인식(램→RAM, 마더보이드→motherboard 등) 사전 치환.
     - 사전은 외부 파일(resources/en_normalization.json). **정렬 전에** 수행.
  2) 연속 반복 발화 태깅: Whisper 끝부분 동일문장 반복(환각 의심)을 REPETITION으로 표시.
     - 삭제하지 않는다(타임스탬프/맥락 보존 원칙). 융합 LLM이 무시 힌트로 쓴다.

**근거성(인과/주의/매뉴얼차이/추론) 판단은 여기서 안 한다 → 융합 LLM이 통째로 한다.**
이유(실측): 한국어 구어체("~하면 됩니다", "~나면 ~거에요", "순서가 있어요")를 정규식이
0건 잡음 — 정규식 근거판단은 폐기. 원문(raw_text)은 끝까지 보존된다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..schema.intermediate import Transcript, Utterance, UtteranceTag


def _norm_key(text: str) -> str:
    """반복 비교용 정규화: 공백 축약 + 양끝 구두점/공백 제거."""
    return re.sub(r"\s+", " ", text).strip().strip(".,!?…").strip()


class NormalizeRefiner:
    """정규화 + 반복감지 정제기. registry 키: 'normalize'.

    (구 RegexTranscriptRefiner의 근거 태깅은 폐기 — 융합 LLM이 판단.)
    """

    def __init__(
        self,
        normalization_dict_path: Optional[str] = None,
        flag_repetitions: bool = True,
        **extra,
    ):
        self.flag_repetitions = flag_repetitions
        self._norm = self._load_norm(normalization_dict_path)

    def _load_norm(self, path: Optional[str]) -> Dict[str, str]:
        if path is None:
            path = str(Path(__file__).parent.parent / "resources" / "en_normalization.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}

    def _normalize(self, text: str) -> str:
        """오인식 용어 치환. 긴 키부터(부분매칭 충돌 방지), 대소문자 무시."""
        out = text
        for key in sorted(self._norm, key=len, reverse=True):
            out = re.sub(re.escape(key), self._norm[key], out, flags=re.IGNORECASE)
        return out

    def refine(self, transcript: Transcript) -> Transcript:
        refined: List[Utterance] = []
        prev_key: Optional[str] = None
        for u in transcript.utterances:
            norm = self._normalize(u.raw_text)  # 정렬 전 정규화
            tags = [UtteranceTag.NONE]
            if self.flag_repetitions:
                key = _norm_key(norm)
                # 직전 발화와 동일하면 반복(Whisper 끝부분 환각 의심)으로 태깅
                if key and key == prev_key:
                    tags = [UtteranceTag.REPETITION]
                prev_key = key
            refined.append(
                Utterance(
                    start=u.start,
                    end=u.end,
                    raw_text=u.raw_text,  # 원문 보존(절대 손실 금지)
                    normalized_text=norm,
                    tags=tags,
                )
            )
        return Transcript(
            video_id=transcript.video_id,
            language=transcript.language,
            model=transcript.model,
            utterances=refined,
        )
