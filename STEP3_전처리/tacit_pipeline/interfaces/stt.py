"""STT + Transcript 정제 인터페이스."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from ..schema.intermediate import FrameMeta, Transcript, Utterance


@runtime_checkable
class STTBackend(Protocol):
    """음성 → 타임스탬프 달린 transcript. (예: Whisper large-v3-turbo)

    구현은 segment 단위 timestamp(초)를 반드시 채워야 한다.
    """

    def transcribe(self, video_path: str, video_id: str) -> Transcript:
        """오디오를 받아 Transcript(초 단위 utterances) 반환. raw_text 손실 금지."""
        ...


@runtime_checkable
class TranscriptRefiner(Protocol):
    """transcript 정제: 영어 정규화 + 근거성 발화 태깅.

    원칙: 비근거 발화도 버리지 않고 태깅만 한다(정렬·맥락·action_only 판단 위해).
    영어 정규화는 정렬 '전에' 먼저 수행한다(스펙 5.5).
    """

    def refine(self, transcript: Transcript) -> Transcript:
        """normalized_text 채우고 tags 부여한 Transcript 반환."""
        ...
