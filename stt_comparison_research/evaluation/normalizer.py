"""
CER/WER 측정 전 정규화 규칙 — 전 조건 동일 적용 (공정 비교 보장).
규칙은 experiment_config.yaml evaluation.normalization 에 명시된 내용과 일치.
"""
import re
from typing import List


# 제거할 간투어 목록 (config와 동기화)
_FILLER_WORDS: List[str] = ["음", "아", "어", "그", "뭐", "이제", "그냥"]

# 아라비아 숫자 → 변환 없이 유지 (숫자 표기 정책: arabic 기준)
# "삼"처럼 한글로 쓴 경우는 GT도 동일 정책으로 작성해야 함

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize(text: str) -> str:
    """
    정규화 순서:
    1. 간투어 제거
    2. 소문자 변환 (영문 혼재 대비)
    3. 문장부호 제거
    4. 연속 공백 → 단일 공백
    5. strip
    """
    for fw in _FILLER_WORDS:
        # 단어 경계 기반 제거 (예: "음 안녕" → "안녕")
        text = re.sub(rf"(?<!\w){re.escape(fw)}(?!\w)", " ", text)

    text = text.lower()
    text = _PUNCTUATION_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
