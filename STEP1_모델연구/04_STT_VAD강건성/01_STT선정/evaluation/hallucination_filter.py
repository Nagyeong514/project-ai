"""
STT 환각 필터 — 채점(evaluate) 전에 하이포시스 텍스트에서 알려진 환각 상투구·반복 루프를 제거한다.

기반: STEP3_전처리/tacit_pipeline/components/transcript_refine.py의 DEFAULT_HALLUCINATION_PHRASES.
그 모듈은 발화(utterance) 단위로 태깅만 하고 원문을 보존하는 설계(융합 LLM이 무시 힌트로 씀)인데,
여기 01_STT선정은 CER/WER을 재는 게 목적이라 실제로 텍스트에서 제거해야 지표가 바뀐다 — 그래서
태깅이 아니라 제거(remove) 방식으로 재구현했다.

한계(정직히 명시): denylist는 "알려진 상투구"만 걷어낸다. 반복 루프 축약은 일반적인 패턴(같은
어절/구가 연속 반복)만 잡는다. 1회성 환각(예: 관련 없는 레시피·지역기관명 삽입처럼 반복도 없고
denylist에도 없는 문장)은 이 필터로 못 잡는다 — 그런 경우 CER 개선은 기대할 수 없다.
"""
import re
from typing import List, Optional

# STEP3_전처리 DEFAULT_HALLUCINATION_PHRASES 그대로 재사용.
STEP3_HALLUCINATION_PHRASES = [
    "다음 영상에서 만나요",
    "다음 영상에서 뵙겠습니다",
    "시청해 주셔서 감사합니다",
    "감사합니다",
    "구독과 좋아요",
    "구독",
    "아멘",
]

# 01_STT선정 v2 실측(2026-07-01, CLIP1~4)에서 추가 관측된 상투구.
# 정규식이라 부분 변형(기자 유무, 앵커 이름)도 함께 잡는다.
V2_HALLUCINATION_PATTERNS = [
    r"-?\(?\s*기자\s*\)?\s*",  # "-(기자)" 접두어 (뉴스 패턴의 일부, 아래와 함께 제거)
    r"MBC\s*뉴스\s*[가-힣]{2,4}\s*입니다\.?",
    r"한글자막\s*by\s*[가-힣]+",
]


def _remove_denylist(text: str, extra_phrases: Optional[List[str]] = None) -> str:
    """리터럴 상투구(대소문자 무시) + v2 정규식 패턴을 제거한다."""
    out = text
    phrases = STEP3_HALLUCINATION_PHRASES + (extra_phrases or [])
    for phrase in phrases:
        out = re.sub(re.escape(phrase), " ", out, flags=re.IGNORECASE)
    for pattern in V2_HALLUCINATION_PATTERNS:
        out = re.sub(pattern, " ", out)
    return out


def _collapse_repeated_phrases(text: str, max_window: int = 10, min_repeats: int = 2) -> str:
    """
    공백 기준 어절 중, 길이 1~max_window 어절의 구가 min_repeats회 이상 연속 반복되면
    한 번으로 축약한다. 짧은 주기(최소 반복 단위)부터 찾아 완전히 축약한다.
    ("감사합니다"×26, "참기름"×7, "러시아내로 완료!!"×4, 문장 통째 반복×2 모두 이 방식으로 잡힘.)
    """
    tokens = text.split()
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        collapsed = False
        for w in range(1, max_window + 1):
            if i + w > n:
                break
            phrase = tokens[i : i + w]
            reps = 1
            j = i + w
            while j + w <= n and tokens[j : j + w] == phrase:
                reps += 1
                j += w
            if reps >= min_repeats:
                out.extend(phrase)
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def filter_hallucinations(text: str, extra_denylist: Optional[List[str]] = None) -> str:
    """채점 전 하이포시스 텍스트 정제: denylist 제거 -> 반복구 축약 -> 공백 정리."""
    out = _remove_denylist(text, extra_denylist)
    out = _collapse_repeated_phrases(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
