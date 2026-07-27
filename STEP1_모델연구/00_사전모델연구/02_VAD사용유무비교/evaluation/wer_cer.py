"""WER / CER 계산. 정규화 규칙을 사전 고정해 공정 비교 보장."""
import re
from typing import List

from jiwer import wer, cer


def normalize_korean(text: str) -> str:
    """
    정규화 규칙 (전 조건 동일 적용):
    - 소문자 변환
    - 문장부호 제거
    - 숫자 → 한국어 표기 (별도 처리 필요 시 확장)
    - 연속 공백 단일화
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(hypothesis: str, reference: str, normalize: bool = True) -> float:
    if normalize:
        hypothesis = normalize_korean(hypothesis)
        reference = normalize_korean(reference)
    return wer(reference, hypothesis)


def compute_cer(hypothesis: str, reference: str, normalize: bool = True) -> float:
    if normalize:
        hypothesis = normalize_korean(hypothesis)
        reference = normalize_korean(reference)
    return cer(reference, hypothesis)


def segments_to_text(segments) -> str:
    return " ".join(seg.text.strip() for seg in segments)


def evaluate_accuracy(
    pred_segments,
    reference_text: str,
) -> dict:
    pred_text = segments_to_text(pred_segments)
    return {
        "wer": compute_wer(pred_text, reference_text),
        "cer": compute_cer(pred_text, reference_text),
        "pred_text": pred_text,
    }
