"""CER / WER 계산."""
from jiwer import cer, wer

from evaluation.normalizer import normalize


def compute_cer(hypothesis: str, reference: str) -> float:
    """주지표. 정규화 적용 후 글자 오류율."""
    return cer(normalize(reference), normalize(hypothesis))


def compute_wer(hypothesis: str, reference: str) -> float:
    """보조지표. 정규화 적용 후 단어 오류율."""
    return wer(normalize(reference), normalize(hypothesis))


def evaluate(hypothesis: str, reference: str) -> dict:
    return {
        "cer": compute_cer(hypothesis, reference),
        "wer": compute_wer(hypothesis, reference),
        "hypothesis": normalize(hypothesis),
        "reference": normalize(reference),
    }
