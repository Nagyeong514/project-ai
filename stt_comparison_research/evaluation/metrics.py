"""
STT 평가 지표 계산.

입력 가정:
    reference, hypothesis 는 normalizer.normalize() 로 정규화된 텍스트.
    이중 정규화 방지를 위해 여기서는 추가 정규화를 하지 않는다.

의존성:
    pip install jiwer
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import jiwer

from evaluation.normalizer import normalize


@dataclass
class Metrics:
    # 정확도
    cer: float           # Character Error Rate (주지표)
    wer: float           # Word Error Rate (보조)

    # 오류 분해 (단어 기준)
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    # 진단용 파생 지표
    ref_words: int
    hyp_words: int
    length_ratio: float  # hyp/ref (>1 환각 경향, <1 누락 경향)
    ins_rate: float      # insertions / ref_words (환각 지표)
    del_rate: float      # deletions / ref_words  (누락 지표)

    # 롱폼 안정성
    cer_early: Optional[float] = None   # 전반부 CER
    cer_late: Optional[float] = None    # 후반부 CER
    cer_degradation: Optional[float] = None  # late - early (양수 = 후반 악화)

    # 한영 혼용
    cs_wer: Optional[float] = None      # 영어 토큰만 계산한 WER (없으면 None)
    cs_ref_tokens: int = 0              # GT 내 영어 토큰 수

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("cer", "wer", "length_ratio", "ins_rate", "del_rate"):
            d[k] = round(d[k], 6)
        for k in ("cer_early", "cer_late", "cer_degradation", "cs_wer"):
            if d[k] is not None:
                d[k] = round(d[k], 6)
        return d


def compute_metrics(reference: str, hypothesis: str) -> Metrics:
    """정규화된 reference/hypothesis 쌍으로 전체 지표 계산."""
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()

    if not ref:
        n_hyp = len(hyp.split())
        return Metrics(
            cer=0.0, wer=0.0,
            substitutions=0, deletions=0, insertions=n_hyp, hits=0,
            ref_words=0, hyp_words=n_hyp,
            length_ratio=0.0, ins_rate=0.0, del_rate=0.0,
        )

    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)

    ref_words = w.hits + w.substitutions + w.deletions
    hyp_words = w.hits + w.substitutions + w.insertions

    return Metrics(
        cer=c.cer,
        wer=w.wer,
        substitutions=w.substitutions,
        deletions=w.deletions,
        insertions=w.insertions,
        hits=w.hits,
        ref_words=ref_words,
        hyp_words=hyp_words,
        length_ratio=(hyp_words / ref_words) if ref_words else 0.0,
        ins_rate=(w.insertions / ref_words) if ref_words else 0.0,
        del_rate=(w.deletions / ref_words) if ref_words else 0.0,
    )


# 영어 토큰 추출 (2글자 이상, 단순 숫자 제외)
_EN_PATTERN = re.compile(r"[a-zA-Z]{2,}")


def _split_gt_by_proportion(reference: str, ratio: float) -> tuple[str, str]:
    """GT를 ratio 비율로 앞/뒤 분할 (단어 단위)."""
    words = reference.split()
    split_idx = max(1, int(len(words) * ratio))
    return " ".join(words[:split_idx]), " ".join(words[split_idx:])


def add_longform_stability(
    metrics: Metrics,
    segments: list,          # STTResult.segments (start, end, text)
    reference: str,          # 정규화된 GT
) -> None:
    """롱폼 안정성 지표를 metrics 객체에 인플레이스로 추가."""
    if not segments:
        return

    total_duration = segments[-1].end - segments[0].start
    if total_duration <= 0:
        return

    midpoint = segments[0].start + total_duration / 2
    early_segs = [s for s in segments if s.end <= midpoint]
    late_segs  = [s for s in segments if s.start >= midpoint]

    if not early_segs or not late_segs:
        return

    early_hyp = " ".join(s.text.strip() for s in early_segs)
    late_hyp  = " ".join(s.text.strip() for s in late_segs)

    # 가설 문자 비율로 GT 분할
    total_chars = len(early_hyp) + len(late_hyp)
    if total_chars == 0:
        return
    ratio = len(early_hyp) / total_chars
    early_ref, late_ref = _split_gt_by_proportion(reference, ratio)

    if not early_ref or not late_ref:
        return

    cer_e = jiwer.process_characters(early_ref, early_hyp).cer
    cer_l = jiwer.process_characters(late_ref, late_hyp).cer

    metrics.cer_early = cer_e
    metrics.cer_late = cer_l
    metrics.cer_degradation = cer_l - cer_e


def add_codeswitching_wer(
    metrics: Metrics,
    hypothesis: str,   # 정규화된 가설
    reference: str,    # 정규화된 GT
) -> None:
    """한영 혼용 WER을 metrics 객체에 인플레이스로 추가."""
    ref_en_tokens = _EN_PATTERN.findall(reference)
    hyp_en_tokens = _EN_PATTERN.findall(hypothesis)

    metrics.cs_ref_tokens = len(ref_en_tokens)

    if len(ref_en_tokens) < 3:
        # 영어 토큰이 3개 미만이면 측정 의미 없음
        return

    ref_str = " ".join(ref_en_tokens)
    hyp_str = " ".join(hyp_en_tokens) if hyp_en_tokens else ""
    metrics.cs_wer = jiwer.wer(ref_str, hyp_str) if hyp_str else 1.0


def evaluate(hypothesis: str, reference: str, segments: list = None) -> Metrics:
    """
    정규화 포함 원스톱 평가.
    segments: STTResult.segments (롱폼 안정성 계산용, 없으면 생략)
    """
    norm_ref = normalize(reference)
    norm_hyp = normalize(hypothesis)

    m = compute_metrics(norm_ref, norm_hyp)

    if segments:
        add_longform_stability(m, segments, norm_ref)

    add_codeswitching_wer(m, norm_hyp, norm_ref)

    return m
