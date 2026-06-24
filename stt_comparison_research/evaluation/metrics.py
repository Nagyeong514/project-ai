"""
STT 평가 지표 계산.

입력 가정:
    reference, hypothesis 는 normalizer.normalize() 로 정규화된 텍스트.
    이중 정규화 방지를 위해 여기서는 추가 정규화를 하지 않는다.

의존성:
    pip install jiwer
"""

from dataclasses import dataclass, asdict

import jiwer

from evaluation.normalizer import normalize


@dataclass
class Metrics:
    # 정확도
    cer: float          # Character Error Rate (주지표)
    wer: float          # Word Error Rate (보조지표)

    # 오류 분해 (단어 기준) — 어떻게 틀리나
    substitutions: int  # 치환 (잘못 알아들음)
    deletions: int      # 삭제 (빠뜨림)
    insertions: int     # 삽입 (없는 단어 생성 → 환각)
    hits: int           # 정답

    # 진단용 파생 지표
    ref_words: int
    hyp_words: int
    length_ratio: float  # hyp/ref (>1 환각 경향, <1 누락 경향)
    ins_rate: float      # insertions / ref_words (환각 지표)
    del_rate: float      # deletions / ref_words  (누락 지표)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("cer", "wer", "length_ratio", "ins_rate", "del_rate"):
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


def evaluate(hypothesis: str, reference: str) -> Metrics:
    """정규화 포함 원스톱 평가. run_all_models.py 에서 호출."""
    return compute_metrics(normalize(reference), normalize(hypothesis))
