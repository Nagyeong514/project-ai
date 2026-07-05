"""S1: STT + transcript 정제. 모델 1회 로드 → 4클립 순회 → release.

무거운 import 는 함수 안에서만(지연 import 원칙).
"""
from __future__ import annotations

import json
import re
from typing import List

from core.config import Cfg
from core.gpu import log_mem, release
from core.paths import Contract, clip_id
from core.schema import Transcript, Utterance

# 실측으로 확정된 것만 활성(추측 활성화 금지 — 잘못된 치환이 원문 훼손)
NORMALIZATION = {
    "마더보이드": "motherboard",
    "마더보드": "motherboard",
    "메인보드": "motherboard",
}

HALLUCINATION_PHRASES = [
    "다음 영상에서 만나요", "다음 영상에서 뵙겠습니다", "시청해 주셔서 감사합니다",
    "감사합니다", "구독과 좋아요", "구독", "아멘",
]


def _norm_key(s: str) -> str:
    return re.sub(r"[\s\W]+", "", s).lower()


def _refine(tr: Transcript) -> Transcript:
    hall = {_norm_key(p) for p in HALLUCINATION_PHRASES}
    keys = sorted(NORMALIZATION, key=len, reverse=True)  # 긴 키 우선(부분매칭 충돌 방지)
    prev = None
    for u in tr.utterances:
        text = u.raw_text
        for k in keys:
            text = text.replace(k, NORMALIZATION[k])
        u.normalized_text = text
        nk = _norm_key(text)
        u.repeat_hallucination = (nk in hall) or (prev is not None and nk == prev)
        prev = nk
    return tr


def run(cfg: Cfg, force: bool = False):
    contract = Contract(cfg.paths.output_dir)
    contract.ensure_dirs()

    todo = []
    for v in cfg.paths.videos:
        cid = clip_id(v)
        if not force and contract.transcript(cid).exists():
            print(f"[S1] skip (exists): {cid}")
            continue
        todo.append((v, cid))
    if not todo:
        return

    from faster_whisper import WhisperModel  # 지연 import

    print(f"[S1] loading whisper {cfg.stt.model} ({cfg.stt.compute_type})")
    model = WhisperModel(cfg.stt.model, device=cfg.stt.device, compute_type=cfg.stt.compute_type)
    log_mem("S1 loaded")

    for video, cid in todo:
        print(f"[S1] transcribe: {cid}")
        segments, _info = model.transcribe(
            video,
            language=cfg.stt.language,
            vad_filter=cfg.stt.vad_filter,
            vad_parameters={"threshold": cfg.stt.vad_threshold},
            condition_on_previous_text=False,   # 환각/반복 전파 차단
            word_timestamps=False,
        )
        utts: List[Utterance] = [
            Utterance(start=float(s.start), end=float(s.end), raw_text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]
        tr = _refine(Transcript(video_id=cid, model=cfg.stt.model, utterances=utts))
        out = contract.transcript(cid)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tr.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[S1] saved {out} ({len(utts)} utterances)")

    release(model)
    log_mem("S1 released")
