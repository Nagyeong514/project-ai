"""한 영상에 대해 모든 조건 실행.

  전처리(구간+STT) → 프레임 추출 → VLM 추론 → 출력 저장

조건(config.conditions):
  A     : 전체영상 천장. a_total_frames 장을 max_frames_per_call 씩 청크로 나눠
          추론 후 knowledge_points 병합 (진짜 upper bound).
  B     : 발화구간 프레임 + STT (가설)
  B_vid : 발화구간 프레임만 (영상 단독 기여)        ← ablation
  B_txt : STT만 (발화 단독 기여)                    ← ablation
  C     : 발화구간 동일길이 랜덤창 (보류 시 enabled=false)

사용:  python -m experiments.run_experiment V03
"""
from __future__ import annotations
import json
import os
import random
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import vad, stt, frames, vlm  # noqa: E402


def load_cfg():
    with open(os.path.join(ROOT, "configs", "config.yaml")) as f:
        return yaml.safe_load(f)


def _abs(cfg, key, *p):
    return os.path.join(ROOT, cfg["paths"][key], *p)


def _merge_points(outputs: list[dict]) -> dict:
    """여러 청크 출력의 knowledge_points 를 하나로 병합 (A 천장용)."""
    pts, flags = [], {}
    for o in outputs:
        pts += o.get("knowledge_points", [])
        for k in ("_parse_error", "_recovered"):
            flags[k] = flags.get(k, False) or o.get(k, False)
    return {"knowledge_points": pts, "n_chunks": len(outputs),
            **{k: v for k, v in flags.items() if v}}


def run(video_id: str):
    cfg = load_cfg()
    t0 = time.time()
    mp4 = _abs(cfg, "raw_dir", f"{video_id}.mp4")
    wav = _abs(cfg, "raw_dir", f"{video_id}.wav")
    assert os.path.exists(mp4) and os.path.exists(wav), "영상/오디오 필요"
    work = _abs(cfg, "work_dir", video_id)
    os.makedirs(_abs(cfg, "out_dir"), exist_ok=True)
    conds = cfg["conditions"]
    cap = cfg["frames"]["max_frames_per_call"]

    # 1) 구간 + STT --------------------------------------------------
    dur = vad.duration(wav)
    if cfg.get("segments", {}).get("source") == "answerkey":
        gt = json.load(open(_abs(cfg, "gt_dir", f"{video_id}_answerkey.json"), encoding="utf-8"))
        segments = [tuple(s["window"]) for s in gt["segments"]]
        print(f"[SEG] 정답지 수동 구간 {len(segments)}개")
    else:
        segments = vad.detect_segments(wav, cfg)
        print(f"[SEG] VAD 구간 {len(segments)}개")
    seg_texts = stt.assign_to_segments(wav, segments, cfg)
    full_text = stt.transcribe_full(wav, cfg)
    print(f"[STT] 전사 완료 ({dur:.0f}s)")

    # 2) 프레임 추출 -------------------------------------------------
    # B 계열이 공유할 구간 프레임 1회 추출
    seg_frames = []
    if conds["B"]["enabled"] or conds["B_vid"]["enabled"] or conds["C"]["enabled"]:
        for i, (a, b) in enumerate(segments):
            fr = frames.uniform_pick(frames.extract(mp4, a, b, os.path.join(work, "B", f"seg{i:02d}"), cfg), cap)
            seg_frames.append(fr)
    a_chunks = []
    if conds["A"]["enabled"]:
        allf = frames.uniform_pick(frames.extract(mp4, 0.0, dur, os.path.join(work, "A"), cfg),
                                   cfg.get("a_total_frames", cap))
        a_chunks = frames.chunk(allf, cap)
    print("[FRAMES] 추출 완료")

    # 3) 작업 구성 + 추론 -------------------------------------------
    results = []

    def infer_cond(name, seg_idx, text, fpaths, window):
        c = conds[name]
        out = vlm.infer(text if c["use_stt"] else None,
                        fpaths if c["use_frames"] else [], cfg)
        results.append({"video": video_id, "condition": name, "seg_idx": seg_idx,
                        "window": window, "n_frames": len(fpaths) if c["use_frames"] else 0,
                        "use_stt": c["use_stt"], "use_frames": c["use_frames"],
                        "stt": text if c["use_stt"] else None, "output": out})

    # A: 청크별 추론 후 병합
    if conds["A"]["enabled"]:
        outs = [vlm.infer(full_text, ch, cfg) for ch in a_chunks]
        merged = _merge_points(outs)
        results.append({"video": video_id, "condition": "A", "seg_idx": None,
                        "window": [0.0, dur], "n_frames": sum(len(c) for c in a_chunks),
                        "use_stt": True, "use_frames": True, "stt": full_text, "output": merged})

    # B / B_vid / B_txt: 구간별
    for i, (win, text) in enumerate(zip(segments, seg_texts)):
        fr = seg_frames[i] if seg_frames else []
        for name in ("B", "B_vid", "B_txt"):
            if conds[name]["enabled"]:
                infer_cond(name, i, text, fr, list(win))

    # C: 구간별 랜덤창 (보류 시 skip)
    if conds["C"]["enabled"]:
        rng = random.Random(cfg["control"]["c_random_seed"])
        for i, (a, b) in enumerate(segments):
            length = b - a
            ra = 0.0 if length >= dur else rng.uniform(0.0, dur - length)
            fr = frames.uniform_pick(
                frames.extract(mp4, ra, ra + length, os.path.join(work, "C", f"seg{i:02d}"), cfg), cap)
            infer_cond("C", i, seg_texts[i], fr, [ra, ra + length])

    backend = cfg["vlm"]["backend"]
    print(f"[VLM/{backend}] 추론 완료: {len(results)} 출력")

    # 4) 저장 -------------------------------------------------------
    meta = {"video": video_id, "duration_sec": dur, "n_segments": len(segments),
            "backend": backend, "elapsed_sec": round(time.time() - t0, 1)}
    out_path = _abs(cfg, "out_dir", f"{video_id}_{backend}.json")
    json.dump({"meta": meta, "results": results}, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_path}  ({meta['elapsed_sec']}s)")
    return out_path


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "V03")
