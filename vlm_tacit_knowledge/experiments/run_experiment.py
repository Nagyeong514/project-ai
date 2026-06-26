"""한 영상에 대해 A/B/C 전 과정 실행.

  전처리(VAD→STT) → 프레임 추출(A/B/C) → VLM 추론 → 출력 저장

사용:
  python -m experiments.run_experiment V01
  (config.yaml 의 vlm.backend='stub' 이면 모델 없이 흐름만 검증)
"""
from __future__ import annotations
import json
import os
import sys
import time

import yaml

# 패키지 루트 기준 실행
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import vad, stt, frames, vlm  # noqa: E402


def load_cfg() -> dict:
    with open(os.path.join(ROOT, "configs", "config.yaml")) as f:
        return yaml.safe_load(f)


def _abs(cfg, key, *parts):
    return os.path.join(ROOT, cfg["paths"][key], *parts)


def run(video_id: str):
    cfg = load_cfg()
    t0 = time.time()
    mp4 = _abs(cfg, "raw_dir", f"{video_id}.mp4")
    wav = _abs(cfg, "raw_dir", f"{video_id}.wav")
    assert os.path.exists(mp4), f"영상 없음: {mp4}"
    assert os.path.exists(wav), f"오디오 없음(먼저 추출 필요): {wav}"

    work = _abs(cfg, "work_dir", video_id)
    os.makedirs(work, exist_ok=True)
    os.makedirs(_abs(cfg, "out_dir"), exist_ok=True)

    # 1) 전처리 ----------------------------------------------------------
    dur = vad.duration(wav)
    segments = vad.detect_segments(wav, cfg)
    print(f"[VAD] {video_id}: {dur:.1f}s, 발화 구간 {len(segments)}개")
    seg_texts = stt.transcribe_segments(wav, segments, cfg)
    full_text = stt.transcribe_full(wav, cfg) if cfg["conditions"]["A"]["enabled"] else ""
    speech_total = sum(b - a for a, b in segments)
    print(f"[STT] 구간 전사 완료. 발화 {speech_total:.1f}s / 비발화 {dur - speech_total:.1f}s")

    # 2) 프레임 추출 -----------------------------------------------------
    jobs = []   # (condition, seg_idx|None, stt_text, frame_paths, window)
    if cfg["conditions"]["A"]["enabled"]:
        a = frames.extract_A(mp4, dur, work, cfg)
        jobs.append(("A", None, full_text, a["frames"], a["window"]))
    if cfg["conditions"]["B"]["enabled"]:
        for it in frames.extract_B(mp4, segments, seg_texts, work, cfg):
            jobs.append(("B", it["seg_idx"], it["stt"], it["frames"], it["window"]))
    if cfg["conditions"]["C"]["enabled"]:
        for it in frames.extract_C(mp4, dur, segments, seg_texts, work, cfg):
            txt = it["stt"]
            if it["stt_source"] == "window":  # 랜덤창 실제 전사
                txt = stt.transcribe_segments(wav, [tuple(it["window"])], cfg)[0]
            jobs.append(("C", it["seg_idx"], txt, it["frames"], it["window"]))
    print(f"[FRAMES] 추출 완료: {len(jobs)} 작업")

    # 3) VLM 추론 --------------------------------------------------------
    results = []
    for cond, sidx, txt, fpaths, window in jobs:
        out = vlm.infer(txt, fpaths, cfg)
        results.append({
            "video": video_id, "condition": cond, "seg_idx": sidx,
            "window": window, "n_frames": len(fpaths), "stt": txt,
            "output": out,
        })
    backend = cfg["vlm"]["backend"]
    print(f"[VLM/{backend}] 추론 완료: {len(results)} 출력")

    # 4) 저장 ------------------------------------------------------------
    meta = {
        "video": video_id, "duration_sec": dur,
        "n_segments": len(segments), "speech_sec": speech_total,
        "speech_ratio": speech_total / dur if dur else 0,
        "segments": segments, "backend": backend,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = _abs(cfg, "out_dir", f"{video_id}_{backend}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_path}  (elapsed {meta['elapsed_sec']}s)")
    return out_path


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "V01"
    run(vid)
