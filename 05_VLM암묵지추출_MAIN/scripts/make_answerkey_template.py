"""정답지 템플릿 생성.

VAD로 구간을 잡고, 각 구간의 STT를 채운 빈 정답지 골격을 만든다.
사용자는 각 구간의 knowledge_points 에 '이 구간 = 이 노하우'를 직접 적는다.

사용:  python scripts/make_answerkey_template.py V01
출력:  data/ground_truth/V01_answerkey.template.json   (확정 후 .template 떼서 사용)
"""
from __future__ import annotations
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402
from pipeline import vad, stt  # noqa: E402


def main(video_id: str):
    with open(os.path.join(ROOT, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    wav = os.path.join(ROOT, cfg["paths"]["raw_dir"], f"{video_id}.wav")
    assert os.path.exists(wav), f"오디오 없음: {wav}"

    dur = vad.duration(wav)
    segments = vad.detect_segments(wav, cfg)
    texts = stt.transcribe_segments(wav, segments, cfg)

    skeleton = {
        "video": video_id, "duration_sec": round(dur, 1),
        "instructions": "각 구간의 knowledge_points 에 실제 노하우를 적으세요. "
                        "없으면 빈 리스트로 두세요 (그 구간엔 암묵지 없음).",
        "segments": [
            {
                "seg_idx": i, "window": [round(a, 1), round(b, 1)],
                "stt": texts[i],
                "knowledge_points": [
                    {"action": "", "tacit": "", "evidence": ""}
                ],
            }
            for i, (a, b) in enumerate(segments)
        ],
    }
    out = os.path.join(ROOT, cfg["paths"]["gt_dir"], f"{video_id}_answerkey.template.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)
    print(f"[OK] 템플릿 생성: {out}")
    print(f"     구간 {len(segments)}개. 채운 뒤 '.template' 떼서 {video_id}_answerkey.json 로 저장.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "V01")
