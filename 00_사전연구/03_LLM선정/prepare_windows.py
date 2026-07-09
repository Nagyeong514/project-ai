"""
융합 LLM 선정 실험 — 윈도우 세트 구성 + 케이스 라벨링 (2026-07-09).

원천 데이터(이미 존재, 재생성 안 함 — 입력 고정 원칙):
  - STT:  STEP3_전처리/transcripts/<video_id>.json
  - VLM:  STEP4_YOLO_VLM관찰/output/vlm_observations/<video_id>.observations.json
  - 정렬: STEP5_LLM으로_암묵지_후보생성/output/aligned_windows/<video_id>.windows.json
    (STT+VLM을 ±4s 윈도우로 이미 묶어놓은 산출물 — 본 실험은 이 윈도우를 그대로 쓴다)

케이스 정의(윈도우 내용 기준, 코드 유도):
  A = fusion(행동+발화) / B = action_only(행동만, 침묵) / C = utterance_only(발화만)
할당량: A 6 / B 5 / C 4 = 총 15. 클립 라운드로빈으로 분산 선정(결정적 — 재실행 동일).

산출물:
  - windows_labels.json  : 사전 라벨링 확정본(케이스별 분해의 유일 기준)
  - frozen_windows.json  : 세 모델이 공유하는 직렬화 payload(윈도우당 1개, window_id="W01"
                           로 통일 — 프롬프트 형태를 모든 호출에서 동일하게 유지)
직렬화는 STEP5 실전 코드(QwenLLMFusion._serialize)를 그대로 사용 — 실사용 비교 원칙.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "STEP5_LLM으로_암묵지_후보생성"))

from tacit_common import artifacts  # noqa: E402
from step5_components.llm_fusion import QwenLLMFusion  # noqa: E402

WINDOWS_DIR = str(PROJECT_ROOT / "STEP5_LLM으로_암묵지_후보생성/output/aligned_windows")
VIDEO_IDS = [
    "CLIP1_정상조립과정.mp4",
    "CLIP2.mp4",
    "CLIP3_재부팅시도 전까지.mp4",
    "CLIP4_재부팅및BIOS.mp4",
]
QUOTA = {"A": 6, "B": 5, "C": 4}
CASE_OF = {"fusion": "A", "action_only": "B", "utterance_only": "C"}


def main() -> None:
    comp = QwenLLMFusion()  # 직렬화만 사용(모델 로딩 없음)
    pool: dict[str, list] = defaultdict(list)  # case -> [(clip_key, vid, idx, window)]
    for vid in VIDEO_IDS:
        wins = artifacts.load_aligned_windows(WINDOWS_DIR, vid)
        clip_key = vid.split(".")[0].split("_")[0]  # CLIP1..CLIP4
        for i, w in enumerate(wins, start=1):
            case = CASE_OF.get(w.case)
            if case:
                pool[case].append((clip_key, vid, i, w))

    labels, frozen = [], {}
    for case, quota in QUOTA.items():
        # 클립 라운드로빈: 클립별 큐에서 번갈아 뽑아 특정 클립 편중 방지(결정적).
        by_clip: dict[str, list] = defaultdict(list)
        for item in pool[case]:
            by_clip[item[0]].append(item)
        clips = sorted(by_clip)
        picked, ptr = [], 0
        while len(picked) < quota and any(by_clip[c] for c in clips):
            c = clips[ptr % len(clips)]
            ptr += 1
            if by_clip[c]:
                picked.append(by_clip[c].pop(0))
        if len(picked) < quota:
            raise SystemExit(f"케이스 {case} 풀 부족: {len(picked)}/{quota}")
        for clip_key, vid, idx, w in picked:
            tid = f"{clip_key}_W{idx:02d}"
            payload = comp._serialize([w])[0]  # 단일 윈도우 → window_id="W01"
            n_utt = sum(1 for u in w.utterances if not u.repeat_hallucination)
            labels.append({
                "id": tid, "clip": clip_key, "video_id": vid, "orig_index": idx,
                "case": case, "window_case_raw": w.case,
                "window_start": w.window_start, "window_end": w.window_end,
                "n_actions": len(w.actions), "n_utterances": n_utt,
            })
            frozen[tid] = {"video_id": vid, "payload": payload}

    labels.sort(key=lambda r: (r["case"], r["clip"], r["orig_index"]))
    (HERE / "windows_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "frozen_windows.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    dist = Counter(r["case"] for r in labels)
    per_clip = Counter((r["case"], r["clip"]) for r in labels)
    print(f"[prepare] 총 {len(labels)}개 — 분포 {dict(dist)}")
    for r in labels:
        print(f"  {r['case']} {r['id']:12s} {r['window_start']:7.1f}~{r['window_end']:7.1f}s "
              f"acts={r['n_actions']} utts={r['n_utterances']}")
    assert all(dist[c] == q for c, q in QUOTA.items()), "할당량 불일치"
    assert len(frozen) == len(labels)
    print("[OK] windows_labels.json / frozen_windows.json 저장")


if __name__ == "__main__":
    main()
