
"""CLIP4(모션관찰 실전런) STAGE1 aligner 추적 — 병합 결정 로그 포함 재현."""
import json, sys
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai/STEP5_LLM으로_암묵지_후보생성")
from tacit_common import artifacts
from step5_components.aligner import WindowAligner

BASE = "/home/ai_user/team_a2/members/안나경/project-ai"
VID = "CLIP4_재부팅및BIOS.mp4"
transcript = artifacts.load_transcript(f"{BASE}/STEP3_전처리/transcripts", VID)
dets = artifacts.load_detections(f"{BASE}/STEP4_YOLO_VLM관찰/output/detections", VID)
acts = artifacts.load_observations(f"{BASE}/STEP4_YOLO_VLM관찰/output_motion/vlm_observations_original_ts", VID)

print(f"발화 {len(transcript.utterances)}건 / 관찰 {len(acts)}건 / 검출 {len(dets)}건")
al = WindowAligner(window_sec=4.0, merge_overlapping=True, merge_cap_sec=20.0)

# 1) 병합 전 상태 재현
utterances = sorted(transcript.utterances, key=lambda u: u.start)
used = set()
pre = []
for a in sorted(acts, key=lambda x: x.timestamp):
    t0 = max(0.0, a.timestamp - 4.0); t1 = a.timestamp + 4.0
    w_utts = [u for i,u in enumerate(utterances) if u.start <= t1 and t0 <= u.end]
    for i,u in enumerate(utterances):
        if u.start <= t1 and t0 <= u.end: used.add(i)
    pre.append(("action", t0, t1, a, w_utts))
leftover = [u for i,u in enumerate(utterances) if i not in used]
print(f"\n행동 앵커 윈도우 {len(pre)}개, leftover 발화 {len(leftover)}건:")
for t in ("action",):
    for _,t0,t1,a,wu in pre:
        print(f"  anchor ts={a.timestamp:6.1f} → [{t0:6.1f},{t1:6.1f}] utt겹침={len(wu)} | {a.action[:30]}")
for u in leftover:
    print(f"  leftover utt [{u.start:.2f},{u.end:.2f}] {u.raw_text[:30]}")

# 2) 진짜 align + 병합 로그(몽키패치)
orig_merge = WindowAligner._merge
def logged_merge(self, windows):
    print(f"\n병합 전 윈도우 {len(windows)}개(시간순 정렬 후):")
    for i,w in enumerate(windows):
        print(f"  [{i}] [{w.window_start:7.2f},{w.window_end:7.2f}] a={len(w.actions)} u={len(w.utterances)}")
    merged = [windows[0]]
    for w in windows[1:]:
        last = merged[-1]
        overlap = w.window_start <= last.window_end
        span = max(w.window_end, last.window_end) - last.window_start
        capped = span <= self.merge_cap_sec
        if overlap and capped:
            print(f"  MERGE: [{last.window_start:.2f},{last.window_end:.2f}] + [{w.window_start:.2f},{w.window_end:.2f}] → span {span:.2f}s ≤ 20")
            before_u = len(last.utterances); add_u = len(w.utterances)
            last.window_end = max(last.window_end, w.window_end)
            last.actions.extend(w.actions)
            self._extend_unique_utt(last.utterances, w.utterances)
            self._extend_unique_det(last.detections, w.detections)
            if before_u + add_u != len(last.utterances):
                print(f"    utt dedup 발생: {before_u}+{add_u} → {len(last.utterances)}")
        else:
            why = "겹침없음" if not overlap else f"cap초과(span {span:.2f}s > 20)"
            print(f"  SPLIT: [{w.window_start:.2f},{w.window_end:.2f}] — {why}")
            merged.append(w)
    return merged
WindowAligner._merge = logged_merge
windows = al.align(acts, transcript, dets)
print(f"\n최종 윈도우 {len(windows)}개")
for i,w in enumerate(windows,1):
    print(f"W{i:02d} [{w.window_start:7.2f},{w.window_end:7.2f}] case={w.case} a={len(w.actions)} u={len(w.utterances)} d={len(w.detections)}")
    for u in w.utterances: print(f"      utt {u.start:.2f} {u.raw_text[:36]}")

# 저장본과 대조
saved = json.load(open(f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/aligned_windows_motion_obs/{VID}.windows.json"))
recomputed = [w.model_dump() for w in windows]
print("\n저장본과 재계산 일치:", recomputed == saved["windows"])
