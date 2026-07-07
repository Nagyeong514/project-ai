"""5-A OOM 진단 전용 — llm_fusion.py/aligner.py 코드는 안 건드리고, 같은 프로세스(모델 1회
로드) 안에서 merge_cap=20(5-A) vs merge_cap=99999(구동작 재현)를 순차로 시도해 OOM이
merge_cap과 무관한 LLM 로딩 자체 문제인지 가려낸다."""
import sys, json, traceback
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai/STEP5_LLM으로_암묵지_후보생성")

import torch
from tacit_common import artifacts
from tacit_common.config import PipelineConfig
from tacit_common.schema.intermediate import FrameMeta
from step5_components.aligner import WindowAligner
from step5_components.llm_fusion import QwenLLMFusion

CFG = "/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/config.yaml"
VIDEO_ID = "CLIP1_정상조립과정.mp4"

cfg = PipelineConfig.load(CFG)
transcript = artifacts.load_transcript(cfg.paths.resolve(cfg.paths.step3_transcript_dir), VIDEO_ID)
flat_dets = artifacts.load_detections(cfg.paths.resolve(cfg.paths.step4_detections_dir), VIDEO_ID)
actions = artifacts.load_observations(cfg.paths.resolve(cfg.paths.step4_observations_dir), VIDEO_ID)
fmeta = artifacts.load_frames_meta(cfg.paths.resolve(cfg.paths.step3_frames_dir), VIDEO_ID)

print(f"[DIAG] actions={len(actions)} utterances={len(transcript.utterances)} detections={len(flat_dets)}")

llm = QwenLLMFusion(**cfg.llm.params)
print("[DIAG] LLM 로딩 시작...")
llm._load()
alloc = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"[DIAG] 로딩 직후: allocated={alloc:.2f}GiB reserved={reserved:.2f}GiB")

meta = FrameMeta(video_id=VIDEO_ID, path=VIDEO_ID, fps=fmeta["fps"], n_frames=len(fmeta["frame_paths"]))

for label, cap in [("5-A(cap=20)", 20.0), ("구동작재현(cap=99999)", 99999.0)]:
    aligner = WindowAligner(window_sec=cfg.aligner.params.get("window_sec", 4.0),
                            merge_overlapping=True, merge_cap_sec=cap)
    windows = aligner.align(actions, transcript, flat_dets)
    widths = [w.window_end - w.window_start for w in windows]
    payload = llm._serialize(windows)
    payload_json = json.dumps(payload, ensure_ascii=False)
    print(f"\n[DIAG] === {label} === 윈도우 {len(windows)}개, 최대폭 {max(widths):.1f}s, "
          f"직렬화 payload {len(payload_json)}자")
    try:
        torch.cuda.reset_peak_memory_stats()
        doc = llm.fuse(windows, meta)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[DIAG] {label}: 성공, 후보 {len(doc.candidates)}건, generate중 peak={peak:.2f}GiB")
    except torch.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[DIAG] {label}: OOM! peak={peak:.2f}GiB, {e}")
    except Exception as e:
        print(f"[DIAG] {label}: 기타 예외 {type(e).__name__}: {e}")
        traceback.print_exc()

print("\n[DIAG] 완료")
