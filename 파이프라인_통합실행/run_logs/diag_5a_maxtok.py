"""4번 연속 완전 동일 OOM(13.20GiB/8.99GiB/8.77GiB)의 원인 진단 전용 스크립트.
llm_fusion.py 코드는 안 건드리고, 실제 fuse()와 동일한 입력으로 generate()의
max_new_tokens 값만 바꿔가며 OOM 재현 여부를 확인해 원인을 max_new_tokens로
좁힐지 배제할지 가른다. 이 스크립트의 결과로 코드를 자동으로 바꾸지 않는다."""
import sys, torch
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai/STEP5_LLM으로_암묵지_후보생성")

from tacit_common import artifacts
from tacit_common.config import PipelineConfig
from tacit_common.schema.intermediate import FrameMeta
from step5_components.aligner import WindowAligner
from step5_components.llm_fusion import QwenLLMFusion
from step5_prompts.llm_fusion_prompt import build_fusion_messages

CFG = "/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/config.yaml"
VIDEO_ID = "CLIP1_정상조립과정.mp4"

cfg = PipelineConfig.load(CFG)
transcript = artifacts.load_transcript(cfg.paths.resolve(cfg.paths.step3_transcript_dir), VIDEO_ID)
flat_dets = artifacts.load_detections(cfg.paths.resolve(cfg.paths.step4_detections_dir), VIDEO_ID)
actions = artifacts.load_observations(cfg.paths.resolve(cfg.paths.step4_observations_dir), VIDEO_ID)

aligner = WindowAligner(**cfg.aligner.params)
windows = aligner.align(actions, transcript, flat_dets)

llm = QwenLLMFusion(**cfg.llm.params)
llm._load()
payload = llm._serialize(windows)
messages = build_fusion_messages(VIDEO_ID, payload)
text = llm._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
target_device = llm._model.device if llm.device == "auto" else llm.device
inputs = llm._tok(text, return_tensors="pt").to(target_device)
print(f"[DIAG] 입력 프롬프트 토큰 수: {inputs['input_ids'].shape[1]}")

print("\n[DIAG] ==== fuse() 전체(재시도 루프 포함) 직접 호출 ====")
orig_infer = llm._infer
call_count = [0]
def traced_infer(messages):
    call_count[0] += 1
    text = llm._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n_tok = len(llm._tok(text)["input_ids"])
    torch.cuda.reset_peak_memory_stats()
    print(f"[DIAG] _infer 호출 #{call_count[0]}: 프롬프트 토큰수={n_tok}")
    try:
        out = orig_infer(messages)
        peak0 = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"[DIAG] _infer 호출 #{call_count[0]}: 성공, GPU0 peak={peak0:.2f}GiB, raw길이={len(out)}")
        # 1차 응답이 왜 스키마 검증에 실패하는지 원문으로 판정하기 위해 저장
        rawpath = f"/home/ai_user/team_a2/members/안나경/project-ai/파이프라인_통합실행/run_logs/diag_raw_attempt{call_count[0]}.txt"
        with open(rawpath, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[DIAG] raw 저장: {rawpath}")
        # 검증 실패 사유도 즉석에서 확인
        try:
            llm._parse_and_validate(out, VIDEO_ID)
            print(f"[DIAG] 호출 #{call_count[0]} 응답: 스키마 검증 통과")
        except Exception as ve:
            print(f"[DIAG] 호출 #{call_count[0]} 응답: 스키마 검증 실패 — {type(ve).__name__}: {str(ve)[:500]}")
        return out
    except torch.OutOfMemoryError as e:
        peak0 = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"[DIAG] _infer 호출 #{call_count[0]}: OOM! GPU0 peak={peak0:.2f}GiB, 프롬프트토큰수={n_tok}")
        raise
llm._infer = traced_infer

from tacit_common.schema.intermediate import FrameMeta as FM
fmeta = artifacts.load_frames_meta(cfg.paths.resolve(cfg.paths.step3_frames_dir), VIDEO_ID)
meta2 = FM(video_id=VIDEO_ID, path=VIDEO_ID, fps=fmeta["fps"], n_frames=len(fmeta["frame_paths"]))
try:
    doc = llm.fuse(windows, meta2)
    print(f"[DIAG] fuse() 최종 성공: 후보 {len(doc.candidates)}건, 총 _infer 호출 {call_count[0]}회")
except Exception as e:
    print(f"[DIAG] fuse() 최종 실패({type(e).__name__}): {e}, 총 _infer 호출 {call_count[0]}회")

print("\n[DIAG] 완료")
