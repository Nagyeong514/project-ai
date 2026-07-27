
"""CLIP4 STAGE2(직렬화)+STAGE3(메시지·chat template) 재현 + 토큰 계산."""
import json, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai/STEP5_LLM으로_암묵지_후보생성")
from tacit_common import artifacts
from step5_components.aligner import WindowAligner
from step5_components.llm_fusion import QwenLLMFusion
from step5_prompts.llm_fusion_prompt import build_fusion_messages, FUSION_SYSTEM_PROMPT

BASE = "/home/ai_user/team_a2/members/안나경/project-ai"
VID = "CLIP4_재부팅및BIOS.mp4"
windows = artifacts.load_aligned_windows(
    f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/aligned_windows_motion_obs", VID)

fus = QwenLLMFusion.__new__(QwenLLMFusion)  # __init__ 불필요(순수 직렬화만)
payload = QwenLLMFusion._serialize(fus, windows)

# W03 저장본 vs payload
saved = json.load(open(f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/aligned_windows_motion_obs/{VID}.windows.json"))
print("=== W03 저장본(windows.json) ===")
print(json.dumps(saved["windows"][2], ensure_ascii=False, indent=1)[:4000])
print("\n=== W03 직렬화 payload ===")
print(json.dumps(payload[2], ensure_ascii=False, indent=1))
print("\n=== W01 payload (repeat_count 접힘 예시) ===")
print(json.dumps(payload[0], ensure_ascii=False, indent=1))

msgs = build_fusion_messages(VID, payload)
sys_p, user_p = msgs[0]["content"], msgs[1]["content"]
print("\n=== user 메시지 앞 900자 ===")
print(user_p[:900])
compact = json.dumps(payload, ensure_ascii=False, separators=(',',':'))
pretty = json.dumps(payload, ensure_ascii=False, indent=2)
print(f"\npayload compact {len(compact)}자 vs indent=2 {len(pretty)}자 (절감 {len(pretty)-len(compact)}자)")
print(f"system {len(sys_p)}자 / user {len(user_p)}자")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text)["input_ids"]
print(f"\nchat template 적용 후 전체 {len(text)}자 / {len(ids)}토큰")
print("=== template 앞 220자 ===");  print(repr(text[:220]))
print("=== template 끝 300자 ===");  print(repr(text[-300:]))
print("system 토큰:", len(tok(sys_p)['input_ids']), "user 토큰:", len(tok(user_p)['input_ids']),
      "payload(compact) 토큰:", len(tok(compact)['input_ids']),
      "payload(indent2) 토큰:", len(tok(pretty)['input_ids']))
# enable_thinking 기본(true)와 비교
t2 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("enable_thinking 미지정과 diff:", repr(t2[-50:]) if t2!=text else "동일")
