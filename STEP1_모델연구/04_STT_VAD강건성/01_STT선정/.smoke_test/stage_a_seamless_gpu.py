import sys
import time

PROJECT = "/home/ai_user/team_a2/members/안나경/project-ai/00_사전연구/01_STT선정"
sys.path.insert(0, PROJECT)

import torch
from pipeline.stt.seamless_runner import SeamlessM4Tv2Runner

print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

t_load0 = time.perf_counter()
runner = SeamlessM4Tv2Runner(
    model_id="facebook/seamless-m4t-v2-large",
    tgt_lang="kor",
    device="cuda",
    window_s=25.0,
)
t_load = time.perf_counter() - t_load0
print(f"[load] {t_load:.1f}s")

if torch.cuda.is_available():
    print(f"[load 직후 GPU0] allocated={torch.cuda.memory_allocated(0)/1e9:.2f}GB "
          f"reserved={torch.cuda.memory_reserved(0)/1e9:.2f}GB")

t0 = time.perf_counter()
result = runner.transcribe(f"{PROJECT}/data/raw_v2/CLIP3.wav")
elapsed = time.perf_counter() - t0
print(f"[transcribe] {elapsed:.1f}s  rtf={result.rtf:.4f}  segments={len(result.segments)}")
for seg in result.segments:
    print(f"  [{seg.start:.1f}-{seg.end:.1f}] {seg.text!r}")

if torch.cuda.is_available():
    print(f"[transcribe 직후 GPU0] allocated={torch.cuda.memory_allocated(0)/1e9:.2f}GB "
          f"max_allocated={torch.cuda.max_memory_allocated(0)/1e9:.2f}GB "
          f"max_reserved={torch.cuda.max_memory_reserved(0)/1e9:.2f}GB")
