"""max_memory가 4bit+device_map=auto에서 실제로 먹히는지 결정적으로 확인.
GPU0에 극단적으로 적은 값(2GiB)을 줘서 hf_device_map이 실제로 그렇게 나뉘는지 직접 본다."""
import sys, torch
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0, "/home/ai_user/team_a2/members/안나경/project-ai/STEP5_LLM으로_암묵지_후보생성")
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-14B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

for mm in [{0: "20GiB", 1: "9GiB"}, {0: "16GiB", 1: "5GiB"}]:
    print(f"\n[DIAG] ==== max_memory={mm} 로 로드 시도 ====")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", quantization_config=bnb, max_memory=mm,
    )
    dm = getattr(model, "hf_device_map", None)
    print(f"[DIAG] hf_device_map = {dm if dm is None else 'dict 있음(아래 요약)'}")
    if dm is not None:
        from collections import Counter
        c = Counter(str(v) for v in dm.values())
        print(f"[DIAG] 디바이스별 모듈 개수: {dict(c)}")
    for i in range(torch.cuda.device_count()):
        print(f"[DIAG] GPU{i} allocated={torch.cuda.memory_allocated(i)/1024**3:.2f}GiB "
              f"reserved={torch.cuda.memory_reserved(i)/1024**3:.2f}GiB")
    # 실제 레이어 텐서가 어느 디바이스에 있는지 직접 확인(hf_device_map 유무와 무관하게 확정적)
    try:
        layers = model.model.layers
        print(f"[DIAG] 총 레이어 수: {len(layers)}")
        print(f"[DIAG] layer[0] 디바이스: {next(layers[0].parameters()).device}")
        print(f"[DIAG] layer[{len(layers)-1}] 디바이스: {next(layers[-1].parameters()).device}")
        mid = len(layers) // 2
        print(f"[DIAG] layer[{mid}](중간) 디바이스: {next(layers[mid].parameters()).device}")
    except Exception as e:
        print(f"[DIAG] 레이어 디바이스 확인 실패: {e}")
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
