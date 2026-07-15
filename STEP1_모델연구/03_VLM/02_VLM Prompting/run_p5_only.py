import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

video_path     = "/home/piai/다운로드/퀸/시은/프롬프팅 엔지니어링_260627/MOV/51662.mp4"
output_dir     = "/home/piai/다운로드/퀸/채림/results"
min_pixels     = 4 * 28 * 28
max_pixels     = 192 * 192
fps            = 0.3
max_new_tokens = 2048

p5_prompt = (
    "이 영상은 하드웨어 조립 작업 영상이다.\n\n"
    "아래 두 파트로 나누어 분석하라.\n\n"
    "=== PART 1: 영상 개요 ===\n"
    "다음 항목을 먼저 기술하라.\n"
    "- 작업 유형: 영상에서 수행되는 작업이 무엇인지 (부품명, 장비명, 작업 목적)\n"
    "- 작업자 수 및 역할 분담\n"
    "- 전체 작업 흐름 요약 (3~5문장)\n"
    "- 영상에서 관찰된 전체 부품 목록\n"
    "- 특이 사항 (재시도, 멈춤, 위치 재조정 등)\n\n"
    "=== PART 2: 구간별 Dense Caption ===\n"
    "영상을 2초 단위 구간으로 나누어 각 구간을 아래 형식으로 기술한다.\n\n"
    "[구간 00:00~00:02]\n"
    "장면 기술: (이 구간에서 카메라에 보이는 것 전체를 기술)\n"
    "주요 행동:\n"
    "  - 행동1: 왼손 동작 + 오른손 동작 + 접촉 객체 + 힘 방향\n"
    "  - 행동2: (있을 경우)\n"
    "세부 관찰:\n"
    "  - 부품 파지 방법 (어느 부위를 어떻게 잡는지)\n"
    "  - 슬롯/소켓 래치 상태 (열림/닫힘/변화)\n"
    "  - 케이블 간섭 여부\n"
    "  - 작업자 손의 떨림 또는 힘 조절 관찰\n"
    "일시 정지: (있으면 위치와 지속시간)\n"
    "확인 행동: (있으면 무엇을 어떻게 확인했는지)\n"
    "이전 구간과의 연속성: (이전 행동에서 이어지는 맥락)\n"
    "다음 구간 예고: (이 구간 마지막 상태 기술)\n\n"
    "규칙:\n"
    "- 부품 이름이 불확실하면 형태(색, 크기, 모양, 홈 개수)로 기술한다.\n"
    "- 추측 금지. 보이는 것만 기술한다.\n"
    "- 재가압, 재확인, 위치 재조정은 반드시 별도 행동으로 기록한다.\n"
    "- 래치, 클립, 걸쇠 등의 상태 변화는 반드시 기록한다.\n"
    "- 두 손이 서로 다른 동작을 할 경우 반드시 분리하여 기술한다."
)

print("모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
print("모델 로드 완료\n")

os.makedirs(output_dir, exist_ok=True)

print("[P5] p5_narrative 실행 중...")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps,
            },
            {"type": "text", "text": p5_prompt},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages, return_video_kwargs=True, return_video_metadata=True
)
video_metadata = None
if video_inputs is not None and len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
    video_inputs, video_metadata = zip(*video_inputs)
    video_inputs = list(video_inputs)
    video_metadata = list(video_metadata)
if isinstance(video_kwargs.get("fps"), list):
    video_kwargs["fps"] = video_kwargs["fps"][0]

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    video_metadata=video_metadata,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
).to(model.device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    repetition_penalty=1.2,
)

output_text = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)[0]

save_path = os.path.join(output_dir, "prompt_5.txt")
with open(save_path, "w", encoding="utf-8") as f:
    f.write("[프롬프트 5: p5_narrative]\n")
    f.write("=" * 60 + "\n")
    f.write("[사용한 프롬프트]\n")
    f.write(p5_prompt + "\n")
    f.write("=" * 60 + "\n\n")
    f.write("[모델 출력 결과]\n\n")
    f.write(output_text)

print(f"  → 저장: {save_path}")
print(output_text)
