# 코드 실행 전 아나콘다 가상환경 만들기
# pip install -U transformers qwen-vl-utils torch torchvision accelerate av decord bitsandbytes

import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

# ────────────────────────────────────────────
# [수정 가능] 설정
# ────────────────────────────────────────────
model_name     = "Qwen/Qwen3-VL-8B-Instruct"
video_path     = "/home/piai/다운로드/퀸/시은/프롬프팅 엔지니어링_260627/MOV/51662.mp4"
output_dir     = "/home/piai/다운로드/퀸/채림/results"
min_pixels     = 4 * 28 * 28
max_pixels     = 192 * 192
fps            = 0.3
max_new_tokens = 2048

# ────────────────────────────────────────────
# 프롬프트 5개 (각기 다른 전략)
# ────────────────────────────────────────────
PROMPTS = {

    # ── P1: CoT (Chain-of-Thought) ──────────────────────────────────────
    # 전략: 모델이 영상을 먼저 파악하고 → 행동을 단계적으로 추론하도록 유도
    # 영상 내용 고정 없이 모델이 스스로 파악
    "p1_cot": (
        "Watch this video carefully and analyze it step by step.\n\n"
        "Step 1. Identify what type of work is being performed in this video.\n"
        "- What hardware component is being handled?\n"
        "- What workstation or equipment is visible?\n"
        "- How many workers are present?\n\n"
        "Step 2. Identify all physical objects visible in the video.\n"
        "- List every component, tool, and part you can see.\n"
        "- Note the location of each object on the workstation.\n\n"
        "Step 3. For each observable action, record the following in order:\n\n"
        "Timestamp:\n"
        "Identified task type:\n"
        "Component being handled:\n"
        "Contact point on component:\n"
        "Contact point on workstation:\n"
        "Left hand action:\n"
        "Right hand action:\n"
        "Force direction (push/pull/rotate/press):\n"
        "Pause or hesitation observed (yes/no):\n"
        "Verification action performed:\n"
        "Result of action:\n"
        "Continuity with next action:\n\n"
        "Step 4. After listing all actions, summarize:\n"
        "- Total number of distinct actions observed\n"
        "- Any repeated actions (re-insertion, re-pressing, re-checking)\n"
        "- Any actions that deviate from a simple straight-line motion\n\n"
        "Rules:\n"
        "- Describe only what is visually observable. Do not infer intent.\n"
        "- Do not assume component names. Describe shape, color, size if unsure.\n"
        "- Split compound actions into the smallest possible units.\n"
        "- Use Korean for all output."
    ),

    # ── P2: Few-Shot (구체적 예시 제공) ────────────────────────────────
    # 전략: 이상적인 출력 예시를 3개 보여줘서 출력 품질을 고정
    # 예시는 조립 작업 일반 예시 (영상 내용 특정 안 함)
    "p2_fewshot": (
        "이 영상은 하드웨어 조립 작업 영상이다.\n"
        "영상을 시간 순서대로 분석하여 작업자의 행동을 최소 단위로 분해하여 기록하라.\n\n"
        "아래는 출력 형식의 예시이다. 이 형식을 반드시 따른다.\n\n"
        "=== 예시 시작 ===\n\n"
        "Timestamp: 00:01\n"
        "행동 분류: 부품 파지\n"
        "작업 대상 부품: 직사각형 초록색 회로기판 (슬롯 홈 1개 있음)\n"
        "파지 위치: 부품 상단 좌우 모서리\n"
        "왼손: 부품 좌측 상단 모서리를 엄지·검지로 집음\n"
        "오른손: 부품 우측 상단 모서리를 엄지·검지로 집음\n"
        "접촉 객체: 없음\n"
        "힘 방향: 없음\n"
        "일시 정지 여부: 없음\n"
        "확인 행동: 없음\n"
        "결과: 부품을 양손으로 수평으로 들어올림\n\n"
        "Timestamp: 00:03\n"
        "행동 분류: 방향 정렬\n"
        "작업 대상 부품: 직사각형 초록색 회로기판\n"
        "파지 위치: 부품 하단 좌우 끝\n"
        "왼손: 부품 하단 좌측을 아래에서 받침\n"
        "오른손: 부품 하단 우측을 잡고 홈 위치를 슬롯 키에 맞추기 위해 좌우로 미세 조정\n"
        "접촉 객체: 메인보드 슬롯 키\n"
        "힘 방향: 수평 이동 (좌우 미세 조정)\n"
        "일시 정지 여부: 있음 (약 1초 정지하여 홈 위치 육안 확인)\n"
        "확인 행동: 부품 하단 홈과 슬롯 키 위치를 육안으로 비교\n"
        "결과: 부품 홈이 슬롯 키와 일치하는 위치에 정렬됨\n\n"
        "Timestamp: 00:05\n"
        "행동 분류: 삽입 가압\n"
        "작업 대상 부품: 직사각형 초록색 회로기판\n"
        "파지 위치: 부품 상단 좌우 끝\n"
        "왼손: 부품 좌측 상단 끝을 엄지로 수직 하방 가압\n"
        "오른손: 부품 우측 상단 끝을 엄지로 수직 하방 가압\n"
        "접촉 객체: 메인보드 슬롯\n"
        "힘 방향: 수직 하방, 양끝 동시 가압\n"
        "일시 정지 여부: 없음\n"
        "확인 행동: 없음\n"
        "결과: 딸깍 소리와 함께 슬롯 래치 2개가 자동으로 올라와 잠김\n\n"
        "=== 예시 끝 ===\n\n"
        "위 형식으로 이 영상의 모든 행동을 처음부터 끝까지 빠짐없이 기록하라.\n\n"
        "추가 규칙:\n"
        "- 부품 이름을 모르면 형태(색상, 크기, 모양, 홈 위치)로 기술한다.\n"
        "- 영상에서 보이지 않는 것은 '확인 불가'로 표기한다.\n"
        "- 하나의 행동을 여러 Timestamp로 쪼개는 것을 허용한다.\n"
        "- 재가압, 재확인, 위치 재조정 행동은 반드시 별도 항목으로 기록한다."
    ),

    # ── P3: Atomic Decomposition + 신체부위 중심 ───────────────────────
    # 전략: 손가락 단위까지 분해, 신체 부위 기술을 최우선으로
    # 작업 내용 추론보다 신체 동작 관찰에 집중
    "p3_bodymotion": (
        "이 영상에서 작업자의 신체 동작을 극도로 세밀하게 분석하라.\n\n"
        "분석 우선순위:\n"
        "1순위: 손가락 각각의 위치와 힘 방향\n"
        "2순위: 손목 각도와 이동 경로\n"
        "3순위: 팔의 이동 방향\n"
        "4순위: 시선이 향하는 위치\n\n"
        "각 행동마다 아래 항목을 모두 채워 출력한다.\n\n"
        "Timestamp:\n"
        "행동 요약 (한 줄):\n"
        "다루는 객체 (모르면 형태 기술):\n"
        "객체 파지 방법:\n"
        "  - 왼손 엄지:\n"
        "  - 왼손 검지:\n"
        "  - 왼손 나머지 손가락:\n"
        "  - 오른손 엄지:\n"
        "  - 오른손 검지:\n"
        "  - 오른손 나머지 손가락:\n"
        "손목 각도 및 방향:\n"
        "  - 왼손목:\n"
        "  - 오른손목:\n"
        "힘 벡터:\n"
        "  - 왼손 힘 방향:\n"
        "  - 오른손 힘 방향:\n"
        "  - 힘의 크기 (강/중/약 추정):\n"
        "접촉면:\n"
        "  - 객체의 어느 면이 어디에 닿는지:\n"
        "동작 속도 (빠름/보통/느림/정지):\n"
        "일시 정지 여부 및 지속 시간:\n"
        "확인 행동 (손으로 눌러보기, 육안 확인 등):\n"
        "동작 결과:\n\n"
        "규칙:\n"
        "- 보이지 않는 손가락은 '화면 밖' 또는 '가려짐'으로 표기한다.\n"
        "- 추측 금지. 보이는 것만 기술한다.\n"
        "- 동작을 최소 단위로 쪼갠다. 0.5초 단위도 가능하다.\n"
        "- 두 손이 동시에 다른 동작을 할 경우 반드시 분리하여 기술한다."
    ),

    # ── P4: Structured JSON + 자기 검증 ────────────────────────────────
    # 전략: JSON 출력 + 모델이 스스로 불확실한 항목을 표기하도록 유도
    # 후처리 자동화 + 신뢰도 정보 확보
    "p4_json_verified": (
        "Analyze this hardware assembly video and output a JSON array.\n\n"
        "First, identify what task is being performed by observing:\n"
        "- The type of component being handled\n"
        "- The location on the workstation where work is done\n"
        "- The number of workers and their roles\n\n"
        "Then decompose every action into atomic units and output as JSON.\n\n"
        "Each action must follow this exact schema:\n"
        "{\n"
        "  \"timestamp\": \"MM:SS\",\n"
        "  \"action_index\": 1,\n"
        "  \"action_summary\": \"한 줄 요약\",\n"
        "  \"component\": \"부품명 또는 형태 기술\",\n"
        "  \"component_confidence\": \"high/medium/low\",\n"
        "  \"contact_on_component\": \"부품의 어느 부위에 접촉\",\n"
        "  \"contact_on_workstation\": \"워크스테이션의 어느 위치에 접촉\",\n"
        "  \"left_hand\": \"왼손 동작 상세\",\n"
        "  \"right_hand\": \"오른손 동작 상세\",\n"
        "  \"force_direction\": \"수직하방/수평/회전/없음\",\n"
        "  \"motion_speed\": \"빠름/보통/느림/정지\",\n"
        "  \"pause_detected\": true or false,\n"
        "  \"pause_duration_sec\": 숫자 또는 null,\n"
        "  \"verification_action\": \"확인 행동 기술 또는 null\",\n"
        "  \"result\": \"동작 결과\",\n"
        "  \"observable_only\": true\n"
        "}\n\n"
        "Rules:\n"
        "- Output JSON array only. No text outside the array.\n"
        "- If a field is not observable, use null.\n"
        "- component_confidence: high=clearly visible, medium=partially visible, low=inferred from context.\n"
        "- Split every compound action. One timestamp = one atomic action.\n"
        "- Re-pressing, re-checking, repositioning must each be separate entries.\n"
        "- Do not name components if not certain. Use shape/color/size description instead."
    ),

    # ── P5: Narrative + Dense Captioning ───────────────────────────────
    # 전략: 내러티브 형식으로 흐름을 서술 + 각 구간 dense caption
    # 행동 간 인과관계와 연속성 포착에 최적화
    "p5_narrative": (
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
    ),

}

# ────────────────────────────────────────────
# 모델 로드 (1회만)
# ────────────────────────────────────────────
print("모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)
print("모델 로드 완료\n")

os.makedirs(output_dir, exist_ok=True)

# ────────────────────────────────────────────
# 프롬프트별 루프
# ────────────────────────────────────────────
for idx, (prompt_name, prompt_text) in enumerate(PROMPTS.items(), start=1):
    print(f"[{idx}/5] {prompt_name} 실행 중...")

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
                {
                    "type": "text",
                    "text": prompt_text,
                },
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
        repetition_penalty=1.05,
    )

    output_text = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    # 결과 저장: prompt_1.txt ~ prompt_5.txt
    save_path = os.path.join(output_dir, f"prompt_{idx}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"[프롬프트 {idx}: {prompt_name}]\n")
        f.write("=" * 60 + "\n")
        f.write("[사용한 프롬프트]\n")
        f.write(prompt_text + "\n")
        f.write("=" * 60 + "\n\n")
        f.write("[모델 출력 결과]\n\n")
        f.write(output_text)

    print(f"  → 저장: {save_path}\n")

    del inputs, output_ids
    torch.cuda.empty_cache()

print("전체 완료!")
