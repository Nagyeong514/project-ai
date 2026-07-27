# 실행: conda run -n qwen-vl python test_qwen_final_v2.py [video_path] [output_dir]

import os, sys, datetime, traceback
os.environ["FORCE_QWENVL_VIDEO_READER"] = "av"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ────────────────────────────────────────────
# [수정 가능] 영상 → 부품명 매핑
# ────────────────────────────────────────────
VIDEOS = {
    "/home/piai/다운로드/퀸/시은/프롬프팅 엔지니어링_260627/MOV/51661.mp4": "RAM(DDR 메모리 모듈)",
    "/home/piai/다운로드/퀸/시은/프롬프팅 엔지니어링_260627/MOV/51662.mp4": "GPU(PCIe 그래픽카드)",
    # 영상 추가 시: "/경로/영상.mp4": "부품명"
}

# ────────────────────────────────────────────
# [수정 가능] 경로 설정
# ────────────────────────────────────────────
DEFAULT_VIDEO  = "/home/piai/다운로드/퀸/시은/프롬프팅 엔지니어링_260627/MOV/51661.mp4"
DEFAULT_OUTPUT = "/home/piai/다운로드/퀸/시은/프롬프팅 엔지니어링_260627/결과"

video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

PART       = VIDEOS.get(video_path, "알 수 없는 부품")
video_name = os.path.splitext(os.path.basename(video_path))[0]
ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_dir, exist_ok=True)

# ────────────────────────────────────────────
# [수정 가능] 모델 설정
# ────────────────────────────────────────────
model_name = "Qwen/Qwen3-VL-8B-Instruct"
SETTINGS = {
    "max_pixels":         192 * 192,   # 시은 실험 검증값
    "fps":                0.5,         # RAM(18.7초)은 0.5, GPU(36.7초)는 0.3 권장
    "max_new_tokens":     4000,
    "do_sample":          False,
    "repetition_penalty": 1.2,         # 짧은 영상 무한반복 방지
}

# ────────────────────────────────────────────
# 모델 로드
# ────────────────────────────────────────────
print(f"[시작] 영상: {video_path}")
print(f"[부품] {PART}")
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

# ────────────────────────────────────────────
# 프롬프트 정의 (2개)
# ────────────────────────────────────────────

# system_prompt: 시은 구조 채택 (역할 고정 + 금지어 통제)
SYSTEM_PROMPT = """
너는 워크스테이션 부품 조립 영상에서 숙련자의 암묵지(tacit knowledge)를 추출하는 전문 분석가다.

핵심 원칙:
- 추상 동사 금지. 모든 동작은 관찰 가능한 신체·물체 움직임으로 분해한다.
  나쁜 예: "부품을 장착한다" / "확인한다"
  좋은 예: "오른손 엄지·검지가 모듈 끝단 좌우를 잡고 수직↓으로 균등 압력을 가함"

- "확인한다"는 절대 단독으로 쓰지 않는다. 반드시 아래 형태로 분해한다:
  · 시각 → "시선을 [방향]으로 이동해 [무엇]을 관찰"
  · 촉각 → "[손·손가락]으로 [부품]을 [방향] [약/중/강] 힘으로 건드려 저항 여부 감지"
  · 청각 → "'딸깍' 소리로 잠금 인식"

- 보조손도 반드시 접촉 부위와 힘 방향을 기술한다.
- 정확히 보이지 않으면 [추정], 직전·직후 프레임으로 추론하면 [프레임 추론]을 붙인다.
- 수치(mm, 각도, kg)는 영상에서 명확히 읽히는 경우에만 기재한다.

절대 금지:
- "확인한다 / 설치한다 / 장착한다 / 작업한다" 단독 사용
- [추정]·[프레임 추론] 없이 보이지 않는 정보 단정
- 영상에 없는 행동 생성
""".strip()

# user_prompt: 채림 Final B (CoT+FewShot) + 시은 [1]~[7] 섹션 + P3 손가락 필드 통합
USER_PROMPT_TEMPLATE = """
[고정 사실 — 반드시 따를 것]
이 영상의 대상 부품은 {PART}이다. 메인보드 슬롯에 장착하는 부품이다.
- 이 부품을 다른 부품명으로 절대 부르지 마라.
- 그 부품에 존재하지 않는 동작을 지어내지 마라.
- 동작 기술(손가락·방향·힘 변화)에만 집중하라.

──────────────────────────────
Step 1. 영상 전체 파악
──────────────────────────────

영상 전체를 보고 다음을 먼저 파악한다.

- 작업 유형 및 작업자 수
- 전체 작업 순서 (3~5문장 요약)
- 특이 행동 메모 (멈춤, 재시도, 재가압, 래치 상태 변화 등)
- 관찰된 전체 부품 목록 (이름 불확실 시 형태로 기술)

──────────────────────────────
Step 2. 출력 형식 (아래 예시를 반드시 따른다)
──────────────────────────────

=== 예시 시작 ===

Timestamp: 00:01
행동 분류: 부품 파지
작업 대상 부품: {PART}
파지 위치: 부품 상단 좌우 모서리
주도손:
  - 오른손 엄지: 부품 우측 상단 모서리 상면 접촉
  - 오른손 검지: 부품 우측 상단 모서리 측면 접촉
  - 오른손 나머지: 부품 우측 하단으로 감싸 받침
보조손:
  - 왼손 엄지: 부품 좌측 상단 모서리 상면 접촉
  - 왼손 검지: 부품 좌측 상단 모서리 측면 접촉
  - 왼손 나머지: 부품 좌측 하단으로 감싸 받침
접촉 객체: 없음
힘 방향: 없음
동작 속도: 느림
일시 정지 여부: 없음
확인 행동: 없음
이전 행동과의 연속성: 시작 동작
결과: 부품을 양손으로 수평으로 들어올림

Timestamp: 00:03
행동 분류: 방향 정렬
작업 대상 부품: {PART}
파지 위치: 부품 하단 좌우 끝
주도손:
  - 오른손 엄지: 부품 우측 하단 끝 상면 접촉
  - 오른손 검지: 부품 우측 하단 끝 측면 접촉
  - 오른손 나머지: 가려짐
보조손:
  - 왼손 엄지: 부품 좌측 하단을 아래에서 받침
  - 왼손 검지: 부품 좌측 측면 접촉
  - 왼손 나머지: 가려짐
접촉 객체: 메인보드 슬롯 키
힘 방향: 수평 (좌우 미세 조정)
동작 속도: 느림
일시 정지 여부: 있음 (약 1초)
확인 행동: 시선을 슬롯 키 방향으로 이동해 부품 홈과 슬롯 키 일치 여부를 육안으로 관찰
이전 행동과의 연속성: 부품 파지 후 슬롯 위로 이동한 직후
결과: 부품 홈이 슬롯 키와 일치하는 위치에 정렬됨

Timestamp: 00:05
행동 분류: 삽입 가압
작업 대상 부품: {PART}
파지 위치: 부품 상단 좌우 끝
주도손:
  - 오른손 엄지: 부품 우측 상단 끝 상면을 수직↓ 가압
  - 오른손 검지: 부품 우측 상단 끝 측면 고정
  - 오른손 나머지: 가려짐
보조손:
  - 왼손 엄지: 부품 좌측 상단 끝 상면을 수직↓ 가압
  - 왼손 검지: 부품 좌측 상단 끝 측면 고정
  - 왼손 나머지: 가려짐
접촉 객체: 메인보드 슬롯
힘 방향: 수직 하방, 양끝 동시 가압
동작 속도: 보통
일시 정지 여부: 없음
확인 행동: 없음
이전 행동과의 연속성: 방향 정렬 완료 직후 삽입
결과: 슬롯 래치 2개가 자동으로 올라와 잠김

Timestamp: 00:06
행동 분류: 재가압 (장착 확인)
작업 대상 부품: {PART}
파지 위치: 부품 상단 중앙
주도손:
  - 오른손 엄지: 부품 우측 상단을 수직↓ 약한 힘으로 가압
  - 오른손 검지: 부품 상면 접촉, 저항 감지
  - 오른손 나머지: 가려짐
보조손:
  - 왼손 엄지: 부품 좌측 상단을 수직↓ 약한 힘으로 가압
  - 왼손 검지: 부품 상면 접촉, 저항 감지
  - 왼손 나머지: 가려짐
접촉 객체: 메인보드 슬롯
힘 방향: 수직 하방, 약한 힘 (강→약 이완)
동작 속도: 느림
일시 정지 여부: 있음 (약 0.5초)
확인 행동: 양 엄지로 부품 상단을 한 번 더 눌러 저항 여부 촉각으로 감지, 저항 소멸 시 잠금 완료 인식
이전 행동과의 연속성: 삽입 직후 재확인
결과: 래치 고정 상태 촉각 확인 완료

=== 예시 끝 ===

──────────────────────────────
Step 3. 행동 로그 생성
──────────────────────────────

Step 1에서 파악한 내용을 바탕으로,
Step 2의 예시 형식을 반드시 따라 영상의 모든 행동을 처음부터 끝까지 기록한다.

추가 규칙:
- 부품 이름은 반드시 [{PART}]로 표기한다.
- 보이지 않는 손가락은 '가려짐' 또는 '화면 밖'으로 표기한다.
- 영상에서 보이는 사실만 기술한다. 추측 금지.
- 행동 하나를 여러 Timestamp로 쪼개는 것을 허용한다.
- 재가압, 재확인, 위치 재조정은 반드시 별도 항목으로 기록한다.
- 래치·클립·걸쇠 상태 변화(열림→닫힘)는 반드시 기록한다.

──────────────────────────────
Step 4. 숙련자 암묵지 분석
──────────────────────────────

Step 3 행동 로그를 바탕으로 아래 항목을 분석한다.

4-A. 정렬 기법
  - 삽입 전 어떤 시각·촉각 단서로 정렬을 인식하는가?
  - 삽입 순서: 한쪽 끝 먼저인가, 양쪽 동시인가?

4-B. 삽입 힘 프로파일
  - 삽입 시작부터 완료까지 힘의 변화를 순서대로 기술.
  - 힘 전환 트리거는 무엇인가? (소리·감각·시각)

4-C. 잠금 메커니즘 조작
  - 잠금 장치를 여는 동작과 잠기는 방식(자동/수동)을 분해하라.
  - 잠금 완료를 어떻게 인식하는가?

4-D. 손상 방지 행동
  - 손이 닿지 않아야 할 부위 회피 여부.
  - 과도한 힘 방지 방식.

4-E. 완료 인식 신호
  - 소리 / 시각 / 촉각 중 어떤 신호로 완료를 판단하는가?

──────────────────────────────
Step 5. 원자 동작 시퀀스
──────────────────────────────

전체 작업을 원자 동작 태그로 연결 (각 동작에 초 구간 병기):
동작1(XX~YYs) > 동작2(YY~ZZs) > ...

──────────────────────────────
절대 금지:
- "확인한다 / 설치한다 / 장착한다 / 작업한다" 단독 사용
- [추정]·[프레임 추론] 없이 보이지 않는 정보 단정
- 영상에 없는 행동 생성
""".strip()

PROMPTS = {
    "최종_CoT_FewShot_손가락통합": USER_PROMPT_TEMPLATE.replace("{PART}", PART),
}

# ────────────────────────────────────────────
# 실행 함수
# ────────────────────────────────────────────
def run_prompt(prompt_name, prompt_text):
    print(f"실행 중: {prompt_name}")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": SETTINGS["max_pixels"],
                    "fps": SETTINGS["fps"],
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    if video_inputs is None or len(video_inputs) == 0:
        raise RuntimeError("video_inputs 비어 있음. 영상 경로·av·ffmpeg 확인 필요.")

    if isinstance(video_kwargs.get("fps"), list):
        video_kwargs["fps"] = video_kwargs["fps"][0]

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=SETTINGS["max_new_tokens"],
        do_sample=SETTINGS["do_sample"],
        repetition_penalty=SETTINGS["repetition_penalty"],
    )

    result = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    del inputs, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# ────────────────────────────────────────────
# 저장 함수
# ────────────────────────────────────────────
def save_result(idx, prompt_name, prompt_text, output_text):
    SEP = "=" * 60
    save_path = os.path.join(output_dir, f"result_{video_name}_{prompt_name}_{ts}.txt")
    header = f"""영상 경로: {video_path}
지정 부품명: {PART}
분析 시각: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
모델: {model_name} (4bit NF4)
설정:
  max_pixels        = {SETTINGS['max_pixels']}  ({int(SETTINGS['max_pixels']**0.5)}×{int(SETTINGS['max_pixels']**0.5)})
  fps               = {SETTINGS['fps']}
  max_new_tokens    = {SETTINGS['max_new_tokens']}
  repetition_penalty= {SETTINGS['repetition_penalty']}
{SEP}
[시스템 프롬프트]
{SYSTEM_PROMPT}
{SEP}
[유저 프롬프트]
{prompt_text}
{SEP}
[모델 출력]

"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(header + output_text)
    print(f"저장 완료: {save_path}\n")

# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────
def main():
    print(f"영상: {video_path}")
    print(f"부품: {PART}")
    print(f"저장 폴더: {output_dir}\n")

    for idx, (name, text) in enumerate(PROMPTS.items(), start=1):
        try:
            result = run_prompt(name, text)
            save_result(idx, name, text, result)
            print(result[:500])
        except Exception as e:
            print(f"에러: {name} — {e}")
            traceback.print_exc()

    print("완료!")

if __name__ == "__main__":
    main()
