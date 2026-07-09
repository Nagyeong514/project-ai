"""
LLM 모델링 평가 — Qwen3-VL-8B-Instruct / Llama-3.1-8B / Gemma-3-4B 비교 벤치마크.

실행 전 필수: GPU 노드(n5)에서 Ollama 서버가 떠 있어야 한다(로그인 노드엔 GPU 없음).
    srun -p RTX6000 -w n5 --gres=gpu:1 --pty bash -l
    OLLAMA_MODELS=/home/ai_user/team_a2/.ollama/models ollama serve
(주의: ~/.local/ollama/data 는 qwen만 있는 옛 경로 — 셋 다 있는 건 ~/.ollama/models)
이 스크립트 자체도 nvidia-smi로 VRAM을 재기 때문에 GPU 노드에서 실행해야 한다(로그인
노드에서 돌리면 VRAM 값이 전부 0/측정실패로 나온다).

정량 지표(자동): 평균 Latency(round-trip), VRAM 점유량(nvidia-smi 폴링 스냅샷 중 최댓값),
지시 이행률(마크다운 금지+출처표기 정규식 검증), 사실적 충실도(★ 아래 캐비어트 필독).

★ 사실적 충실도(faithfulness_pass) 캐비어트: 이건 LLM judge 가 채점한 참된 의미론적
사실검증이 아니라 키워드 겹침 기반 자동 휴리스틱이다(형태소 분석기 없이 정규식으로 자른
근사치라 조사가 붙은 단어도 부분일치로 잡히지만 오탐/누락이 있을 수 있음). RAGPerf/RGB
문서에서 말하는 "충실도"를 정식으로 채점하려면 LLM judge 나 사람이 저장된 JSON을 다시
봐야 한다 — 이 숫자는 그 전 단계의 자동 스크리닝용 근사치로만 써라.

정성 평가(어조/가독성/양자화손상/실패사례)는 이 스크립트가 채점하지 않는다. OUTPUT_FILE
JSON에 generated_answer가 시나리오별로 전부 저장되니 사람이 직접 읽고 판단할 것.
"""

import json
import time
import os
import re
import glob
import subprocess
import requests
from collections import defaultdict

# 설정
OLLAMA_API = "http://localhost:11434/api/chat"
# Qwen3-VL-8B-Instruct는 STEP4 VLM과 같은 모델 — RAG 답변 LLM까지 겸하는 구도.
# ★ STEP4는 transformers+nf4 로딩이지만 여기선 Ollama Q4_K_M GGUF — 같은 4bit여도 양자화
# 방식이 다르므로 결과 해석 시 구분할 것.
# ★ 태그 함정(2026-07-07 실측): qwen3-vl의 기본 태그 "qwen3-vl:8b"는 Instruct가 아니라
# Thinking 변형이다(digest 901cae≠0533d7). think:false를 보내도 30~60초 사고 후 답하거나
# 빈 본문을 반환했다. STEP4와 같은 모델은 명시적으로 "qwen3-vl:8b-instruct"를 써야 한다.
MODELS = ["qwen3-vl:8b-instruct", "llama3.1:8b", "gemma3:4b"]
# Ollama 태그는 셋 다 Q4_K_M GGUF 4bit 양자화 (nf4/FP16 표기는 오류였음)
MODEL_LABELS = {
    "qwen3-vl:8b-instruct": "Qwen3-VL-8B-Instruct Q4_K_M",
    "llama3.1:8b": "Llama-3.1-8B Q4_K_M",
    "gemma3:4b": "Gemma-3-4B Q4_K_M",
}
GOLD_RECORDS_PATH = "/home/ai_user/team_a2/members/안나경/project-ai/STEP7_DB/gold_records"
OUTPUT_FILE = "/home/ai_user/team_a2/members/안나경/project-ai/LLM_모델링_평가/llm_benchmark_results.json"
API_TIMEOUT = 180  # 14B 첫 로드/긴 생성이 60초를 넘길 수 있어 여유 있게(측정값은 성공 호출만 집계)
WARMUP_QUERY = "안녕하세요, 준비됐나요? 한 문장으로만 답하세요."  # 모델 로드 트리거 전용 — 집계에서 제외

SYSTEM_PROMPT = """
너는 조립 현장의 신입 작업자를 돕는 음성 작업 보조 AI다.
아래 [검색된 암묵지]만 근거로 답하고, 근거가 부족하면 솔직히 부족하다고 말하라.
규칙:
- 첫 1~2문장에 핵심 답을 먼저 말한다 (음성으로 읽힐 것을 가정, 간결한 존댓말)
- 절차가 있으면 번호 단계로 나열
- 마지막 줄에 "출처: 영상ID (시작~끝 타임스탬프)" 표기
- 마크다운 강조 기호 사용 금지, 일반 텍스트로만
"""

QUERY_MAP = {
    "tk_dell7920_channel_id_008.json": "Dell Precision 7920 워크스테이션에서 메모리 채널은 어떻게 식별하나요?",
    "tk_dell7920_ch_preread_001.json": "워크스테이션 부팅 전 사전 점검 절차를 알려주세요.",
    "tk_dell7920_bios_verify_009.json": "BIOS 설정에서 무엇을 확인해야 하나요?",
    "default": "이 장비의 올바른 정비 방법을 알려주세요."
}

# Scenario C(무관 질의) 방어 판정용 — SYSTEM_PROMPT가 요구하는 "근거 부족시 솔직히 말하라"의
# 이행 여부를 거부 표현 키워드로 근사 판정한다.
REFUSAL_MARKERS = ["모르겠", "부족", "근거가 없", "확인할 수 없", "정보가 없",
                   "답변 드리기 어렵", "알 수 없", "관련이 없", "제공된 정보"]


def get_gold_data():
    records = []
    for file_path in glob.glob(os.path.join(GOLD_RECORDS_PATH, "*.json")):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            records.append({
                "filename": os.path.basename(file_path),
                "context": data.get("knowledge", {}).get("tacit_insight", ""),
                "video_id": data.get("metadata", {}).get("source", {}).get("video_id", "unknown"),
                "timestamp": f"{data.get('metadata', {}).get('source', {}).get('clip_start', '00:00:00')}~{data.get('metadata', {}).get('source', {}).get('clip_end', '00:02:30')}"
            })
    return records


def validate_format(answer):
    if not answer:
        return False

    # 마크다운 기호 사용 여부 검증
    if re.search(r'[\*\#\`]', answer):
        return False

    # 출처 표기 양식 검증
    lines = [line.strip() for line in answer.split('\n') if line.strip()]
    if not lines or not lines[-1].startswith("출처: "):
        return False
    return True


def _keywords(text, min_len=2):
    """형태소 분석기 없이 쓰는 근사 토큰화 — 2자 이상 한글/영문/숫자 연속 조각.
    조사가 붙어도(예: '채널을') 원문 substring 검사 쪽에서 부분일치로 잡히게
    설계했다(아래 grounding_check/scenario_a 체크가 set 교집합이 아니라 in 검사를 씀)."""
    return set(w for w in re.findall(r"[가-힣A-Za-z0-9]{%d,}" % min_len, text or ""))


def scenario_a_grounding_check(answer, context_a):
    """Scenario A(정상) — 답변이 컨텍스트 핵심어를 최소 하나라도 반영했는지 확인하는
    약한 근거성 체크(사실검증 아님, '완전히 딴 얘기'만 걸러내는 최소 방어선)."""
    if not answer:
        return False
    ctx_tokens = _keywords(context_a)
    if not ctx_tokens:
        return True  # 컨텍스트 자체가 비어있으면 체크 불가 → 통과 처리
    return any(t in answer for t in ctx_tokens)


def grounding_check(answer, context_a, noise_context):
    """Scenario B(노이즈 주입) — noise_context 에만 있고 context_a 엔 없는 단어가
    답변에 스며들었는지 검사(RGB Benchmark 의 노이즈 강건성 측정을 키워드 오염 검사로 근사)."""
    noise_only = _keywords(noise_context) - _keywords(context_a)
    if not answer:
        return False, noise_only
    leaked = {w for w in noise_only if w in answer}
    return len(leaked) == 0, leaked


def refusal_check(answer):
    """Scenario C(무관 질의) — 부정거부(negative rejection) 이행 여부를 거부 표현으로 근사 판정."""
    if not answer:
        return False
    return any(marker in answer for marker in REFUSAL_MARKERS)


def get_vram_used_gib():
    """전체 GPU의 memory.used 합계(GiB). nvidia-smi 스냅샷이지 진짜 순간 peak 추적이 아니다.
    호출 시점마다 폴링해서 run_benchmark()가 그중 최댓값을 모델별로 들고 있는 방식."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        used_mib = sum(int(x) for x in out.splitlines() if x.strip())
        return used_mib / 1024
    except Exception as e:
        print(f"[WARN] nvidia-smi 조회 실패(로그인 노드에서 실행 중이거나 GPU 없음?): {e}")
        return None


# thinking 지원 모델에 think=false를 보내는 목록. 주의: 여기 넣어도 Thinking "변형"
# (예: 기본 태그 qwen3-vl:8b)은 실측상 안 먹혔다 — 변형 자체를 -instruct로 바꾸는 게 답.
# instruct 변형에 think를 보내면 400이 날 수 있어 기본은 빈 set으로 둔다.
THINK_DISABLED_MODELS = set()


def call_ollama(model, prompt):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    if model in THINK_DISABLED_MODELS:
        payload["think"] = False
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API Error for model {model}: {e}")
        return None, 0
    end_time = time.time()

    answer = response.json().get("message", {}).get("content", "")
    return answer, end_time - start_time


def unload_model(model):
    """모델 전환 전에 이전 모델을 VRAM에서 내린다(keep_alive=0). v1 실행에서 Ollama가
    앞 모델을 유지한 채 다음 모델을 올려 VRAM 측정값이 누적되는 문제가 있었음."""
    try:
        requests.post(OLLAMA_API, json={"model": model, "messages": [], "keep_alive": 0}, timeout=30)
        time.sleep(3)  # 언로드 반영 대기
    except requests.exceptions.RequestException as e:
        print(f"  [UNLOAD] 경고: {model} 언로드 실패(VRAM 측정이 누적될 수 있음): {e}")


def warmup_model(model):
    """모델 전환 시 로딩(가중치 디스크→VRAM) 시간이 첫 벤치마크 latency에 섞이는 걸
    막기 위한 워밍업 호출. 결과는 집계/저장하지 않고 로딩 트리거로만 쓴다(버그 수정 2)."""
    print(f"  [WARMUP] {model} 로딩 중...")
    ans, lat = call_ollama(model, WARMUP_QUERY)
    if ans is None:
        print(f"  [WARMUP] 경고: {model} 워밍업 실패 — 이후 요청도 실패할 가능성 있음")
    else:
        print(f"  [WARMUP] 완료 ({lat:.1f}초)")


def print_quantitative_report(results, vram_by_model):
    print("\n" + "=" * 90)
    print(" [자동 산출] LLM 벤치마크 정량 평가 보고서")
    print(" (사실적 충실도는 LLM judge 가 아닌 키워드 휴리스틱 근사치 — 캐비어트는 파일 상단 참고)")
    print("=" * 90)

    metrics = defaultdict(lambda: {
        "latency_sum": 0.0, "latency_count": 0,   # 성공한 호출만 집계(버그 수정 1)
        "format_pass": 0, "faithful_pass": 0, "total": 0,
    })

    for res in results:
        mod = res["model"]
        m = metrics[mod]
        m["total"] += 1
        if res["success"]:
            m["latency_sum"] += res["latency_seconds"]
            m["latency_count"] += 1
        if res["format_adherence_pass"]:
            m["format_pass"] += 1
        if res["faithfulness_pass"]:
            m["faithful_pass"] += 1

    header = f"{'모델명':<24} | {'평균 Latency':<12} | {'VRAM':<10} | {'지시 이행률':<16} | {'사실적 충실도(근사)'}"
    print(header)
    print("-" * len(header) * 2)

    for mod in MODELS:
        label = MODEL_LABELS.get(mod, mod)
        if mod in metrics:
            data = metrics[mod]
            avg_lat = data["latency_sum"] / data["latency_count"] if data["latency_count"] > 0 else 0
            n_fail = data["total"] - data["latency_count"]
            fmt_rate = (data["format_pass"] / data["total"]) * 100 if data["total"] > 0 else 0
            faith_rate = (data["faithful_pass"] / data["total"]) * 100 if data["total"] > 0 else 0
            vram = vram_by_model.get(mod)
            vram_str = f"{vram:.1f} GiB" if vram is not None else "측정불가"
            fail_note = f" (실패 {n_fail}건 제외)" if n_fail else ""
            print(f"{label:<24} | {avg_lat:.2f}초{fail_note:<8} | {vram_str:<10} | "
                  f"{fmt_rate:.1f}% ({data['format_pass']}/{data['total']}) | "
                  f"{faith_rate:.1f}% ({data['faithful_pass']}/{data['total']})")
        else:
            print(f"{label:<24} | 측정 실패 | 측정 실패 | 측정 실패 | 측정 실패")

    print("-" * len(header) * 2)
    print("* 정성 평가(어조/가독성/양자화손상/실패사례)는 저장된 JSON의 generated_answer를 직접 읽고 판단할 것.")
    print(f"* 원본 데이터: {OUTPUT_FILE}")
    print("=" * 90 + "\n")


def run_benchmark():
    records = get_gold_data()
    if not records:
        print("경고: 골든 레코드 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    results = []
    vram_by_model = {}

    prev_model = None
    for model in MODELS:
        print(f"\nBenchmarking model: {model}")

        if prev_model:
            unload_model(prev_model)
        prev_model = model

        warmup_model(model)
        v = get_vram_used_gib()
        vram_by_model[model] = v if v is not None else 0.0

        def _bump_vram():
            v = get_vram_used_gib()
            if v is not None:
                vram_by_model[model] = max(vram_by_model[model], v)

        for record in records:
            query = QUERY_MAP.get(record["filename"], QUERY_MAP["default"])
            metadata_str = f"영상ID: {record['video_id']}, 타임스탬프: {record['timestamp']}"

            # 시나리오 A: 정상 동작
            context_a = record["context"]
            prompt_a = f"[검색된 암묵지]: {context_a}\n[메타데이터]: {metadata_str}\n\n질문: {query}"
            ans_a, lat_a = call_ollama(model, prompt_a)
            ok_a = ans_a is not None
            results.append({
                "model": model,
                "scenario": "Scenario A",
                "query": query,
                "injected_context": context_a,
                "generated_answer": ans_a,
                "latency_seconds": round(lat_a, 2),
                "success": ok_a,
                "format_adherence_pass": validate_format(ans_a) if ok_a else False,
                "faithfulness_pass": scenario_a_grounding_check(ans_a, context_a),
            })
            _bump_vram()

            # 시나리오 B: 노이즈 주입
            noise_context = " ".join([r["context"] for r in records if r["filename"] != record["filename"]][:2])
            context_b = f"{record['context']} \n [무관한 절차]: {noise_context}"
            prompt_b = f"[검색된 암묵지]: {context_b}\n[메타데이터]: {metadata_str}\n\n질문: {query}"
            ans_b, lat_b = call_ollama(model, prompt_b)
            ok_b = ans_b is not None
            ground_ok, leaked = grounding_check(ans_b, context_a, noise_context)
            results.append({
                "model": model,
                "scenario": "Scenario B",
                "query": query,
                "injected_context": context_b,
                "generated_answer": ans_b,
                "latency_seconds": round(lat_b, 2),
                "success": ok_b,
                "format_adherence_pass": validate_format(ans_b) if ok_b else False,
                "faithfulness_pass": ground_ok,
                "leaked_noise_terms": sorted(leaked),  # 오염 근거 확인용(사람이 재검토할 때 참고)
            })
            _bump_vram()

        # 시나리오 C: 무관한 질문 방어 (모델당 1회 실행)
        query_c = "오늘 점심 메뉴 추천해줘"
        context_c = "없음"
        metadata_c = "영상ID: 없음, 타임스탬프: 없음"
        prompt_c = f"[검색된 암묵지]: {context_c}\n[메타데이터]: {metadata_c}\n\n질문: {query_c}"
        ans_c, lat_c = call_ollama(model, prompt_c)
        ok_c = ans_c is not None
        results.append({
            "model": model,
            "scenario": "Scenario C",
            "query": query_c,
            "injected_context": context_c,
            "generated_answer": ans_c,
            "latency_seconds": round(lat_c, 2),
            "success": ok_c,
            "format_adherence_pass": validate_format(ans_c) if ok_c else False,
            "faithfulness_pass": refusal_check(ans_c),
        })
        _bump_vram()

        time.sleep(3)

    # 1. 정성 평가를 위한 JSON 데이터 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Benchmark 데이터 저장 완료: {OUTPUT_FILE}")

    # 2. 정량 평가 보고서 자동 출력
    print_quantitative_report(results, vram_by_model)


if __name__ == "__main__":
    run_benchmark()
