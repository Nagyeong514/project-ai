# 06_LLM브랜드벤치_STEP5융합 (사전연구 이관본)

- **원 위치** (→ **2026-07-09 이관**):
  - `STEP5_LLM으로_암묵지_후보생성/llm_brand_bench_clip2.py` → `./llm_brand_bench_clip2.py`
  - `STEP5_.../output/llm_brand_bench/` → `./llm_brand_bench/` (frozen 입력·비교표·metrics)
  - `STEP5_.../output/step6_adapter_input/motion_run3_{qwen3,nemo}/` → `./bench_outputs/`
  - `파이프라인_통합실행/run_logs/llm_bench_clip2.{sbatch,_2294.log}` → `./`, `./run_logs/`
- **무엇**: STEP5 융합 LLM 브랜드 비교(2026-07-08~09, 잡 2294) — 현행 Qwen2.5-14B 대비
  **Qwen3-14B / Mistral-Nemo-12B** 를 CLIP2 고정 입력(frozen windows)·동일 nf4·동일
  프롬프트로 2회씩 실행. Gemma-3-12B는 gated(HF_TOKEN 없음)+Turing bf16 불가 궁합으로 제외.
- **핵심 산출물**: `llm_brand_bench/llm_compare_clip2.md` (윈도우별 3모델 나란히 비교),
  `llm_brand_bench/metrics.json`, 모델별 tacit 출력 `bench_outputs/motion_run3_*/run{1,2}/`.
- **결과 요약(잡 2294, metrics.json)**:
  - **qwen3(Qwen3-14B)**: 2회 모두 1차 통과, 융합성공 5/5, **utterance 접지율 1.0** —
    자동 지표로는 유력 후보.
  - **nemo(Mistral-Nemo-12B)**: 융합성공 1/4·1/5, 접지율 0.2~0.25 — 탈락 수준.
  - **결론은 미결(2026-07-09 기준)**: 서술 유창성·일관성 눈검사(`llm_compare_clip2.md`)와
    4클립 확장 검증이 남아 있어 현행 **Qwen2.5-14B 유지 중**. 교체 판단 시 이 폴더의
    frozen 입력·metrics를 근거로 재개한다.
- **재실행**: 스크립트/sbatch의 경로는 이관 후 위치 기준으로 갱신됨. 단 STEP5 모듈
  (`step5_preflight` 등)을 sys.path로 끌어오므로 STEP5 폴더 구조 변경 시 함께 깨진다.
