# 05_LLM모델링평가_RAG답변용 (사전연구 이관본)

- **원 위치**: `project-ai/LLM_모델링_평가/` → **2026-07-09 이관** (폴더 원칙:
  사전연구/평가는 STEP 폴더·루트가 아니라 `00_사전연구/`에 둔다).
- **무엇**: STEP8 RAG 답변용 LLM 3종 비교 — Qwen3-VL-8B-Instruct / Llama-3.1-8B /
  Gemma-3-4B (전부 Ollama Q4_K_M, n5). 2026-07-07 실행.
- **결론**: **Qwen3-VL-8B-Instruct 채택** (재채점 충실도 47.4%로 공동 1위 + STEP4 VLM과
  모델 단일화 이점). 상세는 `평가보고서.md`.
- **재발 방지 기록**: Ollama 기본 태그 `qwen3-vl:8b`는 **Thinking 변형** —
  반드시 `qwen3-vl:8b-instruct` 명시(보고서 §0). Qwen2.5 계열 결과는 v1 기록만 보존.
- **재실행**: `run_benchmark_n5.sh` (내부 절대경로는 이관 후 위치로 갱신됨).
  GOLD 데이터는 `STEP7_DB/gold_records` 참조(이관과 무관, 경로 유지).
