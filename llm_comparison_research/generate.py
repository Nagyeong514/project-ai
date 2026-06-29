"""generate.py — 암묵지 후보 생성 (계획서 5절).

골든셋(입력: VLM 결과 + 정제 Transcript)을 후보 모델 3종(Qwen2.5-72B / EXAONE-3.5-32B /
Llama-3.3-70B)에 **완전히 동일한 조건**으로 입력해 구조화(JSON) 암묵지 후보를 생성·저장한다.

동일 조건(계획서 4 통제): System/User 프롬프트, Glossary 주입, Output Schema,
temperature·seed, 4bit 양자화(2.1절)를 전 모델 고정.

서빙: vLLM OpenAI 호환 API (openai 파이썬 클라이언트로 base_url 지정 호출).
입력: config.yaml(candidate_models, generation, paths), data/golden_set/, prompts/
출력: results/generations/{model_key}.jsonl  (샘플별 후보 JSON, config.paths.generations_dir)

TODO: 로직 미구현 — 계획서 4·5절 기준으로 이후 구현.
"""
# TODO: 구현 (계획서 5절: 골든셋 로드 → 모델별 동일조건 호출 → 후보 JSON 저장)
