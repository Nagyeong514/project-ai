"""judge.py — LLM-as-judge 자동 채점 (계획서 3.5 / 5절).

후보와 다른 계열의 심판 모델로 6항목을 1~5점 루브릭(prompts/rubric.md)으로 채점한다.
6항목: Faithfulness(충실도) / Accuracy(정확성) / Usefulness(유용성) /
       CodeSwitch(한·영 혼용 이해) / Fluency(자연스러움) / Format(형식 준수).

통제(계획서 3.5):
- reference-free 채점(정답 일치가 아니라 근거·타당성 기준).
- 제시 순서 무작위화(position bias 완화), 여건 시 다중 심판 앙상블로 확장.
- Format은 JSON 파싱 성공률로 자동 집계.

입력: results/generations/*.jsonl, config.yaml(judge, weights), prompts/judge_prompt.txt, prompts/rubric.md
출력: results/scores.csv  (샘플 × 모델 × 6항목 점수, config.paths.scores_csv)

TODO: 로직 미구현 — 계획서 3.5·5절 기준으로 이후 구현.
"""
# TODO: 구현 (계획서 3.5: 심판 호출 → 6항목 점수 파싱 → 저장)
