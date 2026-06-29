"""aggregate.py — 집계·통계·순위·시각화 (계획서 3.4 / 3.6 / 5 / 6절).

results/scores.csv를 읽어 모델 선정에 필요한 산출물을 만든다.

- 최종 점수(계획서 3.4): Final Score = 0.25·Faithfulness + 0.20·Accuracy + 0.20·Usefulness
  + 0.15·CodeSwitch + 0.10·Fluency + 0.10·Format  (1~5점 척도).
- 통계(계획서 3.6): 항목별·종합 평균/표준편차, 모델 쌍 간 paired t-test(대응표본, α=0.05).
- 순위·의사결정(계획서 6): 종합 1위 채택. 단 1·2위 차가 유의하지 않으면 표준편차·VRAM·추론속도로 결정.
  충실도 유독 낮은 모델은 순위 무관 제외 검토.
- 시각화: results/plot.png.

입력: results/scores.csv, config.yaml(weights, stats)
출력: results/summary.csv, results/summary.md, results/plot.png (config.paths)

TODO: 로직 미구현 — 계획서 3.4·3.6·5·6절 기준으로 이후 구현.
"""
# TODO: 구현 (계획서 3.4 가중합 → 3.6 통계 → 6 의사결정 → 시각화)
