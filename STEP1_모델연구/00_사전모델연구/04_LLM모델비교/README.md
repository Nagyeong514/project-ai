# 암묵지 추출용 LLM 모델 비교·선정 실험

영상 기반 암묵지 후보 생성 파이프라인의 4단계("암묵지 후보 생성")에 투입할 LLM을 선정하는 비교 실험.
단일 기준 문서: [`암묵지추출_LLM_모델비교_계획서_v2.md`](암묵지추출_LLM_모델비교_계획서_v2.md). 모든 설계·통제·채점 기준은 계획서를 따른다.

> 현재 상태: **코드 구현 완료(dry-run 검증)** — generate/judge/aggregate 동작 확인. 실측을 위해 엔드포인트·골든셋 100개·용어집·루브릭 채우기만 남음(아래 체크리스트).

## 개요

- 후보 모델 3종(온프레미스, vLLM OpenAI 호환, **4bit 양자화 통일**): Qwen2.5-72B / EXAONE-3.5-32B / Llama-3.3-70B
- 골든셋 100개(층화추출, 합성 → 전문가 검수)를 **동일 조건**으로 입력 → 후보 생성 → LLM-as-judge 6항목 채점 → 가중 합산·통계로 선정

## 실행 순서

```
config.yaml  →  generate.py  →  judge.py  →  aggregate.py
 (설정)         (후보 생성)      (채점)        (집계·통계·순위)
```

1. **사전 준비**: vLLM로 후보 3종 + 심판 모델을 OpenAI 호환 엔드포인트로 서빙하고, `config.yaml`에 엔드포인트·경로·가중치·생성 파라미터를 채운다. 골든셋(`data/golden_set/`)과 프롬프트(`prompts/`)를 작성·검수한다.
2. **`python generate.py`** — 골든셋을 후보 3종에 동일 조건으로 입력해 구조화(JSON) 암묵지 후보를 `results/generations/{model_key}.jsonl`에 저장.
3. **`python judge.py`** — 심판 모델로 6항목(1~5점) 채점(제시 순서 무작위화) → `results/scores.csv`.
4. **`python aggregate.py`** — 가중 종합(계획서 3.4) · 평균/표준편차 · paired t-test(3.6) · 순위 · 시각화 → `results/summary.csv` · `results/summary.md` · `results/plot.png`.

설치: `pip install -r requirements.txt`

> dry-run으로 흐름만 점검: `python generate.py --dry-run && python judge.py --dry-run && python aggregate.py`
> (엔드포인트·데이터 없이 `results/`까지 산출물이 생성되는지 확인. aggregate는 judge가 만든 `scores.csv`를 그대로 집계한다.)

## ✅ 실제 실행 전 교체해야 할 것 (체크리스트)

코드는 dry-run으로 검증되어 있으나, **실측 전에 아래 placeholder를 실제 값으로 반드시 교체**해야 한다.

- [ ] **엔드포인트** (`config.yaml`): 후보 3종 `models[].base_url`(8001/8002/8003) + 심판 `judge.base_url`(8010)를 실제 vLLM OpenAI 호환 주소로 교체. vLLM 서빙 시 **세 모델 모두 4bit 양자화(AWQ/GPTQ)로 통일**(계획서 2.1).
- [ ] **골든셋 100개** (`data/golden_set/golden_set.jsonl`): 층화추출 100개(합성→전문가 검수, 계획서 3.2)를 채운다. 레코드 형식은 `data/golden_set/_FORMAT.md` 참조(`id` / `input.{vlm_result,transcript}` / `reference`). ※ 파일이 없으면 generate는 실호출을 막고 중단되며, `--dry-run`일 때만 합성 샘플 2개로 대체된다.
- [ ] **glossary 실제 용어** (`prompts/glossary.txt`): 현재 `lockout` 예시 1개뿐. 현장 전문가 검수 용어집으로 채운다(계획서 2.5/4, 전 모델 동일 주입).
- [ ] **채점 루브릭** (`prompts/rubric.md`): 6항목 **1·3·5점 서술 기준**이 아직 TBD. 사람 검수·심판 공용 기준을 확정한다(계획서 3.3).
- [ ] **생성 파라미터 확정** (`config.yaml` `generation`): `temperature`(현재 0.2)·`seed`·`max_tokens`·`n_candidates`를 확정 후 고정(전 모델 동일 적용, 계획서 4).
- [ ] **(선택) 한글 폰트**: 그래프 한글 라벨이 필요하면 `fonts-nanum` 등 설치(미설치 시 plot은 영문 라벨로 폴백).

## 파일 역할 (계획서 5절)

| 경로 | 역할 |
|---|---|
| `config.yaml` | 모델 엔드포인트, 평가 가중치, 샘플 수, temperature·seed, 프롬프트·스키마 경로 |
| `generate.py` | 골든셋을 세 모델에 동일 조건으로 입력 → 후보 생성·저장 |
| `judge.py` | 심판 모델로 6항목 자동 채점(순서 무작위화) → 점수 저장 |
| `aggregate.py` | 가중 합산·평균/표준편차·paired t-test·순위·시각화 |
| `prompts/system_prompt.txt` | 전 모델 동일 System Prompt (계획서 4) |
| `prompts/user_prompt_template.txt` | 후보 생성용 User Prompt 템플릿 (전 모델 동일) |
| `prompts/glossary.txt` | 도메인 용어집 — 전 모델 동일 주입 (계획서 2.5/4) |
| `prompts/output_schema.json` | 암묵지 후보 출력 JSON 스키마 (전 모델 동일) |
| `prompts/judge_prompt.txt` | LLM-as-judge 채점 프롬프트 (계획서 3.5) |
| `prompts/rubric.md` | 6항목 1·3·5점 채점 루브릭 (계획서 3.3) |
| `data/golden_set/` | 층화추출 100개 골든셋 (입력 + 모범답안) |
| `results/generations/` | generate.py 후보 산출물 (`{model_key}.jsonl`) |
| `results/scores.csv` | judge.py 채점 점수 |
| `results/summary.csv` · `summary.md` · `plot.png` | aggregate.py 집계·통계·시각화 |

## 평가 항목·가중치 (계획서 3.3 / 3.4)

`Final Score = 0.25·Faithfulness + 0.20·Accuracy + 0.20·Usefulness + 0.15·CodeSwitch + 0.10·Fluency + 0.10·Format` (1~5점)

## 의사결정 (계획서 6)

종합 1위 채택. 단 1·2위 차가 paired t-test에서 유의하지 않으면(p ≥ 0.05) 표준편차·VRAM·추론 속도로 결정.
충실도가 유독 낮은 모델은 순위와 무관하게 제외 검토(지식 DB 오염 위험).
