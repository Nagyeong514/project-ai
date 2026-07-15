# STEP5 초상세 보고서 (후편) — 프롬프트 투입 → 최종 tacit.json 후보 확정까지

- 작성일: 2026-07-11
- 범위: **최종 프롬프트 문자열이 모델에 들어간 직후부터 `*.tacit.json` 저장까지.** 전처리(윈도우·직렬화·메시지 조립)는 전편(`STEP5_전처리_LLM입력직전_초상세보고서.md`) 참조.
- 기준 실행: **CLIP4_재부팅및BIOS.mp4, 모션관찰 실전런(2026-07-10, 잡 2437)** — 로그 `파이프라인_통합실행/run_logs/step5_motion_obs_r2_2437.log`(CLIP4 구간 109~128행), 산출물 `output/tacit_json_motion_obs/CLIP4_재부팅및BIOS.mp4.{pass1,tacit}.json`
- 검증 방식: 산출물·로그에서 읽은 값 + 결정적 코드 경로(게이트·직렬화·토크나이저)를 그대로 재실행한 실측만 기재. **LLM 원시 출력 문자열은 이 파이프라인이 어디에도 저장하지 않으므로**(로그에도 echo 없음 — `fuse()`는 실패 시에도 원문을 남기지 않는 입력 다이어트 설계, `llm_fusion.py:383-402`), 원시 출력이 필요한 항목은 "재구성 등가물"임을 명시하고 근거를 밝혔다.

## 추적 대상 후보

**`tk_CLIP4_재부팅및BIOS_003`** — `window_ids: ["W03","W04"]` (LLM이 Pass 1에서 인접 2개 병합 예외를 실제로 사용한 후보). 전편의 추적 발화 **82.35초 "화면이 켜졌다고 다가 아니에요."** 와 추적 행동 **81.0초 "Enter 키를 누름"** 이 이 후보에 최종 귀속된다.

---

# STAGE 5. LLM 호출과 원시 출력

## 5-1. 생성 파라미터 전부

호출 코드: `llm_fusion.py:406-444` `_infer` (backend=hf_transformers 경로 `:417-444`).

| 파라미터 | 값 | 결정 위치 |
|---|---|---|
| `temperature` | 0.2 (Pass 1·2 동일. **재시도에도 불변** — 구정책 0.2→0.45→0.7은 접지 붕괴 실측으로 2026-07-08 폐지, `llm_fusion.py:311-317` 주석) | `config_motion_obs.yaml` `llm.params.temperature` / Pass 2는 코드 고정 `llm_fusion.py:795` |
| `do_sample` | **True** — 코드가 `do_sample=temp > 0`으로 계산(`llm_fusion.py:434`) | 파생값 |
| **그리디 전환 조건** | `temperature==0`일 때만 greedy. 현 config(0.2)에서는 **항상 샘플링** | `llm_fusion.py:434` |
| `top_k` / `top_p` | **20 / 0.95** — `generate()` 호출에 미지정이라 모델의 `generation_config.json` 기본값이 그대로 적용됨(temperature 0.6만 0.2로 오버라이드). 실측: HF 캐시 `models--Qwen--Qwen3-14B/.../generation_config.json` | 모델 기본값(코드·config 어디에도 명시 안 됨) |
| `max_new_tokens` | 4096 | `config_motion_obs.yaml` `llm.params.max_new_tokens` |
| `max_retries` | 5 (→ 시도 예산 총 6회) | 〃 `max_retries` |
| `min_utterance_coverage` | 0.5 | config에 없음 → 코드 기본값 `llm_fusion.py:48` |
| attention 커널 | `sdpa_kernel([EFFICIENT_ATTENTION, MATH])` 강제 (nf4+auto 조합의 math 폴백 OOM 대책) | 코드 고정 `llm_fusion.py:430-433` |
| 시드 | **미고정** — STEP5 코드 전체에 `manual_seed`/`set_seed` 없음(grep 실측) | — |
| 생성 후 정리 | `finally: del inputs, gen; empty_cache()` — 시도 1회가 GPU 잔류물을 안 남김(잡 2434 누적 OOM의 수정) | `llm_fusion.py:436-444` |

## 5-2. CLIP4 실제 호출 횟수와 입력 토큰 (잡 2437)

로그 실측 — CLIP4는 **LLM 호출 총 4회**(Pass 1 3회 + Pass 2 1회):

| 호출 | 결과 (로그 원문 발췌) | 입력 토큰(결정적 재구성) | 출력 토큰 |
|---|---|---|---|
| Pass 1 attempt 1 | `attempt 1 실패: ValueError: 접지 위장 의심 1건(window_ids [['W02']])` | **5,116** (전편과 동일 프롬프트) | 미기록 |
| Pass 1 attempt 2 | `attempt 2 실패: ValueError: 접지 위장 의심 1건(window_ids [['W02']])` | **5,331** (base + 피드백 1쌍) | 미기록 |
| Pass 1 attempt 3 | 통과 — `reasoning 접지 통계: 발화 있는 후보 5건 중 utterance 접지 4건` / `미접지: ['W02']` | **5,546** (base + 피드백 2쌍) | 미기록 (통과 초안의 **재구성 등가물 = 2,915자 / 1,173토큰**이 하한 추정치 — 5-3 참조) |
| Pass 2 attempt 1 | `응집 완료(시도 1회): 후보 8건 → 그룹 8건 (병합 그룹 0건)` | **2,686** (system 1,235 + user 1,434 + 템플릿) | 미기록 |

- 입력 토큰은 메시지 리스트가 결정적이라(피드백 문구는 코드 상수 + 로그의 에러 원문) Qwen3-14B 토크나이저로 재구성해 계산했다. 출력 토큰은 원시 출력 미보존으로 **실측 불가**(미기록이라고 명시).
- 참고 — 같은 클립의 직전 잡 이력(CLIP4 도달 여부): 잡 2433은 CLIP1에서 OOM 6연속으로, 잡 2434·2435는 CLIP2 접지 게이트 5회+6차 OOM으로 **CLIP4에 도달하기 전에 죽었고**, 잡 2436은 모델 로딩 중 SLURM 취소(`JOB 2436 ... CANCELLED ... SIGNAL Terminated`). CLIP4의 LLM 호출은 2437의 4회가 전부다.

## 5-3. 추적 후보의 "LLM이 쓴 초안" — 원시 출력의 재구성 등가물

**원시 문자열(파싱 전)은 존재하지 않는다**(저장 안 함 — 없다고 명시). 다만 1.4 설계상 LLM이 쓸 수 있는 필드는 아래가 전부이고, 이 필드들은 파싱·재조립을 거쳐도 **바이트 그대로 보존**되므로(STAGE 7·9에서 증명), pass1.json에서 역으로 "attempt 3이 출력한 초안의 후보 003 부분"을 원문 수준으로 복원할 수 있다:

```json
{
 "window_ids": ["W03", "W04"],
 "metadata": {"task": "BIOS 진입 및 정보 확인", "keywords": ["BIOS", "키보드", "정보 확인"],
              "scenario_title": "BIOS 화면 진입 및 시스템 정보 확인"},
 "knowledge": {
  "situation": "정비공이 키보드를 사용하여 BIOS 화면에 진입하고 시스템 정보를 확인한다.",
  "tacit_insight": "BIOS 화면에 진입한 후 시스템 정보를 확인한다.",
  "reasoning": "정비공이 'BIOS 화면에 진입합니다.'라고 말함 — BIOS 화면에서 시스템 정보를 확인하지 않으면 문제를 진단할 수 없다.",
  "reasoning_origin": "utterance",
  "conflict": false, "conflict_detail": null
 }
}
```

**추적 발화(82.35초)의 첫 등장 지점이 여기다**: LLM은 프롬프트에서 이 발화를 W03·W04 양쪽에서 두 번 봤지만, reasoning 인용에는 옆 발화("BIOS 화면에 진입합니다.")를 선택했다. 추적 발화 자체는 이 초안에는 **문자열로 등장하지 않고**, STAGE 7의 코드 재조립에서 diagnostic_steps·situation_source로 박힌다.

8건 전체의 초안 등가물(compact JSON)은 2,915자 / 1,173토큰 — attempt 3 출력의 크기 하한 추정치다(실제 출력에는 공백·개행이 더 있었을 수 있다).

## 5-4. 탈선 문자 게이트(`_looks_derailed`, `llm_fusion.py:233-238`)

한자(CJK)·대체문자(�)를 정규식 `[一-鿿�]`로 탐지해 발견 시 출력 폐기. **CLIP4(잡 2437)에서 발동 0회. 모션런 잡 2433~2437 전체 로그에도 "생성 탈선" 발동 기록 없음**(grep 실측). 과거 발동 실례는 균일런 CLIP1(2026-07-06, "比特rigesimal" 토큰)이며 이 게이트 신설의 계기다(`llm_fusion.py:224-225` 주석).

---

# STAGE 6. 파싱과 Pydantic 검증 — `_parse_draft` (`llm_fusion.py:486-499`)

## 6-1. 원시 문자열 → FusionDraft 변환 과정

1. `raw.strip()` 후 코드펜스(```` ``` ````)로 시작하면 백틱 제거, 첫 `{`부터 절단(`:494-496`)
2. `text.find("{")` ~ `text.rfind("}")` 구간만 `json.loads` — **JSON 앞뒤의 잡설은 전부 무시**(`:497-498`)
3. `FusionDraft.model_validate(obj)` — `DraftCandidate`(window_ids 필수) / `DraftKnowledge`(situation·tacit_insight 필수) 스키마 검증(`tacit_schema.py:217-239`)

## 6-2. 이 시점에 버려지거나 무시되는 필드 (재현 실험)

원시 출력이 미보존이라 CLIP4 실사례 대신, **금지 필드를 포함한 원시 문자열을 실제 `_parse_draft`에 투입한 재현 실험**(실행 결과 그대로):

투입한 원시 문자열(코드펜스 포함):

```
```json
{"candidates":[{"window_ids":["W03","W04"],
 "metadata":{"task":"...","id":"tk_내맘대로_001","scenario_id":"멋대로"},
 "knowledge":{"situation":"예시 상황", ..., 
  "diagnostic_steps":[{"order":1,"action":"LLM이 지어낸 스텝","evidence":"utterance","timestamp":"00:99:99"}],
  "timestamp":"00:12:34"}}]}
```
```

파싱 결과(실측):

- `knowledge.diagnostic_steps` / `knowledge.timestamp` → **소멸** (`DraftKnowledge`에 필드가 없어 extra='ignore'로 무시 — 프롬프트의 "네가 써도 버려진다" 경고가 코드로 보장되는 지점)
- `metadata.id` → **소멸** (`Metadata`에 id 필드 없음)
- ⚠️ `metadata.scenario_id: "멋대로"` → **파싱은 살아남는다** (`Metadata.scenario_id` 필드가 존재, `tacit_schema.py:66`). 단 러너 `_finalize_metadata`(`step5_runner.py:86-94`)가 저장 직전 무조건 덮어써서 최종본에는 못 남는다 — "파싱에서 안 버려지지만 코드가 덮어쓰는" 2단 방어의 실례.

## 6-3. CLIP4 실제 재시도 기록 (잡 2437)

- **Pydantic 스키마 검증 실패: 0회.** CLIP4의 실패 2회는 모두 파싱·스키마를 통과한 뒤 **접지 게이트**(STAGE 8)에서 발생했다.
- 실패 사유 원문(2회 동일, `fuse()`의 ValueError — 로그에는 140자 절단본, 아래는 코드 포맷 `llm_fusion.py:350-354`로 복원한 전문):

```
접지 위장 의심 1건(window_ids [['W02']]): reasoning_origin=utterance인데 reasoning이
그 구간 발화와 무접점 — 발화 표현을 직접 인용해 reasoning을 다시 쓰거나, 발화에 '왜'가
없으면 reasoning_origin을 model_inferred로 정직 태깅하라
```

- 재시도 시 메시지에 append된 내용 원문(`llm_fusion.py:393-402`, 코드 상수 + 위 에러의 앞 300자 — 총 423자, 결정적 재구성):

```
[assistant] (직전 시도 출력 — 검증 실패로 폐기됨)
[user] 출력 검증 실패: ValueError: 접지 위장 의심 1건(window_ids [['W02']]): reasoning_origin=utterance인데
reasoning이 그 구간 발화와 무접점 — 발화 표현을 직접 인용해 reasoning을 다시 쓰거나, 발화에 '왜'가 없으면
reasoning_origin을 model_inferred로 정직 태깅하라. 모든 후보의 knowledge에
situation/tacit_insight/reasoning/reasoning_origin/conflict를 빠짐없이 채우고, 출력 스키마(candidates[]:
window_ids/metadata/knowledge)를 정확히 지키고, 입력의 모든 window_id를 정확히 한 후보에 귀속시켜(누락·중복
금지, 병합은 인접 2개까지만) JSON만 다시 출력하라.
```

직전 출력 '전문'을 echo하지 않는 것이 핵심 설계(입력 다이어트 — 잡 2264·2283에서 echo가 prefill OOM의 직접 원인으로 실측, `llm_fusion.py:386-392` 주석). attempt 3은 이 쌍이 2개 누적된 5,546토큰 입력으로 성공했고, **W02 후보의 origin을 LLM 스스로 model_inferred로 바꿔** 왔다(강제 교정이 아니라 피드백에 의한 자기 수정 — 로그의 접지 통계가 5/5→5/5→**4/5**로 변한 것이 증거).

---

# STAGE 7. 코드 재조립 — `_rebuild_from_draft` (`llm_fusion.py:501-604`)

## 7-1. 추적 후보 003: LLM 초안 vs 재조립 후 diff

**before** = 5-3의 초안 등가물. **after** = pass1.json의 후보 003 (지면상 diagnostic_steps는 추적 대상 관련 4개만 발췌 — 전체 12스텝):

```json
{
 "id": "tk_CLIP4_재부팅및BIOS_003",          ← [코드] _assign_ids(llm_fusion.py:839-845)
 "schema_version": "1.4",                    ← [코드] 스키마 기본값
 "window_ids": ["W03", "W04"],               ← [LLM 초안 그대로] (sorted(set()) 적용, :520)
 "case": "both",                             ← [코드] 멤버 윈도우 case 집합에서 유도(:562-572)
 "metadata": { "task": …, "keywords": …, "scenario_title": …,   ← [LLM 초안 그대로]
               "scenario_id": "", "equipment": null, "source": {비어 있음} },  ← [코드가 나중에 채움(러너)]
 "knowledge": {
  "conflict": false, "conflict_detail": null,          ← [LLM 초안 그대로]
  "situation": "정비공이 키보드를 사용하여…",           ← [LLM 초안 그대로, 바이트 동일]
  "situation_source": ["00:01:22", "00:01:25"],        ← [코드] 신설(:583)
  "tacit_insight": "BIOS 화면에 진입한 후…",            ← [LLM 초안 그대로]
  "reasoning": "정비공이 'BIOS 화면에 진입합니다.'라고 말함 — …",  ← [LLM 초안 그대로]
  "reasoning_source": ["00:01:22", "00:01:25"],        ← [코드] 신설(:585)
  "reasoning_origin": "utterance",                     ← [LLM 초안 그대로(교정 조건 미해당)]
  "diagnostic_steps": [                                ← [코드] 전체 신설(:537-554)
   {"order": 3, "action": "검지와 엄지를 사용하여 키보드의 Enter 키를 누름",
    "evidence": "action_only", "source_utterance": null,
    "timestamp": "00:01:21", "utterance_timestamp": null},     ← 추적 행동 A
   {"order": 4, "action": "(발화) 화면이 켜졌다고 다가 아니에요.",
    "evidence": "utterance", "source_utterance": "화면이 켜졌다고 다가 아니에요.",
    "timestamp": "00:01:22", "utterance_timestamp": "00:01:22"},  ← 추적 발화 U
   {"order": 5, "action": "모니터 화면에 System Information 으로 표시되는 정보창이 열림", …},
   {"order": 6, "action": "(발화) BIOS 화면에 진입합니다.",
    "evidence": "utterance", "source_utterance": "BIOS 화면에 진입합니다.",
    "timestamp": "00:01:25", "utterance_timestamp": "00:01:25"}, …(총 12스텝)
  ]
 }
}
```

## 7-2. 코드가 채운 각 값의 소스 (숫자로)

- **이벤트 수집**: W03(행동 5 + 발화 2) ∪ W04(행동 5 + 발화 2) → 발화는 `(start,end)` 키 dedup(`seen_utt`, `:531-533`)으로 **4건→2건**, 행동 10건 + 발화 2건 = **12스텝**, float 시각으로 정렬(`:535`).
- **추적 행동 A**: windows.json W03의 `actions[2].timestamp=81.0` → `seconds_to_hhmmss(81.0)`(버림) = `"00:01:21"`. action 문자열은 VLM 관찰문 **원문 그대로**, `evidence="action_only"`, `source_utterance=null` (`:541-544`).
- **추적 발화 U**: windows.json의 `utterances[0].start=82.35` → **`round(82.35)=82`** → `"00:01:22"` (`:546-554`). 발화만 `round()`를 쓰는 이유는 STEP6 transcript 세그먼트 시각과 초 단위 일치를 위해서다(`:546-549` 주석). action 필드에는 `"(발화) "` 접두사가 붙고, `source_utterance`에 raw_text 원문이 박히고, `timestamp`와 `utterance_timestamp`에 같은 값이 이중 기록된다.
- **버림 vs 반올림 1초 불일치 실사례**: 옆 발화 84.67초 — 전편의 직렬화 payload에서는 버림 `"00:01:24"`로 LLM에게 보였지만, 재조립본에서는 `round(84.67)=85` → **`"00:01:25"`**. 같은 발화의 시각이 프롬프트와 최종 문서에서 1초 다르다(설계된 불일치 — 기준은 STEP6 쪽).
- **situation_source / reasoning_source 분기**(`:583-585`): `situation_source = 발화 스텝들의 timestamp 리스트(무조건)`, `reasoning_source = origin이 utterance일 때만 동일 리스트, 아니면 []`. CLIP4 실사례 3분기:
  - 후보 003 (origin=utterance): 둘 다 `["00:01:22","00:01:25"]`
  - 후보 002 (발화 있음 + model_inferred): situation_source `["00:00:22"]`, reasoning_source `[]`
  - 후보 001 (발화 없음): 둘 다 `[]`
- **case 유도**(`:562-572`): 후보 003은 멤버 W03·W04가 모두 fusion → `"both"`.

## 7-3. reasoning_origin 강제 교정 3곳의 발동 여부

| 교정 지점 | 조건 | CLIP4(2437) 발동 |
|---|---|---|
| ① 재조립: 발화 0건 후보의 utterance 신고 (`llm_fusion.py:557-560`) | 발화 스텝이 없는데 origin=utterance | **없음** — 같은 런 CLIP3에서 2회 발동(로그 원문: `[LLM-FUSE] ['W02']: 발화 없는 후보의 reasoning_origin=utterance → model_inferred 로 강제(위장 차단)`) |
| ② 접지 게이트 재시도 소진 시 (`:355-359`) | 마지막 시도에서도 무접점 utterance | **없음** — attempt 3에서 LLM이 자기 수정했으므로 소진 전 해소 |
| ③ Pass 2: utterance 멤버 없는 그룹의 utterance 태깅 (`:750-753`) | 병합 그룹 전용 | **없음** — CLIP4는 병합 그룹 0건 |

---

# STAGE 8. 게이트 통과 기록 (Pass 1 게이트 6종)

`fuse()` 성공 경로의 검증 순서(`llm_fusion.py:325-359`): ①탈선문자 → ②파싱/스키마 → (재조립) → ③귀속 3종 → ④발화 커버리지 → ⑤접지 → (⑥cross_check는 러너에서 경고만).

## 8-1. CLIP4 게이트별 전수 기록 (잡 2437)

| 게이트 | attempt 1 | attempt 2 | attempt 3 |
|---|---|---|---|
| ① 탈선 문자 (`:233-238`) | 통과 | 통과 | 통과 |
| ② 파싱+FusionDraft 스키마 (`:486-499`) | 통과 | 통과 | 통과 |
| ③ 귀속: 전수/중복·병합상한·인접·후보수 하한 (`:447-484`) | 통과 | 통과 | 통과 |
| ④ 발화 커버리지 ≥50% (`:277-292`, `:336-341`) | 통과 | 통과 | 통과 |
| ⑤ 접지 (`:240-275`, `:346-359`) | **실패(W02)** | **실패(W02)** | 통과 (LLM 자기 수정) |
| ⑥ cross_check (`tacit_schema.py:144-191`, 러너 `:63-65`) | — | — | **경고 0건** (로그에 `[CROSS-CHECK]` 행 없음, grep 실측 0) |

같은 런의 다른 클립 참고: CLIP3에서 게이트 ③이 실제 발동했다 — `attempt 1 실패: ValueError: 병합 상한 위반: 후보 하나에 윈도우 3개 ['W03', 'W04', 'W05'] — 인접 2개까지만 병합 가능`.

## 8-2. 귀속 게이트가 추적 후보에서 실제 검사한 값 (재실행 실측)

- 전수 귀속: 9개 윈도우의 귀속 카운트 = `{W01:1, W02:1, W03:1, W04:1, W05:1, W06:1, W07:1, W08:1, W09:1}` → 누락 `[]`, 중복 `[]`
- 후보 003 병합 상한: `len(window_ids)=2` ≤ 2 통과. 인접 검사: `int("W04"[1:]) - int("W03"[1:]) = 1` → 통과 (`:476-479`)
- 후보 수 하한: `ceil(9/2)=5` ≤ 8건 → 통과

## 8-3. 접지 게이트의 실제 문자열 대조 (후보 003, 재실행 실측)

`_ungrounded_utterance_claims`(`:240-275`)의 판정 과정을 실제 데이터로:

1. 후보 003의 reasoning에서 인용문 추출(정규식 `[‘'"]([^‘'"]{4,60})[’'"]`): → `['BIOS 화면에 진입합니다.']`
2. 정규화(`_norm_txt`: 공백·문장부호 제거) 후 앞 12자: `'BIOS화면에진입합니다'`
3. 대조 대상 = W03∪W04의 발화 raw_text 4건(**게이트는 윈도우 간 dedup을 안 하므로** 추적 발화·옆 발화가 각 2회): `['화면이 켜졌다고 다가 아니에요.', 'BIOS 화면에 진입합니다.', ×2]`
4. `'BIOS화면에진입합니다' in _norm_txt('BIOS 화면에 진입합니다.')` → **매칭 → grounded=True → 통과**

attempt 1·2를 죽인 W02 쪽 대조(최종본 데이터로 재현): W02 발화는 "됐습니다. 화면이 나왔습니다.", 최종 reasoning은 "화면이 켜지지 않으면 다음 작업을 수행할 수 없으므로 확인이 필요하다." — 인용문 0건, 정규화 6자 공통부분 0건. **이 reasoning이 utterance로 태깅돼 있으면 접지 실패**임을 재실행으로 확인했다(attempt 1·2의 실패가 정확히 이 유형; 단 그때의 reasoning 원문은 미보존).

## 8-4. 발화 커버리지 게이트 실측값

`input_utts` = 윈도우들의 비환각 발화 **9건(W03/W04 중복 포함 — 기준선도 dedup 안 함)**. 최종 문서의 source_utterance들과 정규화 앞 20자 포함관계 대조 → **미반영 0건** (허용 문턱: 4.5건 초과 시 실패). `경고: 발화 N건 미반영` 로그도 없음.

---

# STAGE 9. Pass 2 응집 — `_maybe_group`/`_group` (`llm_fusion.py:615-837`)

## 9-1. CLIP4 결과: 8건 → 8그룹 → 8건 (병합 0)

- 입력: `_serialize_candidates`(`:630-658`)가 Pass 1 후보 8건을 `c01~c08`로 직렬화. **발화 원문은 일부러 안 넣는다**((b)안 — LLM에게 발화 재창작 재료를 차단, `llm_grouping_prompt.py:11-13`). 추적 후보의 payload 실물:

```json
{"cand_id": "c03", "time_span": "00:01:09~00:01:41", "window_ids": ["W03", "W04"],
 "has_utterance": true,
 "situation": "정비공이 키보드를 사용하여 BIOS 화면에 진입하고 시스템 정보를 확인한다.",
 "tacit_insight": "BIOS 화면에 진입한 후 시스템 정보를 확인한다.",
 "reasoning": "정비공이 'BIOS 화면에 진입합니다.'라고 말함 — …",
 "reasoning_origin": "utterance",
 "metadata": {"task": "BIOS 진입 및 정보 확인", "keywords": ["BIOS","키보드","정보 확인"],
              "scenario_title": "BIOS 화면 진입 및 시스템 정보 확인"}}
```

- 호출 1회(입력 2,686토큰, 온도 0.2 고정 `:795`), 게이트 4종(`_check_grouping` `:671-713`: 전수 귀속 / 비인접 병합 / 과잉병합 / GroupDraft validator) 전부 통과.
- 로그: `[LLM-GROUP] 응집 완료(시도 1회): 후보 8건 → 그룹 8건 (병합 그룹 0건)` — **8건 전부 1건짜리 그룹**, LLM 출력은 `{"groups":[{"member_ids":["c01"]}, …]}` 형태(원문 미보존).

## 9-2. "1건 그룹 바이트 동일" 실제 대조 결과

`_apply_grouping`(`:738-739`)은 1건 그룹이면 Pass 1 후보 **객체를 그대로 재사용**(재조립도 안 함). 실측 대조:

- pass1.json vs tacit.json의 **knowledge 블록: 8건 전부 바이트 동일(True)** — 추적 후보 003 포함.
- 후보 전체 dict는 8건 전부 다르다(False) — 차이는 **전부 러너 `_finalize_metadata`(`step5_runner.py:78-94`)가 Pass 1 스냅샷 저장 '이후', tacit.json 저장 '이전'에 채운 자동 메타데이터뿐**이다. 추적 후보 003의 diff 전수(실측):

```json
// pass1 → tacit (달라진 필드 전부)
"metadata.scenario_id":          ""   → "CLIP4_재부팅및BIOS"
"metadata.equipment":            null → "Dell Precision 7920"      // config equipment
"metadata.source.video_id":      ""   → "CLIP4_재부팅및BIOS.mp4"
"metadata.source.clip_start":    null → "00:00:00"
"metadata.source.clip_end":      null → "00:03:49"                 // frames_meta duration 229.33 → 버림
"metadata.source.transcript_ref": null → "transcripts/CLIP4_재부팅및BIOS.mp4.json"
```

(id는 pass1 시점에 이미 `_assign_ids`로 부여돼 있어 8건 전부 동일 — 실측 True.)

## 9-3. 병합 그룹의 conflict OR 승계·origin 상속 — CLIP4엔 없음, 같은 런 CLIP3 실사례

CLIP4는 병합 0건이라 해당 없음. 같은 잡 2437의 CLIP3에서 실제 발동:

- 로그: `[LLM-GROUP] 병합: ['c03', 'c04', 'c05', 'c06']→['W03', 'W04', 'W05', 'W06', 'W07'] origin=utterance` → `후보 6건 → 그룹 3건 (병합 그룹 1건)`
- 병합 후보 `tk_CLIP3_재부팅시도 전까지_003`: window_ids 5개 합집합, **origin=utterance 상속**(멤버 4건 전원이 utterance — 교정 ③ 미발동), **conflict=false**(멤버 4건 OR: F∨F∨F∨F), 통합 tacit_insight는 LLM이 새로 썼다("RAM 삽입 전 청결과 점검을 하고, 삽입 후 전원 흐름을 확인하는 것이…").
- CLIP3의 1건 그룹 2건도 knowledge 바이트 동일 True(실측).

## 9-4. Pass 2 실패 → Pass 1 복귀 사례

**이 런(2433~2437 전체)에서 없음** — Pass 2는 도달한 모든 클립에서 1회 시도 만에 통과했다. 복귀 경로는 `_group`이 재시도 소진 시 `Pass 1 결과 그대로 출력(응집은 개선이지 필수 아님, 죽는 것보단 파편이 낫다)` 로그와 함께 원본 doc을 반환하는 코드(`llm_fusion.py:834-837`)로 존재하나 미발동.

---

# STAGE 10. 숫자로 보는 요약 (CLIP4, 잡 2437)

| 단계 | 값 |
|---|---|
| 윈도우 | 9개 |
| LLM 호출 | 4회 = Pass 1 **3회**(접지 게이트 실패 2 + 통과 1) + Pass 2 **1회** |
| 입력 토큰 | 5,116 → 5,331 → 5,546 (Pass 1) / 2,686 (Pass 2) |
| Pass 1 후보 | **8건** (병합 예외 사용 1건: 후보 003 = W03+W04) |
| 게이트 재시도 | 2회 (전부 접지 게이트, 대상 W02) — 그 외 게이트 실패 0, cross_check 경고 0 |
| Pass 2 | 그룹 8개 (병합 0) — 1회 통과 |
| **최종 후보** | **8건** → `CLIP4_재부팅및BIOS.mp4.tacit.json` (18,197바이트) |

**LLM 창작 필드 vs 코드 결정 필드** (최종 tacit.json 실측 글자 수, compact JSON 기준):

| 구분 | 필드 | 글자 비중 |
|---|---|---|
| LLM 창작(서술 원문 보존) | window_ids, metadata.task/keywords/scenario_title, knowledge.situation/tacit_insight/reasoning/reasoning_origin/conflict/conflict_detail | 추적 후보 003: **411자/2,917자 = 14.1%** / 8건 전체: **2,844자/10,324자 = 27.5%** |
| 코드 결정 | id, schema_version, case, situation_source, reasoning_source, diagnostic_steps 전체(order/action/evidence/source_utterance/timestamp/utterance_timestamp), metadata.scenario_id/equipment/source.* | 후보 003: 85.9% / 전체: 72.5% |

추적 발화 U(82.35초)의 최종 안착 지점 3곳: `diagnostic_steps[3]`(order 4 — action `"(발화) …"` + source_utterance 원문 + timestamp/utterance_timestamp `"00:01:22"`), `situation_source[0]`, `reasoning_source[0]`.

---

# 부록 A. LLM 출력이 최종 산출물에 "그대로" 살아남는 필드 전수 — 검증 유무와 잔존 리스크

| 필드 | 검증 | 잔존 리스크 |
|---|---|---|
| `knowledge.situation` | **내용 무검증** (스키마 존재 여부만) | 관찰·발화에 없는 상황 창작 가능. cross_check도 source_utterance만 대조하지 서술 필드는 안 봄 |
| `knowledge.tacit_insight` | **내용 무검증** | 동일 — STEP6 정성 판정이 유일한 방어 |
| `knowledge.reasoning` | origin=utterance일 때만 접지 게이트(인용문/6자 공통부분) | **model_inferred reasoning은 완전 무검증** — 일반론·오류 지식 유입 가능(정직 태깅만 보장됨). utterance인 경우에도 "인용문 1개 매칭"이면 통과라 인용 밖 나머지 문장은 자유 |
| `knowledge.reasoning_origin` | 접지 게이트 + 강제 교정 3곳 | utterance→model_inferred 방향 오태깅(과소 신고)은 안 잡음 — 접지 손해일 뿐 위장은 아니라는 설계 |
| `knowledge.conflict`/`conflict_detail` | cross_check 경고만(true인데 detail 없으면) — **하드 게이트 없음** | 충돌 미신고(false 오태깅)는 어떤 게이트도 못 잡음 |
| `metadata.task`/`keywords`/`scenario_title` | **무검증** | STEP7 검색 인덱스로 흘러가는 필드 — 오라벨 시 검색 품질 직결 |
| `window_ids` | 귀속 게이트 3종(전수/상한/인접) | 형식은 강검증. 단 "어느 윈도우 묶음이 의미적으로 맞는가"는 무검증(인접 2개면 뭐든 통과) |

# 부록 B. 재시도 소진 시 각 게이트의 최종 처분 전수표

| 게이트 | 소진 시 처분 | 근거 |
|---|---|---|
| 탈선 문자 | **클립 전체 실패** — `RuntimeError("LLM 융합 검증 6회 실패")` → 러너 전파, tacit.json 미생성 | `llm_fusion.py:404` |
| 파싱/스키마 | 〃 | 〃 |
| 귀속 3종 | 〃 (잡 2434·2435의 CLIP2가 이 경로로 실제 사망 — 단 사인은 접지+OOM) | 〃 |
| 발화 커버리지 | 〃 | 〃 |
| 접지 | **유일하게 안 죽는 하드 게이트** — 마지막 시도에서는 raise하지 않고 `model_inferred 강제 교정(재시도 소진)` 후 통과 | `llm_fusion.py:349-359` |
| cross_check | 항상 **경고 통과** (재시도 자체를 유발 안 함 — print만) | `step5_runner.py:63-65`, `tacit_schema.py:149-156` 주석 |
| Pass 2 게이트 4종 | **Pass 1 결과로 복귀** (버림도 죽음도 아님) | `llm_fusion.py:834-837` |
| OOM (게이트는 아니나 같은 루프) | 누적 피드백 폐기 후 원본 메시지로 리셋해 재시도, 소진 시 RuntimeError | `llm_fusion.py:376-382` |

# 부록 C. 비결정성 지점 전수 — 재실행 시 달라질 수 있는 것

1. **샘플링 자체**: `do_sample=True`(temp 0.2>0, `llm_fusion.py:434`) + **top_k=20/top_p=0.95는 모델 generation_config에서 암묵 상속**(코드 어디에도 명시 없음). 시드 미고정(grep 실측) → 같은 입력이라도 매 실행 다른 초안. → 서술 문구·window_ids 병합 선택·origin 태깅이 전부 달라질 수 있다(실증: 잡 2434는 W02+W05, 2435는 W02+W05, 2437은 CLIP4에서 W02만 접지에 걸림 — 같은 코드·같은 입력).
2. **temperature=0으로 바꿔도 남는 것**: (a) nf4 dequant + 2-GPU 분산 + sdpa 커널의 부동소수점 연산 순서 비결정성 — logit 동점 근처에서 greedy도 흔들릴 수 있음, (b) `device_map=auto`의 모듈 배치가 GPU 가용 상태에 따라 달라질 수 있음.
3. **재시도 경로 의존성**: 실패 횟수·종류가 다음 시도의 입력(피드백 누적분)을 바꾼다 — attempt 3의 성공 초안은 "5,546토큰짜리 입력"의 산물이지 원본 5,116토큰 입력의 산물이 아니다. 실패 패턴이 달라지면 최종 후보도 달라진다.
4. **OOM 여부 = GPU 환경 의존**: 동일 코드가 잡 2433(CLIP1 6연속 OOM)→2437(전 클립 성공)으로 갈린 실증. OOM은 피드백 리셋을 유발해 3번 경로도 바꾼다.
5. **Pass 2 병합 판단**: 온도 0.2 고정이어도 샘플링이라 병합/비병합이 실행마다 갈릴 수 있음(실증: CLIP4 스모크(잡 2339 계열)는 병합 0 결정적이었지만 CLIP3는 2437에서 4건 병합 — 클립별·실행별 변동).
6. **결정적인 부분(재실행해도 불변)**: 윈도우·직렬화·프롬프트 문자열(전편), 그리고 **초안이 정해진 뒤의 모든 것** — 재조립·게이트 판정·id 부여·메타데이터 채움은 순수 함수다(본 보고서의 재실행 대조가 그 증거).

# 부록 D. 재현 방법

```bash
cd /home/ai_user/team_a2/members/안나경/project-ai
파이프라인_통합실행/.venv/bin/python docs/report_packet/scripts_trace/trace_stage5to10.py
# 접지·귀속·커버리지 게이트 재실연, Pass2 입력 재구성+토큰, 재시도 메시지 재구성,
# _parse_draft 금지필드 실험, LLM/코드 글자수 비율 — 본 보고서 수치 전부 산출
```
