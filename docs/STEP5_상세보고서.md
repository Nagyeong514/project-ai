# STEP5(타임스탬프 정렬 + LLM 융합) 상세 기술 보고서

> 목적 (STEP3·STEP4 보고서와 동일)
> 1. **재디벨롭 참고 문서** — 설계 근거·파라미터·손잡이를 바로 찾는다.
> 2. **트러블슈팅 아카이브** — OOM·뭉침 탈선·접지 붕괴 등 시간을 잡아먹은 사건과 해결책.
>
> **작성 규칙(엄수):** 코드를 실제로 따라가 확인된 것만 기술. 확인 불가 항목은 본문에
> "**(확인 불가)**"로 명시하고 §8에 재수집. 모든 주장에 근거 위치(파일:라인 / config 키 / git 커밋).
> 작성 기준 시점: 2026-07-16. 기준 커밋: `45be634`.

---

## §0. 정본(canonical) 확정 — "어느 STEP5 얘기인가"

### 0-1. 디렉토리는 하나, 이름이 한 번 바뀌었다

STEP5 계열 디렉토리는 **`STEP5_LLM_암묵지_후보생성/` 하나뿐**이다. 과거 이름은
`STEP5_LLM으로_암묵지_후보생성`이었고 폴더 재구성 때 개명됐다(`step6_adapter.py:29` 주석
"폴더명 드리프트 수정('LLM으로'→'LLM')"). git 히스토리는 옛 이름 아래에 있어 현재 폴더로는
`git log`가 1건만 보이지만, `--grep`으로 보면 전 이력이 남아있다(§6에서 인용).

### 0-2. registry 키 이름과 실제 모델이 다르다 (혼동 주의)

| 표기 | 값 | 실제 |
|---|---|---|
| registry 키 | `"qwen2_5_14b"` (step5_registry.py:16) | **역사적 이름.** 클래스는 모델 중립 |
| 클래스명 | `QwenLLMFusion` | 모델 중립 어댑터 |
| config `llm.params.model_name` | **`"Qwen/Qwen3-14B"`** | **현재 실제 융합 모델** |
| config `llm.impl` | `"qwen2_5_14b"` | 위 registry 키 |

즉 파일·클래스·registry 키의 "Qwen2.5-14B" 표기는 전부 옛 이름이고, **실제로 도는 모델은
Qwen3-14B**다(2026-07-09 전면 교체, config.yaml `model_name` 주석 + git 76e1cb6). 롤백값은
`Qwen/Qwen2.5-14B-Instruct`(config 주석). 코드 docstring의 "Qwen2.5-14B"도 갱신 안 된 옛 표기다.

### 0-3. 출력 폴더가 여럿 — 정본은 `output/tacit_json`

`output/` 아래에 실험 계열 산출물이 여럿 있다. **정본(E2E 본선)은 `output/tacit_json` +
`output/aligned_windows`** 다(config.yaml `paths.step5_tacit_json_dir`/`step5_aligned_windows_dir`,
`main.py`가 이 경로를 소비).

| 폴더 | 성격 |
|---|---|
| `output/tacit_json` (4파일) | **정본 최종 산출물**(Qwen3-14B, Pass2 응집 포함) |
| `output/aligned_windows` (4파일) | **정본 정렬 산출물** |
| `output/tacit_json_motion_obs`, `aligned_windows_motion_obs` | 모션관찰 계열 실험(STEP4 output_motion 소비) |
| `output/tacit_json_run2_2268_backup_20260709`, `tacit_json_run3_2289` | 과거 런 백업 스냅샷 |
| `output/step6_adapter_input/motion_run*`, `motion_obs_run1` | STEP6로 넘긴 어댑터 입력(런별) |
| `output/mini_harness` | 미니 검증 하네스 산출물(§0-4) |
| `<video_id>.pass1.json` | 응집 전 Pass 1 스냅샷(step5_runner.py:70-75, 검증용) |

### 0-4. 코드에는 본선 외 보조 스크립트가 섞여 있다

`step5_components/`(aligner, llm_fusion)와 진입점(run_step5)이 본선이다. 나머지는 보조:
`mini_fusion_harness.py`·`mini_grouping_harness.py`(빠른 검증용 미니 하네스, git d5d8c04),
`diag_grouping_clip4.py`(진단), `step6_adapter.py`(STEP5→6 형식 변환), `step5_llm_interface.py`
(Protocol 정의). 본 보고서는 **본선(aligner + QwenLLMFusion)** 기준으로 서술하고, 어댑터·
스키마는 별도 절에서 다룬다.

**근거 위치(§0):** `step5_registry.py:15-17`, `config.yaml` llm 섹션, `step6_adapter.py:29`,
`step5_runner.py:70-75`, `main.py:44-45,78`, git `76e1cb6`.

---

## §1. 전체 구조 개관

### 1-1. 엔트리포인트 → 산출물 순서도

STEP5는 **영상 갈래(STEP4)와 음성 갈래(STEP3)가 합류하는 유일한 지점**이다(step5_runner.py:4-6).

```
[입력] STEP3: transcripts/<video_id>.json, frames_meta.json(duration/fps)
       STEP4: output/detections/<video_id>.json, output/vlm_observations/<video_id>.observations.json
        │
        ▼  Step5Runner.run(video_id)  (step5_runner.py:38-76)
   ┌─ [1/2] 정렬 (WindowAligner, 모델 없음·순수 로직) ─────────────────
   │   aligner.align(actions, transcript, detections)  (aligner.py:42-92)
   │     · 행동(VLM) timestamp를 앵커로 ±4초 윈도우 생성
   │     · 윈도우에 안 걸린 발화 → utterance_only 윈도우로 수거
   │     · 겹치는 인접 윈도우 병합(단 merge_cap_sec=20s 상한)
   │   → List[AlignedWindow]  → save_aligned_windows → output/aligned_windows/<id>.windows.json
   │
   ├─ [2/2] LLM 융합 (QwenLLMFusion, Qwen3-14B nf4) ───────────────────
   │   llm.fuse(windows, meta)  (llm_fusion.py:294-404)
   │     Pass 1: _serialize(윈도우→payload) → build_fusion_messages → _infer(생성)
   │             → _parse_draft(FusionDraft) → _rebuild_from_draft(시각·원문은 코드가 채움)
   │             → 게이트 3종(_check_window_binding / _missing_utterances / _ungrounded_...)
   │             (검증 실패 시 피드백 붙여 max_retries=5회 재시도)
   │     Pass 2: _maybe_group → _group(응집 별도 LLM 호출) → _check_grouping → _apply_grouping
   │             (실패해도 Pass 1 결과 그대로 반환 — 응집은 개선이지 필수 아님)
   │   → TacitKnowledgeDocument
   │   llm.unload()
   │
   ├─ _finalize_metadata: id/scenario_id/equipment/source를 코드가 덮어씀(LLM 추측 금지)
   ├─ cross_check: 관찰 둔갑·발화 지어냄 의심을 '경고'로 출력(하드 실패 아님)
   └─ _save + pass1.json 스냅샷
        │
[산출] output/tacit_json/<video_id>.tacit.json  ← 최종 암묵지 후보
       output/aligned_windows/<video_id>.windows.json
       output/tacit_json/<video_id>.pass1.json   (응집 전 스냅샷, 검증용)
        │
        ▼ (STEP5 밖) step6_adapter → STEP6 품질검증
```

### 1-2. 핵심 설계 원칙과 코드 구현

| 원칙 | 코드 구현 |
|---|---|
| **발화·행동 1:1 매칭 금지, ±윈도우로 구간 묶기** | `WindowAligner`(aligner.py:24-30), window_sec=4.0 |
| **LLM은 서술만, 시각·원문은 코드가 채움** | `_rebuild_from_draft`(llm_fusion.py:501-604) — diagnostic_steps/timestamp/source_utterance 전부 윈도우 데이터에서 결정적 생성. LLM은 `DraftKnowledge`(서술 5필드)+window_ids만 출력 |
| **후보 1건 = 윈도우 1개(인접 2개까지 병합)** | 프롬프트 ★★규칙(llm_fusion_prompt.py:52-59) + 하드 게이트 `_check_window_binding`(llm_fusion.py:447-484) |
| **할루시네이션 규율(위장 금지, 정직 태깅)** | 프롬프트 [절대규칙](llm_fusion_prompt.py:85-110) + `_ungrounded_utterance_claims`(llm_fusion.py:240-275) + `cross_check`(tacit_schema.py:144-191) |
| **결정적 메타데이터는 코드가 채움(LLM 추측 금지)** | `_finalize_metadata`(step5_runner.py:78-94): id/scenario_id/equipment/source |
| **스키마 = 단일 진실 공급원** | `step5_schema/tacit_schema.py`(SCHEMA_VERSION="1.4") |

### 1-3. STEP5 관련 config 키 전수 목록

#### (a) `aligner` (config.py:93, aligner.py:32-40)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `aligner.impl` | `"window"` | 레지스트리 키(현재 유일) | — |
| `.params.window_sec` | `4.0` | 행동 앵커 ±초 윈도우 폭 | 넓히면 발화-행동 더 많이 묶임(윈도우 커짐), 좁히면 파편↑ |
| `.params.merge_overlapping` | `true` | 겹치는 인접 윈도우 병합 여부 | false면 병합 안 함(윈도우↑) |
| `.params.merge_cap_sec` | `20.0` | 병합 후 전체 구간 상한 | 없애면 도미노 병합→클립 전체 1윈도우→**LLM 입력 폭발 OOM**(§6-1) |

#### (b) `llm` (config.py:94, llm_fusion.py:35-60)

| 키 | 현재 값 | 의미 | 바꾸면 |
|---|---|---|---|
| `llm.impl` | `"qwen2_5_14b"` | registry 키(역사적 이름, §0-2) | — |
| `.params.model_name` | `"Qwen/Qwen3-14B"` | **실제 융합 모델** | 롤백값 `Qwen/Qwen2.5-14B-Instruct` |
| `.params.backend` | `"hf_transformers"` | vLLM은 이 노드 불가(§6-6) | — |
| `.params.device` | `"auto"` | 2-GPU device_map | gpu:1 단독은 첫 forward OOM(config 주석) |
| `.params.dtype` | `"float16"` | nf4일 땐 **미사용**(compute dtype은 bnb가 지정, config 주석) | — |
| `.params.quantization` | `"nf4"` | 4bit. Turing fp16 compute | 서버 증설 시 해제(§7) |
| `.params.max_memory_gib` | `{0:20, 1:20}` | GPU별 상한(로드 안전망) | 배치 재조정은 nf4+auto서 불가 실측(config 주석) |
| `.params.max_new_tokens` | `4096` | 생성 상한 | 낮추면 JSON 잘림→검증 실패→재시도 소진 사망(§6-3) |
| `.params.temperature` | `0.2` | 생성 온도 | 재시도서 **올리지 않음**(접지 붕괴, §6-2) |
| `.params.grouping_enabled` | `true` | Pass 2 응집 on/off | false면 Pass 1 단독(파편 그대로) |
| `.params.max_retries` | `5` | 검증 실패 재시도(총 6회) | 낮추면 게이트 통과 복불복(config 주석) |
| `.params.min_utterance_coverage` | (미노출→**0.5**) | 입력 발화 최소 반영률 | 코드 기본값 0.5(llm_fusion.py:48) |

> `temperature`는 이중 역할: `_infer`에서 `do_sample = temp > 0`(llm_fusion.py:434)이라 **0이면
> 자동 그리디**. 현재 0.2라 샘플링. 응집(Pass 2)은 항상 temp=0.2 고정(llm_fusion.py:795).

**근거 위치(§1):** `step5_runner.py:38-94`, `aligner.py:24-92`, `llm_fusion.py:35-60,294-404,447-604`,
`tacit_schema.py`, `config.py:93-94`, `config.yaml` aligner/llm 섹션.

---

## §2. 타임스탬프 정렬 축 (WindowAligner)

### 2-1. 알고리즘 (aligner.py:42-92)

모델을 안 쓰는 순수 로직. 공통 초(float) 단위로 이미 통일된 3종을 묶는다.

1. **행동 앵커 윈도우**(aligner.py:54-74): 각 VLM 행동 `act`마다 `[act.timestamp-4, act.timestamp+4]`
   윈도우 생성. 그 구간과 겹치는 발화(`_overlaps`)·검출(`t0≤d.timestamp≤t1`)을 함께 담는다.
   0초 부근은 `t0=max(0, ...)`로 음수 클램프(CLIP4 window_start=-4.0 실측 대응, aligner.py:55-57).
2. **utterance_only 수거**(aligner.py:77-87): 어느 행동 윈도우에도 안 걸린 발화는 자기 구간
   `[u.start, u.end]`로 별도 윈도우(스펙 5.7 케이스3 — 말 먼저/무언 보존).
3. **병합**(`_merge`, aligner.py:98-119): 시간순 정렬 후 겹치는 인접 윈도우 병합. **단 병합 후
   전체 구간이 `merge_cap_sec=20`을 넘으면 병합 안 함**(도미노 병합 방지, §6-1). 발화·검출은
   `(start,end)`·`(frame_idx,cls)`로 중복 제거(aligner.py:121-135).

### 2-2. AlignedWindow와 case (intermediate.py:150-174)

`AlignedWindow` = `{window_start, window_end, actions[], utterances[], detections[]}`.
`case` 프로퍼티(intermediate.py:163-174): 행동·발화 유무로 `fusion`/`action_only`/`utterance_only`/`empty`.

- **detections는 윈도우에 담겨 aligned_windows JSON에 저장되지만 LLM에는 안 감**(STEP4 §2-4와
  동일 결론 — `_serialize`가 detections를 직렬화하지 않음, §3-2). 정렬 단계에서만 위치 힌트로 담긴다.

**근거 위치(§2):** `aligner.py:24-135`, `tacit_common/schema/intermediate.py:150-174`, config `aligner`.

---

## §3. LLM 융합 Pass 1 (초안→재조립→게이트)

### 3-1. 모델 로딩 (llm_fusion.py:85-164)

- **Qwen3-14B, 4bit NF4**(`BitsAndBytesConfig`, compute_dtype=fp16, double_quant, llm_fusion.py:125-133).
  `device_map="auto"` + `max_memory={0:20GiB,1:20GiB}`(2-GPU 분산), `attn_implementation="sdpa"`.
- **Qwen3 thinking 자동 차단**(llm_fusion.py:115-123): `model_name`에 "qwen3"가 있으면
  `apply_chat_template`를 래핑해 `enable_thinking=False` 주입 — `<think>` 블록이 JSON 파싱을
  오염시키는 것 차단. Qwen2.5 등엔 no-op.
- backend `vllm` 경로도 코드에 있으나(llm_fusion.py:88-106) 이 노드에서 사용 불가(§6-6).

### 3-2. 윈도우 → LLM 입력 직렬화 (`_serialize`, llm_fusion.py:187-221)

payload = 윈도우별 `{window_id, case, window_start, window_end, actions[], utterances[]}`.
- actions: `{timestamp, actor, action, objects_visible, (repeat_count, repeat_until)}` — STEP4 dedup이
  접은 반복은 `repeat_count>1`일 때만 필드로 전달(무손실).
- utterances: `{timestamp, raw_text, repeat_hallucination}` — **`normalized_text`는 미전송**
  (2026-07-06, raw와 거의 동일한 문장 2벌이 프롬프트만 키워 prefill OOM 유발, llm_fusion.py:212-214).
- **detections는 payload에 없다**(위치 힌트는 코드 판단용으로만 남고 LLM엔 안 감).
- compact JSON(`separators=(',',':')`, llm_fusion_prompt.py:201-203) — indent 공백이 ~900토큰
  키워 CLIP4가 prefill OOM 넘긴 실측 대응.

### 3-3. LLM은 초안(서술)만 낸다 — 시각·원문은 코드가 재조립

이것이 1.4 재설계의 핵심(SCHEMA_VERSION 주석 tacit_schema.py:19-23, git b460239/28de930).

- **LLM 출력 = `FusionDraft`**(tacit_schema.py:236-239): 후보마다 `window_ids` + `DraftKnowledge`
  (situation/tacit_insight/reasoning/reasoning_origin/conflict/conflict_detail) + metadata(task/
  keywords/scenario_title)뿐. **diagnostic_steps/timestamp/source_utterance는 LLM이 써도 버려진다**
  (프롬프트 llm_fusion_prompt.py:61-70).
- **`_rebuild_from_draft`**(llm_fusion.py:501-604)가 윈도우 데이터로 최종 문서 재조립:
  - 행동 → `evidence=action_only`, action=VLM 관찰문 원문, timestamp=행동 시각.
  - 발화 → `evidence=utterance`, source_utterance=raw_text 원문,
    timestamp=utterance_timestamp=**발화 시작 시각(round())**(step6_adapter와 초 단위 일치, llm_fusion.py:546-554).
  - `situation_source`/`reasoning_source` = 그 후보 윈도우의 발화 시각만.
  - **발화 0건인데 reasoning_origin=utterance면 model_inferred로 강제**(위장 차단, llm_fusion.py:556-560).
  - `case` 필드 = 멤버 윈도우 case 집합에서 코드 유도(both/utterance/action, llm_fusion.py:562-572).

### 3-4. 하드 게이트 3종 (실패 시 재시도)

`fuse()`(llm_fusion.py:294-404) 루프 안에서 검증. 위반 시 `ValueError`→피드백 붙여 재생성.

| 게이트 | 검사 | 위치 |
|---|---|---|
| **탈선 문자** | 한자(一-鿿)·대체문자(�) 등 나올 수 없는 문자 → 출력 폐기 | `_looks_derailed`(llm_fusion.py:233-238) |
| **윈도우 전수 귀속** | ①모든 window_id가 정확히 한 후보에(누락/중복 실패) ②병합 상한=인접 2개까지 ③후보 수 하한=ceil(n/2) | `_check_window_binding`(llm_fusion.py:447-484) |
| **발화 커버리지** | 입력 발화의 `min_utterance_coverage`(0.5) 이상이 source_utterance에 반영 | `_missing_utterances`(llm_fusion.py:277-292) |
| **접지 내용검증** | reasoning_origin=utterance인데 reasoning이 그 구간 발화와 무접점(인용 4자+ 실존 or 정규화 6자 공통) → 재시도, 소진 시 model_inferred 강제 | `_ungrounded_utterance_claims`(llm_fusion.py:240-275, 346-359) |

### 3-5. 재시도 정책 (여러 사고의 산물, §6에서 상술)

- **온도 상승 폐지**(llm_fusion.py:312-317): 구정책 0.2→0.45→0.7이 접지 붕괴(0.2서 5/5 → 0.7서 2/6)를
  유발해 폐기. 재시도 다변화는 **피드백 메시지**가 담당.
- **입력 다이어트**(llm_fusion.py:387-402): 직전 출력 전문 echo 금지 — 짧은 피드백만(prefill OOM 방지).
- **OOM 시 base 리셋**(llm_fusion.py:376-382): 누적 피드백 버리고 원본 메시지로(prefill 최소화).
- **예외 객체 즉시 해제**(llm_fusion.py:363-367): traceback이 OOM 시점 텐서를 물어 다음 시도 메모리
  깎는 것 방지 — 메시지 문자열만 보관.
- 전부 실패 시 `RuntimeError`(llm_fusion.py:404).

### 3-6. `_infer`의 attention 커널 강제 (CLIP4 OOM 근본 해결)

`_infer`(llm_fusion.py:406-445)의 hf_transformers 경로: `sdpa_kernel([EFFICIENT_ATTENTION, MATH])`로
**메모리 효율 커널 명시 강제**(llm_fusion.py:430-434). `attn_implementation="sdpa"`만으로는 이
조합(nf4+auto)에서 O(n²) math 커널로 폴백돼 6,330토큰 prefill서 GPU1 ~14GiB → OOM(실측).
`finally`에서 `del inputs, gen` + `empty_cache`로 생성 잔류물 즉시 정리(llm_fusion.py:436-444).

**근거 위치(§3):** `llm_fusion.py:85-164,187-221,294-604`, `tacit_schema.py:19-23,205-239`,
`llm_fusion_prompt.py:52-70,201-203`, git b460239/28de930/68b4dd5/d5d8c04/9aa6114.

---

## §4. Pass 2 — 지식 단위 응집 (2026-07-09)

### 4-1. 목적과 안전성

Pass 1은 후보=윈도우 결박이라 **하나의 지식이 여러 윈도우에 걸치면 파편**이 된다(run3 실측:
CLIP4 BIOS 확인 절차 6파편 → STEP6 reject, llm_grouping_prompt.py:4-8). Pass 2는 파편을 지식
단위로 다시 묶는 **별도 LLM 호출**이다. **응집은 개선이지 필수가 아님** — 어떤 실패에서도 Pass 1
결과를 그대로 반환한다(llm_fusion.py:615-628, 834-837, "죽는 것보단 파편이 낫다").

### 4-2. 입력·출력과 (b)안 (llm_fusion.py:630-658, llm_grouping_prompt.py)

- 입력(`_serialize_candidates`): Pass 1 후보를 `{cand_id, time_span, window_ids, has_utterance(bool),
  situation, tacit_insight, reasoning, reasoning_origin, metadata}`로. **발화 원문은 안 넣는다((b)안)** —
  병합 그룹 reasoning은 utterance 멤버 표현을 상속하므로 LLM에 재창작 재료를 안 줌.
- 출력(`GroupingDraft`): 그룹마다 `member_ids`. **1건 그룹은 member_ids만**(서술은 Pass 1 원문
  바이트 동일 재사용), **2건+ 그룹만 통합 서술+metadata 작성**(GroupDraft validator가 강제, tacit_schema.py:271-286).

### 4-3. 응집 게이트 (`_check_grouping`, llm_fusion.py:671-713)

①전수 귀속(모든 cand_id 정확히 한 그룹) ②비인접 병합 반려(멤버 윈도우 합집합이 순번 연속) ③과잉병합
방어(한 그룹이 클립 전체 윈도우 포함 시 실패). 위반 시 재시도(온도 0.2 고정).

### 4-4. 적용 (`_apply_grouping`, llm_fusion.py:715-780)

- 1건 그룹: Pass 1 후보 **객체 그대로**(재조립 안 함 — 바이트 동일).
- 병합 그룹: window_ids=멤버 합집합, 서술=Pass 2, diagnostic_steps 등은 `_rebuild_from_draft`
  재사용(합집합 윈도우 기준 재생성). conflict는 멤버 OR 승계.
- **그룹 origin=utterance인데 그런 멤버 없으면 model_inferred 강제**(상속 아닌 위장 차단, llm_fusion.py:750-753).
- id 전부 비우고 시간순 재부여(llm_fusion.py:775-779).

**근거 위치(§4):** `llm_fusion.py:606-837`, `llm_grouping_prompt.py`, `tacit_schema.py:242-292`,
git 76e1cb6.

---

## §5. 최종 스키마 + 교차검증 + 산출물 예시

### 5-1. 스키마 (tacit_schema.py, SCHEMA_VERSION="1.4")

`TacitKnowledgeDocument` = `{schema_version, video_id, candidates[]}`.
`TacitKnowledgeCandidate`:

| 필드 | 출처 | 의미 |
|---|---|---|
| `id` | [자동] | `tk_<stem>_NNN`(step5_runner._finalize_metadata) |
| `schema_version` | 자동 | "1.4" |
| `window_ids` | [자동/LLM 선언] | 이 후보가 온 윈도우("W01") |
| `case` | [자동] | both/utterance/action(코드 유도) |
| `metadata` | task/keywords/scenario_title=[LLM], 나머지=[자동] | |
| `knowledge` | 대부분 [LLM 서술], steps/source=[자동] | |

`Knowledge`: `conflict`, `conflict_detail`, `situation`, `situation_source[]`, `tacit_insight`,
`reasoning`, `reasoning_source[]`, `reasoning_origin`(utterance/model_inferred), `diagnostic_steps[]`.
`DiagnosticStep`: `order`, `action`, `evidence`(utterance/action_only), `source_utterance`,
`timestamp`, `utterance_timestamp`.

### 5-2. cross_check (경고, 하드 실패 아님, tacit_schema.py:144-191)

step5_runner가 fuse 후 호출(step5_runner.py:63-65). 위반 의심을 **로그로만** 출력:
evidence=utterance인데 source_utterance 없음/action_only인데 있음, source_utterance가 VLM 관찰문과
일치(관찰 둔갑)/입력 발화에 없음(지어냄), reasoning_origin=utterance인데 source 없음, conflict인데
detail 없음. **강제 실패가 아닌 이유**: 후보는 빠짐없이 살리는 게 우선, 판별은 STEP6 몫(tacit_schema.py:150-152).

### 5-3. 실제 산출물 발췌 (`output/tacit_json/CLIP2.mp4.tacit.json`)

schema_version=1.4, 후보 3건. candidate[0]:
```json
{"id": "tk_CLIP2_001", "window_ids": ["W01"], "case": "both",
 "knowledge": {"conflict": false, "reasoning_origin": "utterance",
   "situation": "정비공이 컴퓨터의 전원 버튼을 누르고, LED의 깜빡임 패턴을 관찰한다.",
   "diagnostic_steps": [
     {"order": 1, "action": "손가락으로 컴퓨터 케이스 우측 면의 원형 전원 버튼을 누르는 듯한 움직임을 한다",
      "evidence": "action_only", "source_utterance": null,
      "timestamp": "00:00:16", "utterance_timestamp": null}, ...]}}
```
- `_finalize_metadata`가 id를 `tk_CLIP2_001`로(video_id stem 기반), source/equipment를 코드값으로 덮어씀.

### 5-4. (죽은 경로) 프롬프트의 human_observation 처리

fusion 프롬프트(llm_fusion_prompt.py:151-155)는 `source="human_observation"`/`chunk="out_of_coverage"`
관찰 처리를 지시하지만, **현재 데이터 흐름상 도달 불가**다: `ActionDescription`에 `source` 필드가
없어 로드 시 소멸(STEP4 §4-3), `_serialize`도 source를 출력 안 함. 즉 이 프롬프트 블록은
**미실현(dead) 지침**이다(사람 관찰 주입 경로가 정식화되면 살아남).

**근거 위치(§5):** `tacit_schema.py:54-201`, `step5_runner.py:63-94`, `llm_fusion_prompt.py:151-155`,
실측 `output/tacit_json/CLIP2.mp4.tacit.json`.

---

## §6. 트러블슈팅 아카이브

### 6-1. LLM 입력 폭발 OOM — 무제한 도미노 병합

- **증상**: STEP5 OOM. 프롬프트 하나에 클립 전체 발화/행동이 다 들어감.
- **원인**: aligner가 겹치는 윈도우를 **상한 없이** 병합해 클립 전체(-2~194초)가 윈도우 1개로 뭉침
  (aligner.py:36-40 주석).
- **해결**: `merge_cap_sec=20.0` — 병합 후 전체 구간이 20초 넘으면 분리(aligner.py:98-119, git 원 tacit2 이식).
- **재발 시**: `aligner.merge_cap_sec` 확인, 윈도우 수/최대 폭 로그.

### 6-2. 접지 붕괴 + 재시도 온도 정책 역전

- **증상**: 재시도할수록 reasoning 접지가 무너짐(0.2서 5/5 → 0.7서 2/6), 스키마 필드 누락.
- **원인**: 구 재시도 정책이 온도를 0.2→0.45→0.7로 **올렸다**(탈선 다변화 의도). 온도↑가 접지
  붕괴·새 탈선을 유발(잡 2282/2283/2286 실측).
- **해결**: **온도 상승 폐지**(llm_fusion.py:312-317, git d5d8c04). 재시도 다변화는 게이트 피드백
  메시지가 담당(위반 수렴 5누락→병합1→1누락은 피드백 효과).
- **방어 장치**: temp=self.temperature 고정, 피드백 메시지 누적.

### 6-3. JSON 잘림 → 검증 실패 → 재시도 prefill OOM 연쇄

- **증상**: `max_new_tokens`가 작으면(500/2048) CLIP2·CLIP1서 JSON 중간 잘림 → 스키마 검증 실패 →
  재시도 프롬프트가 커져 prefill OOM으로 파이프라인 사망(config.yaml `max_new_tokens` 주석, 잡 2159).
- **원인**: LLM 파서에 잘린 JSON 복구 로직 없음(VLM과 달리). 재시도 때 직전 출력 echo로 프롬프트 비대.
- **해결**: (1)`max_new_tokens=4096`(config). (2)**입력 다이어트** — echo 없이 짧은 피드백만
  (llm_fusion.py:387-402, git 9aa6114). (3)OOM 시 base 메시지 리셋(llm_fusion.py:376-382).
- **재발 시**: `[LLM-FUSE] attempt N 실패` 로그, prefill 토큰 수, max_new_tokens.

### 6-4. 재시도 경로 OOM 2종 (텐서 잔류)

- **증상**: 접지 게이트 재시도가 쌓이며 GPU1 allocated 5.6→20.05GiB, 6차 OOM(잡 2434).
- **원인**: (a)예외 객체 `__traceback__`가 OOM 시점 activation 텐서를 물어 empty_cache로도 안 풀림
  (잡 2264). (b)생성 호출 안 텐서(inputs/gen/KV)가 예외 전파 중 트레이스백에 물림.
- **해결**: (a)`del e` + 메시지 문자열만 보관(llm_fusion.py:363-367). (b)`_infer` finally서
  `del inputs, gen`+empty_cache(llm_fusion.py:436-444). (c)시도 사이 gc+empty_cache(llm_fusion.py:371-375).

### 6-5. CLIP4 OOM — sdpa가 math 커널로 폴백

- **증상**: `attn_implementation="sdpa"`인데도 CLIP4서 GPU0 8.99GiB 요구 OOM. max_memory 배분
  (14/12, 6/16, 20/20 전부)과 무관하게 동일 재현.
- **원인**: nf4+auto 조합에서 sdpa가 O(n²) **math 커널로 폴백**, 6,330토큰 prefill서 GPU1 활성화 ~14GiB.
- **해결(사용자 예외승인)**: `_infer`서 `sdpa_kernel([EFFICIENT_ATTENTION, MATH])` 명시 강제
  (llm_fusion.py:430-434). max_memory 배분 재조정은 이 조합서 **불가**로 실측 확정(config 주석) —
  값은 로드 안전망으로만.
- **재발 시**: attention 커널 강제 살아있는지, `[LLM-DEVICE] hf_device_map` 로그.

### 6-6. vLLM 포기 + Qwen2.5→Qwen3 전환 + thinking 오염

| 사건 | 내용 | 근거 |
|---|---|---|
| **vLLM 포기** | 이 노드(Turing+sudo 없음)서 triton JIT에 python3.12-devel 필요→불가. hf_transformers+bnb nf4로 전환 | config 주석, CLAUDE.md §2, llm_fusion.py:88-106(死코드 잔존) |
| **Qwen2.5→Qwen3-14B** | Pass2 A/B(잡 2333): Qwen2.5-14B 병합 0/11(결정적 거부) vs Qwen3-14B 5/5. 전면 교체 | config 주석, git 76e1cb6 |
| **Qwen3 thinking 오염** | `<think>` 블록이 JSON 파싱 오염 | `enable_thinking=False` 주입(llm_fusion.py:115-123) |

### 6-7. 뭉침 탈선 + 시각 혼입 (1.4 재설계의 계기)

- **증상**: 최종 스키마 전체를 LLM에 출력시키니 (a)클립 전체를 후보 1건으로 뭉침(윈도우 37→후보 5)
  (b)diagnostic_steps.timestamp에 행동·발화 시각을 섞어 STEP6 0단계 전멸.
- **해결**: 후보=윈도우 결박 재설계(git b460239/28de930/68b4dd5) — LLM 재량을 서술로 축소,
  시각·원문은 코드가 재조립, 하드 게이트로 전수 귀속 강제(§3-3·§3-4).

### 6-8. step6_adapter 세그먼트 경계 겹침 → STEP6 false reject

- **증상**: STEP6 0단계 FAIL 4건(CLIP1) — 멀쩡한 후보가 "발화 지어냄"으로 오판정.
- **원인**: STT 발화가 빈틈없이 이어져 반올림 후 앞 세그먼트 end == 뒤 세그먼트 start. STEP6
  `find_transcript_segment`가 경계포함+최초매칭이라 앞 세그먼트가 먼저 잡혀 원문 불일치.
- **해결(STEP6 무수정 원칙)**: 어댑터가 end를 다음 start와 안 겹치게 깎음(step6_adapter.py:59-74, git 5cc482b).
- **재발 시**: 어댑터 변환 검증 assert(step6_adapter.py:76-94)가 건수/내용/시각 전수 대조.

### 6-9. 침묵 암묵지 diagnostic_steps 폭발 → 잘림

- **증상**: 침묵 암묵지 규칙+행동누락금지로 diagnostic_steps 18개+ 생성 → 5,044자서 잘림 →
  재시도 prefill 7,765토큰 OOM(잡 2159). (§6-3과 연결.)
- **해결**: max_new_tokens 2048→4096, 입력 다이어트(§6-3).

**근거 위치(§6):** `aligner.py:36-40,98-119`, `llm_fusion.py:88-123,312-402,430-444`, `step6_adapter.py:59-94`,
config.yaml aligner/llm 주석, git d5d8c04/9aa6114/76e1cb6/b460239/5cc482b.

---

## §7. 재디벨롭 가이드

### 7-1. 실행 커맨드

```bash
# STEP5 단독(STEP3·STEP4 산출물 있어야 함)
cd project-ai/STEP5_LLM_암묵지_후보생성
srun -p RTX6000 -w n5 --gres=gpu:2 ../파이프라인_통합실행/.venv/bin/python3 \
  run_step5.py --config ../파이프라인_통합실행/config.yaml --video-id CLIP1_정상조립과정.mp4

# 전체 4클립(STEP3→4→5) / 원샷(STEP3~7)
cd project-ai/파이프라인_통합실행
srun -p RTX6000 -w n5 --gres=gpu:2 .venv/bin/python3 run_all_clips.py
srun -p RTX6000 -w n5 --gres=gpu:2 .venv/bin/python3 main.py --run-tag myrun

# 정렬만(GPU 불필요, aligner는 순수 로직) — 별도 진입점은 없음. run_step5가 정렬+융합 일괄.
# 미니 검증(빠른 반복): mini_fusion_harness.py / mini_grouping_harness.py

# STEP5→STEP6 형식 변환(원본 무변경)
python3 step6_adapter.py --tacit-dir output/tacit_json --output output/step6_adapter_input/<tag>
```

### 7-2. 개선 후보 우선순위

| 순위 | 항목 | 근거 |
|---|---|---|
| ① | **detections를 LLM에 미전달** — payload에 위치 정보 추가하거나 코드 판단에 활용(STEP4 §2-4와 동일 미완성) | §2-2, §3-2 |
| ② | **human_observation/out_of_coverage 경로 정식화** — 프롬프트 지침은 있으나 데이터 도달 불가(source 필드 소멸) | §5-4 |
| ③ | **normalized_text 활용** — 현재 미전송. 정규화(용어 표준화) 효과가 융합에 안 감 | §3-2 |
| ④ | **registry 키·클래스·docstring의 "Qwen2.5" 옛 표기 정리** — 실제는 Qwen3-14B, 혼동 유발 | §0-2 |
| ⑤ | **서버 증설 시 양자화 해제** — 바꿀 config: `llm.quantization`(nf4→null), `max_memory_gib`, 필요시 `max_new_tokens`↑. attention 커널 강제(§6-5)는 유지 검토 | config.yaml |

### 7-3. 건드리면 위험한 불변식

1. **timestamp 원본 좌표계 + 발화/행동 시각 분리.** utterance 스텝 timestamp=발화 시작 시각,
   action 스텝=행동 시각. 섞이면 STEP6 0단계(transcript 대조) 전멸(tacit_schema.py:82-88, §6-7).
   `_rebuild_from_draft`의 시각 부여 로직·round() 규칙을 바꾸면 STEP6 어댑터와 초 단위 어긋남.
2. **LLM은 서술만, 시각·원문은 코드가 채움.** LLM에 diagnostic_steps/timestamp를 출력시키면
   시각 혼입·뭉침 재발. `_rebuild_from_draft`가 유일한 시각·원문 출처(§3-3).
3. **후보=윈도우 결박 + 전수 귀속 게이트.** `_check_window_binding`을 완화하면 뭉침 탈선 재발(§6-7).
4. **접지 위장 금지.** reasoning_origin=utterance는 발화 실접지가 전제. 발화 0건 후보의 utterance
   태깅은 강제 교정됨(llm_fusion.py:556-560). 하드 게이트로 강제하면 오히려 위장 유도라 프롬프트
   유도+소진 시 교정 조합 유지(llm_fusion_prompt.py:22).
5. **merge_cap_sec 상한 유지.** 없애면 도미노 병합→OOM(§6-1).
6. **Pass 2 실패는 Pass 1로 폴백.** 응집을 하드 의존으로 바꾸면 응집 실패가 곧 파이프라인 사망.
7. **재시도 온도 상승 금지 + 입력 다이어트.** 되돌리면 접지 붕괴·prefill OOM(§6-2·§6-3).

**근거 위치(§7):** `run_step5.py`, `main.py`, `step6_adapter.py`, `llm_fusion.py:312-402,447-604`,
`tacit_schema.py:82-88`, `aligner.py:98-119`, config.yaml.

---

## §8. "확인 불가" 항목 목록

1. **registry 키/클래스/docstring의 "Qwen2.5-14B" 표기** — 실제 모델은 Qwen3-14B(config). 코드 표기가
   갱신 안 된 것인지 의도인지는 주석으로만 판단(git 76e1cb6이 "전면 교체"라 옛 표기 잔존으로 판단).
2. **human_observation/out_of_coverage 프롬프트 지침의 원래 의도** — 데이터 경로가 없어(source 필드
   소멸) 실제로 작동한 적 있는지 코드로 확인 불가. 프롬프트 지침만 존재.
3. **`output/`의 여러 실험 계열(motion_obs, run2/run3 백업, motion_run*)을 생성한 정확한 실행 조건** —
   config 스냅샷이 런별로 남아있지 않아, 각 폴더가 어느 config/모델/커밋으로 나왔는지 산출물만으로는
   완전 확정 불가(폴더명·백업 접미사로 부분 추정).
4. **min_utterance_coverage 실제 적용값** — config 미노출이라 코드 기본값 0.5로 서술(런타임 병합 후
   최종값을 로그로 확정하진 못함).
5. **잡 번호(2159/2264/2282/2333/2434 등)의 원 로그** — config/코드 주석에 인용된 SLURM 잡 번호의
   실제 로그 파일은 이 보고서 범위에서 대조하지 않음(주석 진술 기준).
6. **cross_check 경고의 실제 발생 빈도** — 경고는 stdout 로그로만 남고 산출물에 집계되지 않아
   빈도·분포를 코드로 확인 불가.

---

*(끝) 본 보고서는 조사·설명 전용이며 코드를 수정하지 않았다.*
