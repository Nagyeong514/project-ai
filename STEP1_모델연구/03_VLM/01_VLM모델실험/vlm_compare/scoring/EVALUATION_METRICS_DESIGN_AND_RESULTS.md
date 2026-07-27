# VLM 3종 비교 — 평가 지표 설계 및 결과 종합 (2026-07-05)

## 0. 배경

Qwen3-VL-8B-Instruct, InternVL3.5-8B, MiniCPM-V 4.5 세 VLM이 동일한 키프레임 + YOLO
검출 결과 + 공통 프롬프트(prompt v2, `vlm_compare/prompts/common_prompt.md`)를 입력받아
암묵지 후보를 구조화된 한국어 JSON으로 생성한 결과(4클립 × 3모델 = 978프레임, 전량 완료)를
비교한다. 사람이 만든 정답(GT)이 아직 없는 단계이므로, GT 없이 계산 가능한 지표 3개만
우선 자동화했다.

## 1. 데이터 구조 (사전 확인 완료)

**VLM 출력**: `vlm_compare/results/{model}/{video_id}_{frame_idx}.json`
- `model` ∈ {`qwen3vl`, `internvl35`, `minicpm45`}
- 최상위 필드: `video_id, frame_idx, frame_file, model_response_raw, parsed, inference_time_sec, gpu_memory_peak_gib, attn_implementation`(, `quantization` — qwen3vl 결과 파일에는 없음. 나중에 다른 두 모델 스크립트에만 추가된 메타데이터라 프롬프트 v2 필수 스키마 밖이라 문제 없음)
- `parsed` 필드: `action, tools_or_parts, notable_technique, visual_confidence, uncertain_points` (파싱 실패 시 `parsed: null`이고 `model_response_raw`에 원문 보존)

**YOLO 탐지 결과**: `260704_VLM/Detection/results/{video_id}.detections.json`
- 최상위: `video_id, n_frames, frame_width, frame_height, bbox_format, frames`
- `frames[].{frame_idx, timestamp_sec, detections}`, `detections[].{cls, conf, bbox}`
- **전체 YOLO 클래스는 4클립 통틀어 딱 7개**: `GPU, RAM, RAM_slot, eraser, hand, monitor, power_button_LED`

**프롬프트 v2 출력 스키마** (`common_prompt.md` 기준)
- 필수 필드 5개: `action, tools_or_parts(list), visual_confidence(high/medium/low), notable_technique(50자 이하 또는 null), uncertain_points(list)`
- 일관성 규칙: `uncertain_points`가 비어있지 않으면 `visual_confidence`는 medium 또는 low여야 함 (high면 위반)

**결과 저장 위치**: 원본 VLM 출력(`vlm_compare/results/`)과 섞이지 않도록 채점 스크립트/결과는 별도 폴더에 둠
- 스크립트: `vlm_compare/scoring/{schema_compliance,hallucination_rate,latency_score}.py`
- 결과: `vlm_compare/scoring/results/`

---

## 2. 지표 ① schema_compliance.py — 스키마 준수율

### 설계

필수 필드 5개 키 존재 여부 + 타입/값 검증 + 일관성 규칙을 프레임 단위로 채점. 위반이 하나도
없으면 "유효(pass)", 하나라도 있으면 "위반(fail)"으로 이분화하고 위반 유형 코드를 기록.

**위반 코드 정의**
| 코드 | 조건 |
|---|---|
| `NO_OBJECT` | VLM 출력 자체가 없음 |
| `PARSE_FAILED` | 모델 응답 JSON 파싱 자체 실패 (`parsed is None`) |
| `MISSING_FIELD_*` | 필수 필드 5개 중 키 자체가 없음 |
| `ACTION_INVALID` | `action`이 문자열이 아니거나 빈 문자열 |
| `TOOLS_NOT_LIST` | `tools_or_parts`가 list 타입이 아님 |
| `TECH_NOT_STRING_OR_NULL` | `notable_technique`이 문자열도 null도 아님 |
| `TECH_TOO_LONG` | `notable_technique`이 50자 초과 |
| `CONF_INVALID_VALUE` | `visual_confidence`가 {high,medium,low} 밖의 값 |
| `UNCERTAIN_NOT_LIST` | `uncertain_points`가 list 타입이 아님 |
| `CONF_UNCERTAIN_MISMATCH` | `uncertain_points` 비어있지 않은데 `visual_confidence == "high"` (일관성 규칙 위반) |

`notable_technique`이 `null`인 것 자체는 위반이 아님(프롬프트 v2가 "없으면 null"로 명시).

### 결과 (전체 978프레임, 모델당 326)

| 모델 | 전체 | 유효 | 위반 | 준수율 | 위반 코드별 건수 |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 326 | 321 | 5 | 98.47% | `ACTION_INVALID` 5 |
| InternVL3.5-8B | 326 | 320 | 6 | 98.16% | `TECH_TOO_LONG` 6 |
| MiniCPM-V 4.5 | 326 | 317 | 9 | 97.24% | `TECH_TOO_LONG` 2, `PARSE_FAILED` 4, `ACTION_INVALID` 2, `CONF_UNCERTAIN_MISMATCH` 1 |

**해석**: 세 모델 다 97~98.5%로 준수율 자체는 비슷하지만, 위반 *유형*은 모델마다 다르다.
- Qwen3-VL: 애매한 장면을 `action: null`로 처리하다 스키마 위반(전부 `ACTION_INVALID`)
- InternVL3.5: `notable_technique` 서술이 장황해서 50자 초과(전부 `TECH_TOO_LONG`)
- MiniCPM-V 4.5: 유일하게 `PARSE_FAILED`(모델 응답 JSON 파싱 실패, 4건) 발생 — 아래 3절에서 원인 상세 분석

### PARSE_FAILED 4건 근본원인 분석 (MiniCPM-V 4.5)

| video_id | frame_idx | 앞뒤 프레임(±2) 상태 | 원인 |
|---|---|---|---|
| CLIP1_normal_assembly | 83 | 전부 정상 | 반복생성 루프("케이스" 반복) → 512토큰 상한 도달 추정 |
| CLIP1_normal_assembly | 94 | 전부 정상 | 반복생성 루프("NASbayes" 반복) → 512토큰 상한 도달 추정 |
| CLIP2_0704 | 0 | 전부 정상 | JSON 문법 깨짐 (action 값 안에 이스케이프 안 된 내부 큰따옴표) |
| CLIP4_reboot_bios | 43 | 전부 정상 | 반복생성 루프("BIOS" 반복) → 512토큰 상한 도달 추정 |

- 4건 모두 3개 클립에 흩어져 있고 서로 인접하지 않음 → "특정 구간이 어려운 장면"이 아니라 **산발적("가끔 튀는") 문제**
- 반복루프 3건의 `inference_time_sec`이 136~138초로, 정상 322개 평균(25.55초) 대비 **5.3~5.4배** — 512토큰 한도까지 다 채운 것으로 추정되는 강한 정황
- JSON 문법 깨짐 1건은 44.57초로 정상 범위(1.74배) — 반복루프와는 무관한 별개의 실패 유형
- GPU 메모리(7.88~8.03GiB)는 정상 범위 → OOM 아님. 타임아웃도 없었음(job에 시간제한 없었고 정상 종료)

### MiniCPM 정상 322개 vs 다른 모델 지연시간 비교 (참고)

| 구분 | 값 |
|---|---|
| MiniCPM 정상 322개 평균 | 25.55초 |
| MiniCPM 정상 322개 중앙값 | 24.81초 |
| MiniCPM 정상 322개 표준편차 | 5.59초 |
| MiniCPM 정상 322개 최소/최대 | 16.18초 / 54.79초 |
| Qwen3-VL 326개 평균 | 21.04초 |
| InternVL3.5 326개 평균 | 14.51초 |

---

## 3. 지표 ② hallucination_rate.py — 환각률

### 설계 배경 및 확정 과정 (사용자 승인 사항 전체)

**알고리즘 기본 구조** (지시받은 대로 구현)
1. `action`, `tools_or_parts`, `notable_technique` **세 필드를 모두 스캔** (tools_or_parts만 보면 서술형 필드로 새는 환각을 놓치기 때문)
2. YOLO 클래스 ↔ 한국어/영어 표현 매핑을 위한 **폐쇄형 동의어 사전**을 실제 데이터 기반으로 구축
3. 헷지(추측)·부정 표현과 함께 언급된 경우는 환각이 아니라 `boundary_case`로 별도 기록
4. 그 프레임의 YOLO 검출 결과에 없는 클래스가 헷지/부정 없이 언급된 경우만 환각으로 카운트
5. 모델별 "프레임 단위 환각률"과 "언급 단위 환각률(건/프레임)" 모두 계산
6. 결과 파일에 필수 문구 포함(아래 5절 참고)

**동의어 사전 구축 — 실제 데이터 기반 확인 과정**

3모델 합산 `tools_or_parts` 전체 표현 빈도를 직접 뽑아서 확인(주요 항목):
```
232 RAM     125 손      116 모니터    104 키보드    100 RAM 모듈   100 RAM 슬롯
 80 GPU      73 monitor  61 hand      57 keyboard    52 컴퓨터 케이스
 43 메인보드  40 케이블   38 eraser    35 메모리 모듈  24 컵    24 전원 버튼
 18 CPU 케이스 17 컴퓨터 본체 17 motherboard 13 마우스 ...
```

이 중 **키보드/마우스/케이스류/컵/메인보드/케이블류/테이블 등은 YOLO 7개 클래스에
대응어 자체가 없음** — YOLO가 애초에 이런 대상을 탐지하도록 학습되지 않았기 때문. 이걸
문자 그대로 "환각"으로 잡으면 지표가 왜곡된다는 판단 하에 **사용자 확인 후 결정**:

> **확정: YOLO 7개 클래스에 대응어가 없는 언급은 환각 판정 대상에서 완전히 제외**한다.
> 근거: 이 지표의 목적은 "YOLO가 본 것과 VLM이 말한 것의 불일치"를 보는 것이므로,
> YOLO 어휘에 아예 없는 대상까지 불일치로 잡으면 지표가 왜곡된다.
> 단, 완전히 버리지 않고 `out_of_vocabulary_mentions.csv`에 모델별로 로그를 남겨
> 나중에 "YOLO 클래스를 늘려야 하나?" 판단할 참고자료로 사용한다.

**최종 승인된 동의어 사전** (7개 YOLO 클래스)

| YOLO 클래스 | 매핑된 표현 |
|---|---|
| GPU | NVIDIA GeForce RTX 그래픽카드, GPU(GeForce RTX), 그래픽카드, 그래픽 카드, GPU, gpu |
| RAM | 메모리 모듈, RAM 모듈, RAM |
| RAM_slot | RAM 슬롯, RAM_slot, RAM Slot, 램슬롯 |
| hand | 손, hand |
| monitor | 컴퓨터 모니터, 모니터, monitor |
| eraser | 지우개, eraser |
| power_button_LED | 전원 버튼 LED, 전원 버튼, 파워 버튼, 전원 LED |

**사전에서 의도적으로 제외한 2가지 (실측 확인 후 결정)**

1. **"메모리"/"모듈" 단독 표기 → RAM에 자동 매핑하지 않음.** "메모리 모듈"/"RAM 모듈"처럼
   다른 단어와 함께 나온 경우는 그대로 RAM으로 매핑하되, "메모리"나 "모듈"만 단독으로 나온
   경우는 일반명사로 쓰였을 가능성(과탐지 위험)이 있어 `unresolved_ambiguous`로 별도 분류,
   `unresolved_mentions.csv`에 문장 전체와 함께 기록하고 환각률 계산에서 제외.
2. **monitor의 "화면" → 제외.** 실제 등장 110건을 전수 확인한 결과, 대부분이
   "BIOS 설정 화면을 확인", "화면에 ~가 표시됨"처럼 **디스플레이 내용/상태**를 가리키는
   용법이었고(예: "컴퓨터 모니터를 통해 BIOS 설정 화면을 확인" — "모니터"와 "화면"이
   한 문장에 별개 개념으로 공존), 물리적 모니터 장치 자체를 가리키는 명확한 용례는
   사실상 없었음. 오탐지 방지를 위해 사전에서 제외.

**헷지/부정 표현 목록** (승인된 초안, 완전한 리스트를 처음부터 만들려 하지 않고 실측하며 보강하는 방식으로 진행)
```
헷지: "일 수도", "인 것 같", "로 보임", "로 추정", "불확실", "확인 불가", "판단 불가", "명확하지 않"
부정: "없음", "없다", "아님", "아니다", "보이지 않"
```
매칭된 표현의 앞뒤 ±20자 윈도우 안에 위 패턴이 있으면 `boundary_case`로 분류.

**환각 판정 로직 요약**
1. 3개 필드 텍스트에서 동의어 사전으로 **긴 문자열부터 greedy 매칭**(예: "RAM 슬롯"을 먼저 매칭해 소비하고, 남은 구간에서만 "RAM" 단독 매칭 — 이중 카운트 방지)
2. 매칭된 표현이 해당 프레임의 YOLO 검출 클래스 목록에 **있으면** → 집계 대상 아님(정상)
3. **없으면** → 앞뒤 문맥에 헷지/부정 있으면 `boundary_case`, 없으면 `hallucination`
4. 사전 매칭 후 남은 구간에서 "메모리"/"모듈" 단독 표기 탐색 → `unresolved_ambiguous`
5. `tools_or_parts` 리스트 항목 중 사전에도 unresolved 패턴에도 안 걸린 것 → `out_of_vocabulary`
6. 3개 필드 원문 텍스트에서 별도로 **인코딩 손상 여부** 검사(아래 참고) → `possible_encoding_corruption`

**인코딩 손상 탐지 — 개발 중 실제 버그 발견 및 수정**

최초 버전은 "한글↔라틴 문자가 공백 없이 붙어있으면 손상"으로 양방향 검사했으나, CLIP1
테스트에서 "RAM을", "GPU를" 같은 **완전히 정상적인 한국어 조사 결합**(라틴 약어에 조사가
바로 붙는 표준 표기법)까지 60건이나 오탐지되는 걸 발견. 라틴→한글 방향 탐지를 제거하고
**한글→라틴 방향(하이픈 허용)만** 남기도록 수정 → 오탐 0건, 진짜 손상 사례만 12건(CLIP1
기준) 남음. 최종 정규식:
```
CJK 비한국어: [぀-ヿ一-鿿�]   (히라가나/가타카나/한자/치환문자)
한글-라틴 결합: [가-힣]-?[a-zA-Z]{2,}   (한글 뒤에 라틴 2자 이상, 하이픈 허용)
```
이 휴리스틱은 완전 검출을 보장하지 않는 "참고용 신호"임을 전제로 함.

### 결과 (전체 978프레임)

| 모델 | 프레임단위 환각률 | 언급단위 환각률(건/프레임) | 환각 언급 총건수 | 환각 있는 프레임 |
|---|---|---|---|---|
| Qwen3-VL-8B | 19.63% | 0.331 | 108 | 64/326 |
| InternVL3.5-8B | 43.56% | 0.748 | 244 | 142/326 |
| MiniCPM-V 4.5 | 48.77% | 0.982 | 320 | 159/326 |

**모델별 환각 클래스 분포** (어떤 YOLO 클래스에서 환각이 몰리는지)
| 모델 | GPU | RAM | RAM_slot | hand | monitor |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 11 | 57 | 6 | 21 | 13 |
| InternVL3.5-8B | 2 | 102 | 129 | 0 | 11 |
| MiniCPM-V 4.5 | 0 | 81 | 215 | 12 | 12 |

→ **환각의 절대다수가 RAM/RAM_slot에 집중**됨(특히 InternVL3.5·MiniCPM-V). RAM 삽입 작업
특성상 손에 가려져 YOLO가 그 프레임에서 RAM/RAM_slot을 놓치는 경우가 많을 것으로 추정 —
정확히 캐비어트 문구("YOLO 미검출과 VLM 환각을 구분 못함")가 경고하는 상황과 일치.

**부가 로그 건수 (모델별)**

| 모델 | unresolved_ambiguous | out_of_vocabulary | possible_encoding_corruption |
|---|---|---|---|
| Qwen3-VL-8B | 24 | 187 | 0 |
| InternVL3.5-8B | 142 | 214 | 19 |
| MiniCPM-V 4.5 | 150 | 150 | 32 |

- `out_of_vocabulary` 상위 항목(전체): 키보드 104, keyboard 57, 컴퓨터 케이스 52, 메인보드 43, 케이블 40, 컵 24, CPU 케이스 18, 컴퓨터 본체 17, motherboard 17, 마우스 13 등
- `possible_encoding_corruption` 대표 사례: internvl35의 "전원ユニット"(일본어 혼입), "스crewdriver"(오타/깨짐); minicpm45의 "電源供給ケーブルを接続"(문장 전체가 일본어), "電源開關区域检查"(문장 전체가 중국어) — **문장 전체가 다른 언어로 나온 사례**는 minicpm45에서만 관찰됨
- `unresolved_ambiguous`는 minicpm45(150)·internvl35(142)가 qwen3vl(24)보다 훨씬 많음 — "메모리"/"모듈"을 단독으로 쓰는 경향이 강함을 시사

### 필수 캐비어트 (결과 파일에 포함됨)

> 이 지표는 YOLO의 미검출(false negative)과 VLM의 실제 환각을 구분하지 못하며,
> YOLO 검출 결과를 정답이 아닌 약한 기준(weak reference)으로 사용함

---

## 4. 지표 ③ latency_score.py — 지연시간

### 설계
`inference_time_sec` 필드가 이미 모든 VLM 출력 JSON에 존재함을 확인 → 별도 SLURM 로그
조회 불필요. 모델별, 모델×클립별로 mean/median/p95(최근접 순위 방식)/min/max 계산.

### 결과 (모델별, 전체 326프레임)

| 모델 | mean | median | p95 | min | max |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 21.04s | 20.18s | 28.58s | 15.73s | 37.6s |
| InternVL3.5-8B | 14.51s | 14.36s | 18.14s | 10.73s | 20.58s |
| MiniCPM-V 4.5 | 26.64s | 24.91s | 38.47s | 16.18s | 138.17s |

### 모델×클립별 상세

| 모델 | 클립 | n | mean | median | p95 | min | max |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | CLIP1_normal_assembly | 99 | 21.95 | 20.94 | 29.56 | 15.73 | 37.6 |
| Qwen3-VL-8B | CLIP2_0704 | 57 | 21.4 | 21.37 | 27.88 | 16.81 | 28.73 |
| Qwen3-VL-8B | CLIP3_before_reboot_0704 | 55 | 19.73 | 19.54 | 21.63 | 17.94 | 24.08 |
| Qwen3-VL-8B | CLIP4_reboot_bios | 115 | 20.7 | 19.81 | 27.02 | 16.08 | 31.97 |
| InternVL3.5-8B | CLIP1_normal_assembly | 99 | 14.76 | 14.65 | 17.98 | 11.19 | 20.2 |
| InternVL3.5-8B | CLIP2_0704 | 57 | 15.18 | 15.27 | 19.53 | 11.3 | 20.58 |
| InternVL3.5-8B | CLIP3_before_reboot_0704 | 55 | 14.86 | 15.28 | 18.33 | 11.71 | 19.14 |
| InternVL3.5-8B | CLIP4_reboot_bios | 115 | 13.79 | 13.72 | 17.43 | 10.73 | 18.4 |
| MiniCPM-V 4.5 | CLIP1_normal_assembly | 99 | 29.04 | 27.04 | 38.85 | 17.89 | 138.17 |
| MiniCPM-V 4.5 | CLIP2_0704 | 57 | 26.81 | 26.17 | 40.96 | 18.28 | 44.57 |
| MiniCPM-V 4.5 | CLIP3_before_reboot_0704 | 55 | 23.95 | 22.47 | 30.6 | 18.74 | 42.2 |
| MiniCPM-V 4.5 | CLIP4_reboot_bios | 115 | 25.78 | 23.25 | 36.74 | 16.18 | 136.5 |

특정 클립에 지연시간이 몰리는 패턴은 뚜렷하지 않음(반복생성 루프로 인한 MiniCPM의 극단값
2건이 CLIP1·CLIP4의 max를 끌어올린 것 외에는 클립 간 큰 차이 없음).

---

## 5. 종합 비교 (세 지표 나란히)

> 이 지표는 YOLO의 미검출(false negative)과 VLM의 실제 환각을 구분하지 못하며, YOLO 검출 결과를 정답이 아닌 약한 기준(weak reference)으로 사용함

| 지표 | Qwen3-VL-8B | InternVL3.5-8B | MiniCPM-V 4.5 |
|---|---|---|---|
| **스키마 준수율** | 98.47% (321/326) | 98.16% (320/326) | 97.24% (317/326) |
| **환각률(프레임 단위)** | 19.63% (64/326) | 43.56% (142/326) | 48.77% (159/326) |
| **환각률(언급 단위, 건/프레임)** | 0.331 | 0.748 | 0.982 |
| **out-of-vocabulary 언급 건수** | 187 | 214 | 150 |
| **평균 추론시간** | 21.04s | 14.51s | 26.64s |
| **추론시간 p95** | 28.58s | 18.14s | 38.47s |
| **추론시간 최대** | 37.6s | 20.58s | 138.17s |

**요약 해석**
- **스키마 준수**: 세 모델 다 97~98.5%로 비슷하지만 위반 유형이 다름(Qwen=action null, InternVL=서술 과다, MiniCPM=파싱실패+일관성위반까지 다양)
- **환각률**: Qwen3-VL이 가장 낮고(19.6%), MiniCPM-V가 가장 높음(48.8%) — 단, 절대다수가 RAM/RAM_slot에 몰려있어 YOLO의 손 가림 미검출 영향이 클 것으로 추정(GT 없이는 확정 불가)
- **속도**: InternVL3.5가 가장 빠르고(14.5s) 안정적(p95-mean 격차 작음), MiniCPM-V가 가장 느리고 변동성도 큼(반복생성 루프로 인한 4건의 극단값 포함)
- **출력 품질**: MiniCPM-V만 유일하게 "문장 전체가 다른 언어(중국어/일본어)로 출력"되는 심각한 이상현상 관찰됨(32건)

---

## 6. 파일 색인

**스크립트**: `/home/ai_user/team_a2/members/오유빈/vlm_compare/scoring/`
- `schema_compliance.py`, `hallucination_rate.py`, `latency_score.py`

**결과**: `/home/ai_user/team_a2/members/오유빈/vlm_compare/scoring/results/`
- `schema_compliance_summary.json`, `schema_compliance_detail.csv`
- `latency_summary.json`, `latency_detail.csv`
- `hallucination_summary.json`, `hallucination_detail.csv`
- `hallucinations.csv` (환각으로 카운트된 언급 전체 목록)
- `boundary_cases.csv` (헷지/부정으로 제외된 목록 — 전체 데이터셋에서 0건)
- `unresolved_mentions.csv` ("메모리"/"모듈" 단독 표기)
- `out_of_vocabulary_mentions.csv` (YOLO 7클래스 밖 언급)
- `possible_encoding_corruption.csv` (인코딩/언어 혼입 의심 사례)
- `combined_summary.md` (5절과 동일한 종합 비교표)
- `*_test_clip1.*` (CLIP1만 대상으로 한 사전 테스트 결과, 대조용으로 보존)

**원본 VLM 출력**: `/home/ai_user/team_a2/members/오유빈/vlm_compare/results/{qwen3vl,internvl35,minicpm45}/`
**원본 YOLO 탐지 결과**: `/home/ai_user/team_a2/members/오유빈/260704_VLM/Detection/results/`
