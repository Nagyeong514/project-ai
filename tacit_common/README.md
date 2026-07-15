# tacit_common — STEP3·4·5 공용 계약 층

## 1. 정체

`tacit_common`은 **STEP3(전처리)·STEP4(YOLO+VLM 관찰)·STEP5(LLM 융합)가 공유하는 공용 계약(contract) 층**이다.
**실행되는 파이프라인 단계가 아니다** — 여기엔 `run_*` 진입점이 없고, 단계 사이의 데이터 규격·인터페이스·설정 로더·프리플라이트 뼈대만 있다.

한 GPU 프로세스로 STEP3→4→5가 순차 실행되든(통합 러너), 각 STEP이 단독 실행되든, **단계 사이 데이터는 항상 파일로 주고받는다.** 그 파일의 형식과 타입, 그리고 컴포넌트 교체 규격을 이 패키지가 한곳에 고정해서 "한쪽 STEP만 형식을 바꾸고 다른 쪽이 못 읽는" 사고를 막는다.

소비자는 STEP3/4/5뿐 아니라 `파이프라인_통합실행`, `00_모델연구`(실험), `docs/report_packet`(추적 스크립트)까지 걸쳐 있다 — 즉 **특정 STEP 묶음 전용이 아니라 프로젝트 범용 공용 층**이다.

## 2. 파일별 역할

| 파일 | 역할 |
|------|------|
| `config.py` | `config.yaml` 로더 — **모델 교체의 단일 스위치**. 각 컴포넌트는 `impl`(구현 선택)+`params`를 갖고, 새 모델 추가 = registry 등록 + `impl` 한 줄 교체(기존 코드 무수정). STEP3/4/5 산출물 경로를 `PROJECT_ROOT` 기준 상대경로로 보관. |
| `artifacts.py` | STEP3↔4↔5 **중간 산출물의 저장/로드를 한 곳에 고정**(프레임 메타·transcript·YOLO 검출·VLM 관찰·정렬 결과). 단독/통합 실행 어느 경로든 이 함수만 거치면 형식이 어긋나지 않는다. |
| `interfaces/` | 컴포넌트 **추상 인터페이스(Protocol)** — `sampler`(프레임 샘플링: uniform/motion/hybrid)·`stt`(STT+정제)·`detector`(YOLO)·`vlm`(행동 추출). 구체 구현은 각 STEP의 `step{N}_components/`에 둔다. |
| `schema/intermediate.py` | 단계 간 **데이터 계약 타입**(`AlignedWindow`·`BBox`·`FrameMeta`·`Transcript`·`Utterance` 등). 원칙: 타임스탬프가 뼈대 — 모든 중간 산출물은 공통 시간 단위(초, float)로 정규화. |
| `preflight_base.py` | STEP3/4/5 + 통합 러너가 공유하는 **프리플라이트 뼈대**. 각 STEP은 자기에게 필요한 조건부 모듈 검사만 넘긴다(고정 리스트로 다 검사하지 않음 — CLAUDE.md §0/§1 "무거운 모델 로드 위로는 5초 안에 죽을 검증만"). |

## 3. 설계 원칙 — 공용 층은 특정 STEP에 종속되면 안 된다

이 패키지는 어떤 STEP의 코드도 거꾸로 import하지 않는다(단방향 의존: STEP → tacit_common).

그래서 **최종 출력 스키마 `tacit_schema`(=`TacitKnowledgeDocument`)와 `LLMBackend` 인터페이스는 일부러 여기 두지 않고 STEP5 소유**로 뒀다(`STEP5_LLM으로_암묵지_후보생성/step5_schema/`, `step5_llm_interface.py`).
이유: 그것들은 LLM 융합 결과물이라 STEP5 전용 타입에 의존한다. 공용 패키지가 특정 STEP 전용 타입에 의존하면, 그 패키지를 import하는 다른 STEP까지 함께 깨진다(실제로 한 번 겪은 뒤 분리함). 공용 층에는 **모든 STEP이 공통으로 합의한 계약만** 남긴다.

## 4. ⚠️ 이동·이름변경 금지

**이 폴더를 옮기거나 이름을 바꾸지 말 것.** project-ai 최상위의 `tacit_common`이라는 위치가 코드에 박혀 있다:

- **38개 파일 / 65개 import 라인**이 `from tacit_common …` 형태로 이 패키지를 **최상위 패키지**로 참조한다(모든 소비자가 `sys.path`에 project-ai 루트를 넣고 import). 옮기면 이 import가 전부 깨진다.
- `config.py`의 `PROJECT_ROOT = Path(__file__).resolve().parent.parent`가 "tacit_common은 최상위, STEP들은 형제"를 전제한다. 위치가 바뀌면 이 계산이 어긋나 STEP3/4/5 산출물 경로 해석이 전부 틀어진다.

즉 최상위 유지가 설계 원칙(범용 공용 층 = 모든 소비자의 중립 형제)에도 맞고 참조 변경도 0이다. 옮길 이유가 없다. 정말 이동이 필요하면 위 두 지점(38파일 import + `PROJECT_ROOT`)을 동시에 고쳐야 한다.
