# project-ai — 작업 지침 (모든 CLI/에이전트 공통)

이 프로젝트에서 코드를 짜거나 실행하기 전에, 아래를 **반드시** 먼저 확인할 것.
사람이 매번 안 챙겨도 되게 하는 게 목적이므로, 요청에 없어도 스스로 확인한다.

## 0. 필독 문서

- **`docs/실행전_방어_체크리스트.md`** — 무거운 모델/GPU 잡을 실행하기 전 반드시 지킬 순서.
  이 문서를 안 읽고 GPU 작업을 진행하지 말 것. (2026-07-03, STEP6에서 필수 패키지 미설치로
  GPU 할당 시간을 날린 사고 이후 만들어짐. 이후 STEP3에서도 같은 사고 예방으로 적용함.)
- **`docs/전처리_파이프라인_계획서.md`** — STEP3_전처리의 확정 설계 문서. STEP3 관련 작업 시
  애매하면 이 문서 기준으로 판단한다.

## 1. 핵심 원칙 한 줄

> **무거운 거(모델 로드) 위로는 전부 5초 안에 죽을 수 있는 검증만, 무거운 거 아래로 내려간
> 에러만 진짜 GPU/로직 문제로 취급한다.**

새 실행 스크립트를 만들거나 기존 스크립트를 고칠 때:
1. 그 폴더에 `preflight.py`가 있으면 진입점(`main()`/`run()`) 맨 앞에서 호출되고 있는지 확인한다.
   없으면 `STEP3_전처리/preflight.py` 또는 `STEP6_품질검증/preflight.py`를 템플릿으로 새로 만든다.
2. **무거운 라이브러리(torch/transformers/ultralytics/faster_whisper/langgraph/langchain* 등)를
   파일 최상단(모듈 레벨)에서 무조건 import하지 않는다.** `_load()` 함수 안이나, preflight 통과
   이후로 미룬다. (2026-07-03 실제 사고: `main.py`가 최상단에서 `langgraph`를 무조건 import해서,
   `run()` 안에 프리플라이트를 넣어도 그 줄에 도달하기 전에 이미 죽었음.)
3. config는 argparse로 사람이 손으로 입력하게 두지 말고 pydantic `BaseModel`로 검증한다.
4. 프리플라이트는 고정 리스트로 다 검사하지 말고, **config에서 실제로 선택된 impl/backend에
   필요한 패키지만** 검사한다(`STEP3_전처리/preflight.py`의 `check_config_dependent_imports`
   참고) — 안 쓰는 backend(예: vllm)의 미설치를 오탐으로 잡지 않기 위함.
5. 스모크 테스트는 "에러 없이 끝남"만으로 통과 처리하지 않는다. 결과 개수·값이 비어있지
   않은지 `assert`까지 확인한다(exit 0으로 조용히 빈 결과를 내는 게 제일 위험한 실패다).

## 2. 이 클러스터(SLURM) 필수 정보

- 로그인 노드는 GPU 없음. GPU는 `srun`으로 잡아야 함:
  ```
  srun -p RTX6000 -w n5 --exclusive --gres=gpu:2 --pty bash -l
  ```
  `--pty`는 사람이 터미널에 앉아있어야 하는 인터랙티브 셸이라 자동화 스크립트/에이전트는
  못 씀 — 항상 `srun ... python3 script.py` 형태로 실행할 명령을 직접 넘길 것.
- GPU는 **Quadro RTX 6000 (Turing, sm75)**. **fp16 / 4bit(nf4·awq·gptq)만 지원. bf16·FP8 금지**
  (bf16으로 짠 코드가 있으면 다른 환경에서 이식된 것 — float16으로 바꿀 것. 2026-07-03 STEP6에서
  실제로 발견·수정한 사례 있음).
- **vLLM은 이 클러스터에서 근본적으로 불가능**(python3.12-devel 없음, sudo 없음 → Turing용
  TRITON_ATTN의 런타임 JIT 컴파일 실패). LLM/VLM 서빙은 `transformers` + `bitsandbytes`(nf4)로
  직접 로딩하는 방식으로 설계할 것. `requirements.txt`에 `vllm`이 남아있어도 설치/사용하지 말 것.
- `LD_LIBRARY_PATH`를 새로 설정할 때 **덮어쓰지 말고 항상 앞에 추가**할 것
  (`LD_LIBRARY_PATH="새경로:$LD_LIBRARY_PATH"`). 덮어쓰면 CUDA 13.0 toolkit의
  `libnvrtc.so.13`이 사라져 torch/triton이 깨진다.
- **로그인 노드와 GPU 노드(n5)의 pip 패키지가 완전히 같다고 가정하지 말 것.** 실측 사례:
  `distro` 패키지가 로그인 노드엔 있는데 n5엔 없어서 실제 실행 시에만 터짐. 패키지 확인은
  가능하면 `srun -w n5 --gres=gpu:1 python3 -c "import ..."`로 실제 GPU 노드에서 한다.
- 각 실행 프로젝트(`STEP3_전처리/`, `STEP6_품질검증/`)는 `--system-site-packages` venv를
  쓴다(이미 깔린 torch/transformers 등 공유 패키지 재사용, 새 패키지만 venv에 추가 설치).
  다른 스텝의 venv를 공유 전역 환경에 설치하지 말 것 — 스텝마다 요구하는 패키지 버전이
  달라 서로의 환경을 깨뜨릴 수 있다(예: langchain 계열이 요구하는 transformers 버전이
  STEP3용으로 맞춰둔 버전과 다를 수 있음).

## 3. 폴더 구조 원칙

- `STEP{N}_{이름}/` — 실행 가능한 코드 + 실행 산출물만. 문서/사전연구는 여기 두지 않는다.
- `docs/` — 계획서, 서버이관 기록, 체크리스트 등 전부. 새 결정/사고가 생기면 관련 문서에 추가한다.
- `00_사전연구/` — STEP들의 착수 전 예비 연구.
- 새 문서/스크립트를 어디 둘지 애매하면 이미 있는 STEP 폴더 구조(README/config/requirements
  는 루트, 나머지는 하위 폴더로 분리)를 그대로 따른다.
