"""
Knowledge Synthesis Phase - 암묵지 합성 워커
-----------------------------------------------
파이프라인 위치: Feature Extraction → [Knowledge Synthesis] → DB 적재

역할:
    extractor.py가 생성한 통합 타임라인 JSON과 NCS 표준 매뉴얼 텍스트를 결합하여
    Gemini 1.5 Pro API에 전송하고, 공정 단계별 암묵지 인사이트를 정형화된 JSON으로 반환.
    반환 데이터는 tacit_knowledge_insights 테이블에 1:1로 적재 가능한 구조로 설계됨.

실행 예시:
    python synthesizer.py features.json --ncs-file ncs_manual.txt
    python synthesizer.py features.json --ncs-text "1단계: 엔진 커버 분리..."
    python synthesizer.py features.json --ncs-file ncs_manual.txt --mock  # API 없이 테스트

의존 패키지:
    pip install google-generativeai python-dotenv
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# .env 파일에서 GEMINI_API_KEY 로드
load_dotenv()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 시스템 프롬프트 (System Instruction)
# Gemini 모델의 역할과 분석 기준을 고정하는 페르소나 정의.
# ──────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """
너는 산업 현장의 숙련자 암묵지 추출 전문가이다.

제공된 타임스탬프별 오디오/비전 통합 데이터와 NCS 표준 매뉴얼을 대조 분석하여,
매뉴얼에 기재되지 않은 명장만의 실전 팁, 손 무브먼트의 비밀, 공구 활용 노하우를
공정 단계(step_number)별로 정밀하게 도출하라.

분석 원칙:
1. NCS 표준 매뉴얼의 기술 내용을 기준선(형식지)으로 삼아라.
2. 오디오 트랜스크립트에서 명장의 언어적 설명을 해석하라.
3. YOLO 탐지 결과(도구 사용 빈도, 순서)와 MediaPipe 손 관절 데이터(각도, 속도 패턴)에서
   계량화 가능한 수치적 차이를 반드시 언급하라.
4. 매뉴얼과 명장 행동의 차이가 '암묵지'임을 명확히 서술하라.
5. 모든 출력은 반드시 아래 JSON 배열 형식을 엄격히 지켜라. 다른 텍스트는 절대 포함하지 마라.
""".strip()


# ──────────────────────────────────────────────────────────────
# 사용자 프롬프트 템플릿
# Gemini에게 전달할 실제 분석 요청 본문.
# {integrated_timeline}과 {ncs_manual_text} 두 자리를 동적으로 채운다.
# ──────────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """
## 분석 대상 데이터

### 1. 타임스탬프별 오디오/비전 통합 데이터 (extractor.py 출력)
각 항목의 의미:
- timestamp_sec : 영상 내 시간 위치 (초)
- audio_context : 해당 시점 명장의 음성 발화 내용 (Whisper 추출)
- detected_tools : YOLO가 탐지한 공구/부품 목록과 신뢰도
- hand_movement_vector : MediaPipe가 추출한 양손 21개 관절 좌표 (정규화)

```json
{integrated_timeline}
```

### 2. NCS 표준 매뉴얼 (형식지 기준선)
```
{ncs_manual_text}
```

## 출력 형식 (JSON 배열 엄수)

아래 구조의 JSON 배열만 반환하라. 마크다운 코드블록이나 설명 텍스트는 포함하지 마라.

[
  {{
    "step_number": 1,
    "standard_manual_text": "NCS 매뉴얼에서 이 단계에 해당하는 내용을 그대로 인용",
    "tacit_knowledge_description": "명장이 매뉴얼과 다르게 행동한 구체적인 방식과 그 이유. 수치 데이터(각도, 시간 비율 등)를 반드시 포함하여 서술"
  }},
  ...
]
""".strip()


# ──────────────────────────────────────────────────────────────
# Mock 응답 데이터
# GEMINI_API_KEY 없이 파이프라인 흐름 검증 시 사용.
# ──────────────────────────────────────────────────────────────

MOCK_RESPONSE = [
    {
        "step_number": 1,
        "standard_manual_text": "엔진 커버 고정 볼트 6개를 규정 토크(25Nm)로 풀어 분리한다.",
        "tacit_knowledge_description": (
            "명장은 볼트를 푸는 순서를 대각선 패턴으로 진행하였으나 NCS 매뉴얼은 순서를 지정하지 않는다. "
            "YOLO 분석 결과 Spanner 사용 시간이 전체 공정의 38%를 차지하였으며, "
            "MediaPipe 데이터 상 오른손 엄지-검지 관절 각도가 NCS 기준 자세 대비 평균 17도 더 내측으로 "
            "꺾인 채로 토크를 가하는 패턴이 관찰됨. 이는 좁은 공간에서 반력을 팔꿈치로 분산시키는 "
            "실전 습득 자세로, 매뉴얼에 기재되지 않은 핵심 노하우다."
        ),
    },
    {
        "step_number": 2,
        "standard_manual_text": "분리된 부품을 세척액에 5분 이상 침지 후 와이어 브러시로 이물질을 제거한다.",
        "tacit_knowledge_description": (
            "명장은 세척 전 육안 확인 단계를 NCS 매뉴얼에 없는 추가 공정으로 수행하였다. "
            "오디오 트랜스크립트에서 '빛에 비춰보면 마모 방향을 알 수 있다'는 발화가 확인되었으며, "
            "해당 구간에서 MediaPipe 기준 머리 기울기 벡터의 변화가 0.8초간 집중 고정되는 패턴을 보임. "
            "이는 마모 방향성을 파악하는 시선 집중 구간으로, 이후 브러싱 방향 결정에 영향을 주는 "
            "숙련자 고유의 판단 프로세스다."
        ),
    },
    {
        "step_number": 3,
        "standard_manual_text": "부품 재조립 시 토크 렌치를 사용하여 규정 토크로 체결한다.",
        "tacit_knowledge_description": (
            "YOLO 분석 결과 토크 렌치(Torque Wrench) 체결 직전 Socket 교체 동작이 2회 발생하였으나 "
            "매뉴얼에는 소켓 선택 기준이 명시되어 있지 않다. "
            "오디오에서 '처음 한 바퀴는 손으로만 돌린 뒤 렌치를 써야 나사산이 안 죽는다'는 발화가 확인됨. "
            "이 수작업 선조립 루틴은 MediaPipe 데이터에서도 렌치 그립 이전 0.5초간 손가락 접촉 패턴으로 "
            "일관되게 관찰되었으며, 나사산 보호를 위한 장인의 핵심 암묵 루틴이다."
        ),
    },
]


# ──────────────────────────────────────────────────────────────
# KnowledgeSynthesizer 클래스
# ──────────────────────────────────────────────────────────────

class KnowledgeSynthesizer:
    """
    Knowledge Synthesis Phase 핵심 클래스.

    extractor.py의 출력(features JSON)과 NCS 표준 매뉴얼 텍스트를 입력으로 받아
    Gemini 1.5 Pro API를 호출하고, tacit_knowledge_insights 테이블에 바로 적재할 수 있는
    정형화된 인사이트 목록을 반환한다.

    초기화 파라미터:
        use_mock (bool): True이면 실제 API 호출 없이 MOCK_RESPONSE를 반환.
                         API 키 없이 파이프라인 전체 흐름을 검증할 때 사용.
        model_name (str): 사용할 Gemini 모델 ID.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, use_mock: bool = False, model_name: str = DEFAULT_MODEL):
        self._use_mock   = use_mock
        self._model_name = model_name
        self._client     = None  # 실제 API 사용 시 _init_client()에서 초기화

        if not use_mock:
            self._init_client()

    def _init_client(self) -> None:
        """
        google-generativeai SDK 클라이언트를 초기화한다.
        GEMINI_API_KEY 환경 변수가 없으면 즉시 예외를 발생시켜 조용한 실패를 방지한다.

        실제 SDK 사용법:
            pip install google-generativeai
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self._client = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=SYSTEM_INSTRUCTION,
            )
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        print("\n[디버그] 라이브러리 초기화 시작")
        if not api_key:
            print("[오류] .env 파일에서 GEMINI_API_KEY를 읽어오지 못했습니다.")
            raise EnvironmentError(
                "GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. "
                ".env 파일에 GEMINI_API_KEY=<키> 를 추가하거나 "
                "--mock 플래그로 실행하세요."
            )
        print(f"[정보] API 키 로드 성공 (길이: {len(api_key)}자, 시작 글자: {api_key[:2]})")

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            logger.info(f"[Synthesizer] Gemini 클라이언트 초기화 완료: {self._model_name}")
        except ImportError:
            raise ImportError(
                "google-generativeai 패키지가 설치되지 않았습니다. "
                "pip install google-generativeai 를 실행하세요."
            )

    # ------------------------------------------------------------------
    # 퍼블릭 인터페이스
    # ------------------------------------------------------------------

    def synthesize(self, features_json_path: str, ncs_manual_text: str) -> list[dict]:
        """
        Feature Extraction 결과와 NCS 매뉴얼을 결합하여 암묵지 인사이트를 도출한다.

        처리 흐름:
            1. features JSON 파일 로드 → integrated_timeline 슬라이싱
            2. 사용자 프롬프트 빌드 (타임라인 + NCS 매뉴얼 삽입)
            3. Gemini API 호출 (또는 mock 반환)
            4. 응답 JSON 파싱 및 유효성 검증
            5. tacit_knowledge_insights 테이블 구조와 일치하는 list[dict] 반환

        Args:
            features_json_path: extractor.py가 생성한 _features.json 파일 경로
            ncs_manual_text:    분석 기준이 될 NCS 표준 매뉴얼 전문 텍스트

        Returns:
            list[dict] - 각 항목은 tacit_knowledge_insights 테이블 1행에 대응.
            필드: step_number, standard_manual_text, tacit_knowledge_description
        """
        logger.info("[Synthesizer] Knowledge Synthesis Phase 시작")

        # Step 1: 특징 데이터 로드
        timeline = self._load_integrated_timeline(features_json_path)
        logger.info(f"[Synthesizer] 통합 타임라인 로드 완료: {len(timeline)}개 항목")

        # Step 2: 프롬프트 구성
        prompt = self._build_prompt(timeline, ncs_manual_text)
        logger.info("[Synthesizer] 프롬프트 빌드 완료")

        # Step 3: Gemini API 호출 또는 mock 반환
        if self._use_mock:
            logger.info("[Synthesizer] Mock 모드: Gemini API 호출 생략")
            raw_response = json.dumps(MOCK_RESPONSE, ensure_ascii=False)
        else:
            raw_response = self._call_gemini(prompt)

        # Step 4: 응답 파싱 및 검증
        insights = self._parse_and_validate(raw_response)
        logger.info(f"[Synthesizer] 인사이트 {len(insights)}개 도출 완료")

        return insights

    # ------------------------------------------------------------------
    # 내부 구현 메서드
    # ------------------------------------------------------------------

    @staticmethod
    def _load_integrated_timeline(features_json_path: str) -> list:
        """
        extractor.py 출력 JSON에서 integrated_timeline 키만 추출한다.
        Gemini에 넘길 토큰 수를 최소화하기 위해 불필요한 raw frame_features 데이터는 제외.
        """
        path = Path(features_json_path)
        if not path.is_file():
            raise FileNotFoundError(f"특징 데이터 파일을 찾을 수 없습니다: {features_json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        timeline = data.get("integrated_timeline")
        if not timeline:
            raise ValueError(
                f"'integrated_timeline' 키가 없거나 비어있습니다: {features_json_path}\n"
                "extractor.py를 먼저 실행하여 features JSON을 생성하세요."
            )
        return timeline

    @staticmethod
    def _build_prompt(timeline: list, ncs_manual_text: str) -> str:
        """
        USER_PROMPT_TEMPLATE에 타임라인 JSON과 NCS 매뉴얼 텍스트를 삽입하여 최종 프롬프트를 생성한다.

        토큰 절약 전략:
            - hand_movement_vector의 좌표 배열은 모든 프레임을 전달하면 토큰이 폭증하므로,
              audio_context가 존재하는 프레임만 필터링하여 의미 있는 구간만 전달한다.
            - detected_tools가 비어있는 프레임도 제외한다.
        """
        # 의미 있는 프레임만 필터링 (오디오 발화 또는 도구 탐지가 있는 구간)
        filtered = [
            entry for entry in timeline
            if entry.get("audio_context") or entry.get("detected_tools")
        ]

        # 프레임이 너무 많을 경우 최대 50개로 제한 (컨텍스트 윈도우 보호)
        if len(filtered) > 50:
            step = max(1, len(filtered) // 50)
            filtered = filtered[::step][:50]
            logger.warning(
                f"[Synthesizer] 타임라인 항목이 많아 {len(filtered)}개로 샘플링하여 전달합니다."
            )

        timeline_json_str = json.dumps(filtered, ensure_ascii=False, indent=2)

        return USER_PROMPT_TEMPLATE.format(
            integrated_timeline=timeline_json_str,
            ncs_manual_text=ncs_manual_text.strip(),
        )

    def _call_gemini(self, prompt: str) -> str:
        """
        Gemini 1.5 Pro API를 호출하고 응답 텍스트를 반환한다.

        GenerationConfig 설명:
            - temperature=0.2  : 창의성보다 일관성 우선. 구조화된 JSON 출력에 낮은 값이 적합.
            - max_output_tokens: 인사이트 10개 기준 충분한 출력 공간 확보.
            - response_mime_type: "application/json"으로 설정하면 Gemini가
              JSON만 반환하도록 강제하는 네이티브 구조화 출력 기능 활성화.
              (gemini-1.5-pro-002 이상에서 지원)

        실제 SDK 코드:
            import google.generativeai as genai
            response = self._client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            return response.text
        """
        logger.info(f"[Synthesizer] Gemini API 호출 중... (model: {self._model_name})")

        import google.generativeai as genai

        try:
            print(f"[디버그] Gemini API 호출 시도 (모델명: {self._model_name})")
            response = self._client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            print("[디버그] Gemini API 호출 성공")
        except Exception as e:
            print("\n" + "=" * 50)
            print("[오류] Gemini API 호출 중 예외가 발생했습니다.")
            print(f"오류 타입: {type(e).__name__}")
            print(f"오류 내용: {str(e)}")
            if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                print("[진단] 입력된 API 키가 유효하지 않습니다. .env 파일의 키 값을 재확인하십시오.")
            elif "404" in str(e) or "not found" in str(e).lower():
                print("[진단] 모델명을 찾을 수 없습니다. 모델명이 gemini-2.5-flash로 정확히 수정되었는지 확인하십시오.")
            print("=" * 50 + "\n")
            raise e

        logger.info("[Synthesizer] Gemini API 응답 수신 완료")
        return response.text

    @staticmethod
    def _parse_and_validate(raw_response: str) -> list[dict]:
        """
        Gemini 응답 문자열을 JSON으로 파싱하고 필수 필드 존재 여부를 검증한다.

        Gemini가 간혹 JSON 앞뒤에 마크다운 코드블록(```json ... ```)을 붙이는 경우를 방어적으로 처리.
        필수 필드 누락 항목은 경고 로그와 함께 건너뛴다.
        """
        # 마크다운 코드블록 제거 (방어 처리)
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Gemini 응답을 JSON으로 파싱하지 못했습니다.\n"
                f"파싱 오류: {e}\n"
                f"원본 응답 (앞 500자):\n{raw_response[:500]}"
            )

        if not isinstance(parsed, list):
            raise ValueError(
                f"Gemini 응답이 배열(list) 형태가 아닙니다. 받은 타입: {type(parsed).__name__}"
            )

        REQUIRED_FIELDS = {"step_number", "standard_manual_text", "tacit_knowledge_description"}
        validated = []
        for i, item in enumerate(parsed):
            missing = REQUIRED_FIELDS - set(item.keys())
            if missing:
                logger.warning(f"[Synthesizer] 항목 {i} 필수 필드 누락으로 건너뜀: {missing}")
                continue
            validated.append(item)

        return validated


# ──────────────────────────────────────────────────────────────
# 결과 저장 헬퍼
# ──────────────────────────────────────────────────────────────

def save_insights(insights: list[dict], output_path: str) -> None:
    """
    도출된 암묵지 인사이트를 JSON 파일로 저장한다.
    이 파일이 Persistence Phase(DB 적재)의 직접 입력 소스가 된다.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)
    logger.info(f"[Synthesizer] 인사이트 저장 완료: {output_path}")


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="암묵지 추출 시스템 - Knowledge Synthesis Phase 워커",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "features_json",
        help="extractor.py가 생성한 _features.json 파일 경로",
    )

    # NCS 매뉴얼 입력: 파일 또는 직접 텍스트 중 하나 선택
    ncs_group = parser.add_mutually_exclusive_group(required=True)
    ncs_group.add_argument(
        "--ncs-file",
        metavar="PATH",
        help="NCS 표준 매뉴얼 텍스트 파일 경로 (.txt)",
    )
    ncs_group.add_argument(
        "--ncs-text",
        metavar="TEXT",
        help="NCS 표준 매뉴얼 내용을 직접 문자열로 전달",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 JSON 파일 경로 (기본값: <features명>_insights.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Gemini API 호출 없이 mock 응답으로 파이프라인 흐름 검증",
    )
    parser.add_argument(
        "--model",
        default=KnowledgeSynthesizer.DEFAULT_MODEL,
        help=f"사용할 Gemini 모델 ID (기본값: {KnowledgeSynthesizer.DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    # NCS 매뉴얼 텍스트 준비
    if args.ncs_file:
        ncs_path = Path(args.ncs_file)
        if not ncs_path.is_file():
            raise FileNotFoundError(f"NCS 매뉴얼 파일을 찾을 수 없습니다: {args.ncs_file}")
        ncs_manual_text = ncs_path.read_text(encoding="utf-8")
    else:
        ncs_manual_text = args.ncs_text

    # 출력 경로 결정
    output_path = args.output or (
        str(Path(args.features_json).with_suffix("")).replace("_features", "") + "_insights.json"
    )

    # 실행
    synthesizer = KnowledgeSynthesizer(use_mock=args.mock, model_name=args.model)
    insights = synthesizer.synthesize(args.features_json, ncs_manual_text)
    save_insights(insights, output_path)

    # 결과 미리보기 (첫 번째 인사이트)
    if insights:
        print("\n[결과 미리보기 - Step 1]")
        print(json.dumps(insights[0], ensure_ascii=False, indent=2))
    print(f"\n총 {len(insights)}개 인사이트 → {output_path}")


if __name__ == "__main__":
    main()
