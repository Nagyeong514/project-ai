"""
Knowledge Synthesis Phase - 암묵지 합성 워커
-----------------------------------------------
파이프라인 위치: Feature Extraction -> [Knowledge Synthesis] -> DB 적재

역할:
    extractor.py가 생성한 통합 타임라인 JSON과 NCS 표준 매뉴얼 텍스트를 결합하여
    Gemini API에 전송하고, 공정 단계별 암묵지 인사이트를 정형화된 JSON으로 반환.
    반환 데이터는 tacit_knowledge_insights 테이블에 1:1로 적재 가능한 구조로 설계됨.

실행 예시:
    python synthesizer.py features.json --ncs-file ncs_manual.txt
    python synthesizer.py features.json --ncs-text "1단계: 엔진 커버 분리..."
    python synthesizer.py features.json --ncs-file ncs_manual.txt --mock

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

load_dotenv()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 시스템 프롬프트
# 모델 역할과 출력 규칙을 최소한의 토큰으로 고정한다.
# 장황한 설명을 제거하여 출력 토큰 공간을 최대화하는 것이 핵심.
# ──────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """
너는 산업 현장 암묵지 추출 전문가다.
NCS 표준 매뉴얼(형식지)과 타임스탬프 데이터를 대조하여 공정 단계별 암묵지를 도출하라.
규칙:
1. YOLO 탐지 결과(공구명, 사용 시간 비율)와 MediaPipe 수치(관절 각도 등)를 반드시 인용하라.
2. 매뉴얼과 명장 행동의 차이를 암묵지로 명확히 서술하라.
3. tacit_knowledge_description은 200자 이내로 간결하게 작성하라.
4. JSON 배열만 반환하라. 마크다운, 설명 텍스트, 코드블록 금지.
""".strip()


# ──────────────────────────────────────────────────────────────
# 사용자 프롬프트 템플릿
# 필드 설명 섹션을 제거하고 데이터와 출력 형식만 남겨 입력 토큰을 절감한다.
# ──────────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """
## 타임스탬프별 오디오/비전 통합 데이터

```json
{integrated_timeline}
```

## NCS 표준 매뉴얼

```
{ncs_manual_text}
```

## 출력 형식 (JSON 배열만 반환)

[
  {{
    "step_number": 1,
    "standard_manual_text": "해당 단계의 NCS 매뉴얼 내용을 그대로 인용",
    "tacit_knowledge_description": "YOLO/MediaPipe 수치를 포함한 암묵지 설명 (200자 이내)"
  }}
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
            "명장은 볼트를 대각선 순서로 풀었으나 NCS 매뉴얼은 순서를 미지정. "
            "YOLO: Spanner 사용 시간 38%. MediaPipe: 오른손 엄지-검지 각도가 기준 대비 평균 17도 내측. "
            "좁은 공간에서 팔꿈치로 반력을 분산시키는 실전 자세."
        ),
    },
    {
        "step_number": 2,
        "standard_manual_text": "분리된 부품을 세척액에 5분 이상 침지 후 와이어 브러시로 이물질을 제거한다.",
        "tacit_knowledge_description": (
            "명장은 세척 전 육안 확인 공정을 추가 수행. 오디오: '빛에 비춰보면 마모 방향을 알 수 있다'. "
            "MediaPipe: 해당 구간 0.8초간 시선 고정 벡터 변화 없음. 브러싱 방향 결정을 위한 숙련자 판단 루틴."
        ),
    },
    {
        "step_number": 3,
        "standard_manual_text": "부품 재조립 시 토크 렌치를 사용하여 규정 토크로 체결한다.",
        "tacit_knowledge_description": (
            "YOLO: 렌치 체결 직전 Socket 교체 2회 발생, 매뉴얼에 소켓 선택 기준 없음. "
            "오디오: '처음 한 바퀴는 손으로만 돌린 뒤 렌치를 써야 나사산이 안 죽는다'. "
            "MediaPipe: 렌치 그립 이전 0.5초간 손가락 접촉 패턴 일관 관찰."
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
    Gemini API를 호출하고, tacit_knowledge_insights 테이블에 바로 적재할 수 있는
    정형화된 인사이트 목록을 반환한다.

    초기화 파라미터:
        use_mock (bool): True이면 실제 API 호출 없이 MOCK_RESPONSE를 반환.
        model_name (str): 사용할 Gemini 모델 ID.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, use_mock: bool = False, model_name: str = DEFAULT_MODEL):
        self._use_mock   = use_mock
        self._model_name = model_name
        self._client     = None

        if not use_mock:
            self._init_client()

    def _init_client(self) -> None:
        """
        google-generativeai SDK 클라이언트를 초기화한다.
        GEMINI_API_KEY 환경 변수가 없으면 즉시 예외를 발생시켜 조용한 실패를 방지한다.
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
            1. features JSON 파일 로드 -> integrated_timeline 슬라이싱
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

        timeline = self._load_integrated_timeline(features_json_path)
        logger.info(f"[Synthesizer] 통합 타임라인 로드 완료: {len(timeline)}개 항목")

        prompt = self._build_prompt(timeline, ncs_manual_text)
        logger.info("[Synthesizer] 프롬프트 빌드 완료")

        if self._use_mock:
            logger.info("[Synthesizer] Mock 모드: Gemini API 호출 생략")
            raw_response = json.dumps(MOCK_RESPONSE, ensure_ascii=False)
        else:
            raw_response = self._call_gemini(prompt)

        insights = self._parse_and_validate(raw_response)
        logger.info(f"[Synthesizer] 인사이트 {len(insights)}개 도출 완료")

        return insights

    # ------------------------------------------------------------------
    # 내부 구현 메서드
    # ------------------------------------------------------------------

    @staticmethod
    def _load_integrated_timeline(features_json_path: str) -> list:
        """extractor.py 출력 JSON에서 integrated_timeline 키만 추출한다."""
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

        입력 토큰 절감 전략:
            1. 오디오 발화 또는 도구 탐지가 있는 프레임만 필터링한다.
            2. 최대 30개 항목으로 샘플링하여 컨텍스트 과부하를 방지한다.
            3. hand_movement_vector 원시 좌표(21개 관절 x 3축 = 63개 수치)를
               요약 통계(손목 좌표 + 평균 x/y)로 축소한다.
               이것만으로도 항목당 입력 토큰을 약 80% 절감할 수 있다.
        """
        filtered = [
            entry for entry in timeline
            if entry.get("audio_context") or entry.get("detected_tools")
        ]

        if len(filtered) > 30:
            step = max(1, len(filtered) // 30)
            filtered = filtered[::step][:30]
            logger.warning(f"[Synthesizer] 타임라인 항목이 많아 30개로 샘플링하여 전달합니다.")

        # hand_movement_vector 원시 좌표를 요약 통계로 교체
        slim = []
        for entry in filtered:
            e = dict(entry)
            vec = e.get("hand_movement_vector")
            if vec and isinstance(vec, list) and len(vec) > 0:
                xs = [p[0] for p in vec if isinstance(p, (list, tuple)) and len(p) >= 2]
                ys = [p[1] for p in vec if isinstance(p, (list, tuple)) and len(p) >= 2]
                e["hand_movement_vector"] = {
                    "wrist_xy": vec[0][:2] if isinstance(vec[0], (list, tuple)) else None,
                    "mean_x": round(sum(xs) / len(xs), 4) if xs else None,
                    "mean_y": round(sum(ys) / len(ys), 4) if ys else None,
                    "landmark_count": len(vec),
                }
            slim.append(e)

        timeline_json_str = json.dumps(slim, ensure_ascii=False, indent=2)

        return USER_PROMPT_TEMPLATE.format(
            integrated_timeline=timeline_json_str,
            ncs_manual_text=ncs_manual_text.strip(),
        )

    def _call_gemini(self, prompt: str) -> str:
        """
        Gemini API를 호출하고 응답 텍스트를 반환한다.

        GenerationConfig 설명:
            - temperature=0.2           : 구조화된 JSON 출력에 적합한 낮은 창의성.
            - max_output_tokens=8192    : gemini-2.5-flash 기본값(4096)의 두 배.
                                          thinking 토큰 소비분을 감안하여 넉넉하게 설정.
            - response_mime_type        : JSON 전용 출력 모드 강제 활성화.
                                          모델이 JSON 구조를 중간에 끊지 않도록 보장.
        """
        logger.info(f"[Synthesizer] Gemini API 호출 중... (model: {self._model_name})")

        import google.generativeai as genai

        try:
            print(f"[디버그] Gemini API 호출 시도 (모델명: {self._model_name})")
            response = self._client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
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
                print("[진단] 모델명을 찾을 수 없습니다. gemini-2.5-flash로 정확히 수정되었는지 확인하십시오.")
            print("=" * 50 + "\n")
            raise e

        logger.info("[Synthesizer] Gemini API 응답 수신 완료")
        return response.text

    @staticmethod
    def _parse_and_validate(raw_response: str) -> list[dict]:
        """
        Gemini 응답 문자열을 JSON으로 파싱하고 필수 필드 존재 여부를 검증한다.

        response_mime_type="application/json" 설정 시에도 간혹 마크다운 코드블록이
        붙는 경우가 있어 방어적으로 제거한다.
        """
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
    """도출된 암묵지 인사이트를 JSON 파일로 저장한다."""
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

    if args.ncs_file:
        ncs_path = Path(args.ncs_file)
        if not ncs_path.is_file():
            raise FileNotFoundError(f"NCS 매뉴얼 파일을 찾을 수 없습니다: {args.ncs_file}")
        ncs_manual_text = ncs_path.read_text(encoding="utf-8")
    else:
        ncs_manual_text = args.ncs_text

    output_path = args.output or (
        str(Path(args.features_json).with_suffix("")).replace("_features", "") + "_insights.json"
    )

    synthesizer = KnowledgeSynthesizer(use_mock=args.mock, model_name=args.model)
    insights = synthesizer.synthesize(args.features_json, ncs_manual_text)
    save_insights(insights, output_path)

    if insights:
        print("\n[결과 미리보기 - Step 1]")
        print(json.dumps(insights[0], ensure_ascii=False, indent=2))
    print(f"\n총 {len(insights)}개 인사이트 -> {output_path}")


if __name__ == "__main__":
    main()
