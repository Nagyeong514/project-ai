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

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 시스템 프롬프트
# 3단계 판정 가드레일을 심어 억지 매핑(할루시네이션)을 원천 차단한다.
# temperature=0.0과 함께 사용할 때 최대 효과를 발휘한다.
# ──────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """
너는 엔진본체 정비 공정 암묵지 추출 감사관이다.
제공된 비전/오디오 타임스탬프 데이터와 NCS 엔진본체 정비 매뉴얼을 대조하여
각 공정 단계를 아래 3가지 상태 중 정확히 하나로 판정하라.

[판정 기준]
- MISMATCH: 영상 데이터가 엔진본체 정비(실린더 헤드, 크랭크축, 타이밍 체인 등) 도메인과
  무관한 공정일 때. 이 판정이 나오면 해당 단계의 분석을 즉시 중단하라.
- STANDARD_COMPLIANCE: 명장의 행위가 NCS 매뉴얼 기준선과 100% 일치하며
  초과 노하우나 수치 보정이 관찰되지 않을 때.
- SUCCESS: YOLO 탐지 공구 데이터, MediaPipe 관절 수치, Whisper 인용 대사로
  입증 가능한 매뉴얼 기준선 초과 정량 노하우가 확인될 때만 판정하라.

[엄수 규칙]
1. fact_evidence.detected_tools에 YOLO 탐지 공구명을 반드시 기재하라.
2. fact_evidence.verified_quotes에 Whisper가 인식한 정비사 발화(수치 포함)를 인용하라.
   발화가 없으면 빈 배열([])을 기재하라.
3. MISMATCH 또는 STANDARD_COMPLIANCE 판정 시 tacit_knowledge_description은 반드시 '해당 없음'으로 기재하라.
4. SUCCESS 판정 근거가 데이터에 없으면 절대로 만들어내지 마라.
5. JSON 배열만 반환하라. 마크다운, 설명 텍스트, 코드블록 일체 금지.
""".strip()


# ──────────────────────────────────────────────────────────────
# 사용자 프롬프트 템플릿
# 데이터와 출력 스키마만 포함하여 입력 토큰을 최소화한다.
# ──────────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """
## 타임스탬프별 오디오/비전 통합 데이터

```json
{integrated_timeline}
```

## NCS 엔진본체 정비 표준 매뉴얼

```
{ncs_manual_text}
```

## 출력 스키마 (JSON 배열만 반환)

[
  {{
    "step_number": 1,
    "status": "SUCCESS | MISMATCH | STANDARD_COMPLIANCE",
    "standard_manual_text": "NCS 매뉴얼에서 해당 단계 기준선 요약",
    "tacit_knowledge_description": "SUCCESS 시 정량적 암묵지 상세 / 그 외 '해당 없음'",
    "fact_evidence": {{
      "detected_tools": ["YOLO 탐지 공구명 목록"],
      "verified_quotes": ["Whisper 인식 정비사 발화 및 수치"]
    }}
  }}
]
""".strip()


# ──────────────────────────────────────────────────────────────
# Mock 응답 데이터
# --mock 플래그로 API 없이 파이프라인 흐름을 검증할 때 사용한다.
# ──────────────────────────────────────────────────────────────

MOCK_RESPONSE = [
    {
        "step_number": 1,
        "status": "SUCCESS",
        "standard_manual_text": "실린더 헤드 볼트는 각도법으로 규정 토크와 각도에 맞게 조립한다.",
        "tacit_knowledge_description": (
            "Whisper 인식: '처음 한 바퀴는 손으로만 돌린 뒤 렌치를 써야 나사산이 안 죽는다'. "
            "YOLO: Torque Wrench 체결 직전 Socket 교체 2회. MediaPipe: 렌치 그립 이전 "
            "0.5초간 손가락 접촉 패턴 일관 관찰. 매뉴얼 미기재 선조립 루틴."
        ),
        "fact_evidence": {
            "detected_tools": ["Torque Wrench", "Socket"],
            "verified_quotes": ["처음 한 바퀴는 손으로만 돌린 뒤 렌치를 써야 나사산이 안 죽는다"],
        },
    },
    {
        "step_number": 2,
        "status": "STANDARD_COMPLIANCE",
        "standard_manual_text": "크랭크축 메인 저널 베어링 캡 볼트를 4~5회 나누어 단계적으로 규정 토크로 조인다.",
        "tacit_knowledge_description": "해당 없음",
        "fact_evidence": {
            "detected_tools": ["Wrench", "Spanner"],
            "verified_quotes": [],
        },
    },
    {
        "step_number": 3,
        "status": "MISMATCH",
        "standard_manual_text": "해당 없음",
        "tacit_knowledge_description": "해당 없음",
        "fact_evidence": {
            "detected_tools": [],
            "verified_quotes": [],
        },
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
            1. features JSON 로드 -> integrated_timeline 슬라이싱
            2. 프롬프트 빌드 (타임라인 + NCS 매뉴얼 삽입)
            3. Gemini API 호출 (또는 mock 반환)
            4. 응답 JSON 파싱 및 유효성 검증
            5. list[dict] 반환

        Args:
            features_json_path: extractor.py가 생성한 _features.json 파일 경로
            ncs_manual_text:    분석 기준이 될 NCS 표준 매뉴얼 전문 텍스트

        Returns:
            list[dict] - 각 항목은 tacit_knowledge_insights 테이블 1행에 대응.
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
        USER_PROMPT_TEMPLATE에 타임라인 JSON과 NCS 매뉴얼 텍스트를 삽입한다.

        입력 토큰 절감 전략:
            1. 오디오 발화 또는 도구 탐지가 있는 프레임만 필터링한다.
            2. 최대 30개 항목으로 샘플링하여 컨텍스트 과부하를 방지한다.
            3. hand_movement_vector 원시 좌표(21개 관절)를 요약 통계로 축소한다.
        """
        filtered = [
            entry for entry in timeline
            if entry.get("audio_context") or entry.get("detected_tools")
        ]

        if len(filtered) > 30:
            step = max(1, len(filtered) // 30)
            filtered = filtered[::step][:30]
            logger.warning(f"[Synthesizer] 타임라인 항목 수가 많아 30개로 샘플링합니다.")

        slim = []
        for entry in filtered:
            e = dict(entry)
            vec = e.get("hand_movement_vector")
            if vec and isinstance(vec, list) and len(vec) > 0:
                xs = [p[0] for p in vec if isinstance(p, (list, tuple)) and len(p) >= 2]
                ys = [p[1] for p in vec if isinstance(p, (list, tuple)) and len(p) >= 2]
                e["hand_movement_vector"] = {
                    "wrist_xy":      vec[0][:2] if isinstance(vec[0], (list, tuple)) else None,
                    "mean_x":        round(sum(xs) / len(xs), 4) if xs else None,
                    "mean_y":        round(sum(ys) / len(ys), 4) if ys else None,
                    "landmark_count": len(vec),
                }
            slim.append(e)

        return USER_PROMPT_TEMPLATE.format(
            integrated_timeline=json.dumps(slim, ensure_ascii=False, indent=2),
            ncs_manual_text=ncs_manual_text.strip(),
        )

    def _call_gemini(self, prompt: str) -> str:
        """
        Gemini API를 호출하고 응답 텍스트를 반환한다.

        GenerationConfig:
            temperature=0.0        : 팩트 기반 출력 강제. 창의적 추론 차단.
            max_output_tokens=8192 : thinking 토큰 소비분을 감안한 충분한 출력 공간.
            response_mime_type     : JSON 전용 출력 모드 강제 활성화.
        """
        logger.info(f"[Synthesizer] Gemini API 호출 중... (model: {self._model_name})")

        import google.generativeai as genai

        try:
            print(f"[디버그] Gemini API 호출 시도 (모델명: {self._model_name})")
            response = self._client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
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
                print("[진단] API 키가 유효하지 않습니다. .env 파일의 키 값을 재확인하세요.")
            elif "404" in str(e) or "not found" in str(e).lower():
                print("[진단] 모델명을 찾을 수 없습니다. gemini-2.5-flash로 설정되어 있는지 확인하세요.")
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

        REQUIRED_FIELDS = {
            "step_number", "status", "standard_manual_text",
            "tacit_knowledge_description", "fact_evidence",
        }
        VALID_STATUSES = {"SUCCESS", "MISMATCH", "STANDARD_COMPLIANCE"}

        validated = []
        for i, item in enumerate(parsed):
            missing = REQUIRED_FIELDS - set(item.keys())
            if missing:
                logger.warning(f"[Synthesizer] 항목 {i} 필수 필드 누락으로 건너뜀: {missing}")
                continue
            if item.get("status") not in VALID_STATUSES:
                logger.warning(
                    f"[Synthesizer] 항목 {i} status 값 비정상 ({item.get('status')}) 건너뜀"
                )
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
    parser.add_argument("features_json", help="extractor.py가 생성한 _features.json 파일 경로")

    ncs_group = parser.add_mutually_exclusive_group(required=True)
    ncs_group.add_argument("--ncs-file", metavar="PATH", help="NCS 매뉴얼 텍스트 파일 경로 (.txt)")
    ncs_group.add_argument("--ncs-text", metavar="TEXT", help="NCS 매뉴얼 내용을 직접 문자열로 전달")

    parser.add_argument("--output", "-o", default=None, help="출력 JSON 파일 경로")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Gemini API 호출 없이 mock 응답으로 파이프라인 흐름 검증")
    parser.add_argument("--model", default=KnowledgeSynthesizer.DEFAULT_MODEL,
                        help=f"사용할 Gemini 모델 ID (기본값: {KnowledgeSynthesizer.DEFAULT_MODEL})")
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
