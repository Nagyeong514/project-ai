"""
암묵지 추출 파이프라인 오케스트레이터
---------------------------------------
파이프라인 전체를 단일 명령으로 실행한다.

    extractor.py  →  synthesizer.py  →  loader.py

실행 예시 (실제 API/DB 사용):
    python pipeline.py sample.mp4 \\
        --ncs-file ncs_auto_repair.txt \\
        --master-name "김철수" \\
        --job-category "자동차 정비" \\
        --ncs-code "LM1501010106_19v4"

실행 예시 (API/DB 없이 흐름 검증 - mock 모드):
    python pipeline.py sample.mp4 \\
        --ncs-file ncs_auto_repair.txt \\
        --master-name "김철수" \\
        --job-category "자동차 정비" \\
        --mock

옵션 설명:
    --mock      Gemini API 호출과 DB 적재를 모두 건너뛰고 mock 데이터로 흐름만 검증
    --skip-db   Gemini는 실제 호출하되 DB 적재만 건너뜀
               (API 키는 있으나 MySQL 미설정 상태에서 사용)

의존 패키지:
    pip install opencv-python google-generativeai pymysql python-dotenv
"""

import os
import sys
import time
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymysql
import pymysql.cursors
from dotenv import load_dotenv

load_dotenv()

# Windows 터미널에서 ANSI 색상 코드 활성화
if sys.platform == "win32":
    os.system("")

# ──────────────────────────────────────────────────────────────
# 터미널 색상 상수
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# NCS 마스터 테이블 조회
# ncs_modules 테이블에서 ncs_code에 해당하는 매뉴얼 텍스트를 가져온다.
# 이 함수 덕분에 --ncs-text를 손으로 입력할 필요가 없어진다.
# ──────────────────────────────────────────────────────────────

def fetch_ncs_text_from_db(ncs_code: str) -> str:
    """
    ncs_modules 테이블에서 ncs_code로 standard_manual_text를 조회한다.

    Args:
        ncs_code: NCS 능력단위 코드 (예: LM1506030201_24v5)

    Returns:
        조회된 매뉴얼 텍스트 문자열

    Raises:
        EnvironmentError: .env DB 설정 누락 시
        ValueError: 해당 ncs_code가 ncs_modules 테이블에 없을 때
    """
    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f".env 파일에 다음 항목이 없습니다: {', '.join(missing)}"
        )

    conn = pymysql.connect(
        host     = os.environ["DB_HOST"],
        port     = int(os.environ.get("DB_PORT", 3306)),
        user     = os.environ["DB_USER"],
        password = os.environ["DB_PASSWORD"],
        database = os.environ["DB_NAME"],
        charset  = "utf8mb4",
        cursorclass = pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT standard_manual_text FROM ncs_modules WHERE ncs_code = %s",
                (ncs_code,),
            )
            row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        raise ValueError(
            f"ncs_modules 테이블에 '{ncs_code}' 코드가 없습니다. "
            "MySQL Workbench에서 INSERT가 완료되었는지 확인하세요."
        )

    return row["standard_manual_text"]


class _C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"
    LINE   = "\033[90m" + ("-" * 60) + "\033[0m"
    DLINE  = "\033[36m" + ("=" * 60) + "\033[0m"


# ──────────────────────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """단일 파이프라인 스테이지 실행 결과."""
    name:        str
    success:     bool
    elapsed_sec: float
    summary:     str   # 성공 시 요약, 실패 시 오류 메시지


@dataclass
class PipelineReport:
    """전체 파이프라인 실행 결과 보고서."""
    video_path:        str
    stage_results:     list  = field(default_factory=list)
    total_sec:         float = 0.0
    frame_count:       int   = 0
    insight_count:     int   = 0
    highlight_count:   int   = 0
    features_path:     str   = ""
    insights_path:     str   = ""
    highlights_path:   str   = ""


# ──────────────────────────────────────────────────────────────
# 파이프라인 오케스트레이터
# ──────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    4단계 암묵지 추출 파이프라인을 순서대로 실행하는 오케스트레이터.

    실행 순서:
        Stage 1. Feature Extraction  (extractor.py 로직)
        Stage 2. Knowledge Synthesis (synthesizer.py 로직)
        Stage 3. Persistence         (loader.py 로직)

    실패 처리:
        어느 단계에서든 예외가 발생하면 이후 단계를 실행하지 않고
        오류 내용과 함께 최종 보고서를 출력한 뒤 프로세스를 종료한다.

    Args:
        mock    : True이면 Gemini API 호출 없이 mock 인사이트를 사용하고 DB 적재도 건너뜀
        skip_db : True이면 Gemini는 실제 호출하되 DB 적재 단계만 건너뜀
    """

    def __init__(self, mock: bool = False, skip_db: bool = False):
        self._mock    = mock
        self._skip_db = skip_db or mock  # mock이면 DB도 자동 스킵

    # ------------------------------------------------------------------
    # 퍼블릭 인터페이스
    # ------------------------------------------------------------------

    def run(
        self,
        video_path:      str,
        ncs_manual_text: str,
        master_name:     str,
        job_category:    str,
        ncs_code:        str,
    ) -> None:
        """
        파이프라인 3단계를 순서대로 실행하고 최종 보고서를 출력한다.

        Args:
            video_path      : 처리할 MP4 영상 경로
            ncs_manual_text : NCS 표준 매뉴얼 전문 텍스트
            master_name     : 명장 식별자
            job_category    : 직군 분류명
            ncs_code        : NCS 능력단위 코드
        """
        report       = PipelineReport(video_path=video_path)
        pipeline_start = time.perf_counter()

        self._print_banner(video_path, master_name, job_category)

        # 중간 산출물 경로 결정 (output/ 디렉토리에 저장)
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        stem               = Path(video_path).stem
        features_path      = str(output_dir / f"{stem}_features.json")
        insights_path      = str(output_dir / f"{stem}_insights.json")
        highlights_path    = str(output_dir / f"{stem}_highlights.json")
        report.features_path   = features_path
        report.insights_path   = insights_path
        report.highlights_path = highlights_path

        # ── Stage 1: Feature Extraction ──────────────────────────────
        result_1 = self._run_stage(
            num=1, total=4, label="Feature Extraction",
            sub="extractor.py  |  Whisper + YOLO + MediaPipe",
            fn=lambda: self._stage_extract(video_path, features_path, report),
        )
        report.stage_results.append(result_1)
        if not result_1.success:
            self._print_report(report, pipeline_start)
            sys.exit(1)

        # ── Stage 2: Behavioral Highlights ───────────────────────────
        result_2 = self._run_stage(
            num=2, total=4, label="Behavioral Highlights",
            sub="highlighter.py  |  규칙 기반 행동 특징 추출 (AI 없음)",
            fn=lambda: self._stage_highlight(features_path, highlights_path, report),
        )
        report.stage_results.append(result_2)
        if not result_2.success:
            self._print_report(report, pipeline_start)
            sys.exit(1)

        # ── Stage 3: Knowledge Synthesis ─────────────────────────────
        result_3 = self._run_stage(
            num=3, total=4, label="Knowledge Synthesis",
            sub="synthesizer.py  |  Gemini 2.5 Flash + NCS 매뉴얼 (3단계 판정)",
            fn=lambda: self._stage_synthesize(
                features_path, insights_path, ncs_manual_text, report
            ),
        )
        report.stage_results.append(result_3)
        if not result_3.success:
            self._print_report(report, pipeline_start)
            sys.exit(1)

        # ── Stage 4: Persistence ──────────────────────────────────────
        result_4 = self._run_stage(
            num=4, total=4, label="Persistence",
            sub="loader.py  |  MySQL 단일 트랜잭션 적재",
            fn=lambda: self._stage_load(
                master_name, job_category, ncs_code,
                video_path, features_path, insights_path, highlights_path,
            ),
        )
        report.stage_results.append(result_4)
        if not result_4.success:
            self._print_report(report, pipeline_start)
            sys.exit(1)

        self._print_report(report, pipeline_start)

    # ------------------------------------------------------------------
    # 스테이지 실행기 (공통 래퍼)
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        num:   int,
        total: int,
        label: str,
        sub:   str,
        fn,
    ) -> StageResult:
        """
        단일 스테이지를 실행하고 소요 시간과 성공 여부를 StageResult로 반환한다.
        스테이지 시작/완료 헤더를 터미널에 출력한다.
        """
        print(f"\n{_C.BOLD}{_C.CYAN}[{num}/{total}] {label}{_C.RESET}")
        print(f"  {_C.GRAY}{sub}{_C.RESET}")
        print(f"  {_C.GRAY}{'─' * 52}{_C.RESET}")

        t0 = time.perf_counter()
        try:
            summary = fn()
            elapsed = time.perf_counter() - t0
            print(
                f"  {_C.GREEN}완료{_C.RESET}  "
                f"{_C.WHITE}{summary}{_C.RESET}  "
                f"{_C.GRAY}({elapsed:.1f}s){_C.RESET}"
            )
            return StageResult(
                name=label, success=True, elapsed_sec=elapsed, summary=summary
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(
                f"  {_C.RED}실패{_C.RESET}  "
                f"{_C.YELLOW}{exc}{_C.RESET}  "
                f"{_C.GRAY}({elapsed:.1f}s){_C.RESET}"
            )
            logging.error(f"[{label}] 오류 상세: {exc}", exc_info=True)
            return StageResult(
                name=label, success=False, elapsed_sec=elapsed, summary=str(exc)
            )

    # ------------------------------------------------------------------
    # 스테이지 구현
    # ------------------------------------------------------------------

    def _stage_extract(
        self,
        video_path:    str,
        features_path: str,
        report:        PipelineReport,
    ) -> str:
        """
        Feature Extraction 단계를 실행한다.
        FeatureExtractionWorker를 직접 임포트하여 호출하므로
        별도 프로세스를 생성하지 않고 동일 프로세스 내에서 처리한다.
        """
        from core.extractor import FeatureExtractionWorker
        worker = FeatureExtractionWorker()
        result = worker.run(video_path, features_path)
        report.frame_count = len(result.frame_features)
        return (
            f"{report.frame_count:,}개 프레임 추출  |  "
            f"산출물: {Path(features_path).name}"
        )

    def _stage_highlight(
        self,
        features_path:   str,
        highlights_path: str,
        report:          PipelineReport,
    ) -> str:
        """Behavioral Highlights 단계: 규칙 기반으로 주목 구간을 추출하고 저장한다."""
        from core.highlighter import run_from_file
        highlights = run_from_file(features_path, highlights_path)
        report.highlight_count = len(highlights)
        return (
            f"{report.highlight_count}개 하이라이트 추출  |  "
            f"산출물: {Path(highlights_path).name}"
        )

    def _stage_synthesize(
        self,
        features_path: str,
        insights_path: str,
        ncs_manual_text: str,
        report: PipelineReport,
    ) -> str:
        """
        Knowledge Synthesis 단계를 실행한다.
        --mock 플래그 전달 여부를 KnowledgeSynthesizer에 위임한다.
        """
        from core.synthesizer import KnowledgeSynthesizer, save_insights
        synthesizer = KnowledgeSynthesizer(use_mock=self._mock)
        insights    = synthesizer.synthesize(features_path, ncs_manual_text)
        save_insights(insights, insights_path)
        report.insight_count = len(insights)

        mode_tag = "  [mock]" if self._mock else ""
        return (
            f"{report.insight_count}개 인사이트 도출{mode_tag}  |  "
            f"산출물: {Path(insights_path).name}"
        )

    def _stage_load(
        self,
        master_name:     str,
        job_category:    str,
        ncs_code:        str,
        video_path:      str,
        features_path:   str,
        insights_path:   str,
        highlights_path: str = "",
    ) -> str:
        """
        Persistence 단계를 실행한다.
        --skip-db 또는 --mock 플래그가 있으면 DB 적재를 건너뛴다.
        """
        if self._skip_db:
            return "DB 적재 건너뜀  (--skip-db / --mock 플래그 활성)"

        from core.loader import DatabaseLoader
        loader = DatabaseLoader()
        loader.insert_pipeline_data(
            master_name=master_name,
            job_category=job_category,
            ncs_code=ncs_code,
            video_file_path=video_path,
            features_json_path=features_path,
            insights_json_path=insights_path,
            highlights_json_path=highlights_path or None,
        )
        return "MySQL COMMIT 완료"

    # ------------------------------------------------------------------
    # 터미널 출력 메서드
    # ------------------------------------------------------------------

    @staticmethod
    def _print_banner(video_path: str, master_name: str, job_category: str) -> None:
        """파이프라인 시작 헤더를 출력한다."""
        print()
        print(_C.DLINE)
        print(f"  {_C.BOLD}{_C.WHITE}암묵지 추출 파이프라인{_C.RESET}")
        print(f"  {_C.GRAY}영상      : {Path(video_path).name}{_C.RESET}")
        print(f"  {_C.GRAY}명장      : {master_name}  |  직군: {job_category}{_C.RESET}")
        print(_C.DLINE)

    @staticmethod
    def _print_report(report: PipelineReport, pipeline_start: float) -> None:
        """파이프라인 완료 후 최종 요약 보고서를 출력한다."""
        total_sec  = time.perf_counter() - pipeline_start
        all_ok     = all(r.success for r in report.stage_results)
        status_txt = (
            f"{_C.GREEN}{_C.BOLD}파이프라인 완료{_C.RESET}"
            if all_ok
            else f"{_C.RED}{_C.BOLD}파이프라인 중단{_C.RESET}"
        )

        print()
        print(_C.DLINE)
        print(f"  {status_txt}")
        print(_C.LINE)

        # 스테이지별 결과 요약
        for r in report.stage_results:
            icon   = f"{_C.GREEN}OK{_C.RESET}" if r.success else f"{_C.RED}NG{_C.RESET}"
            timing = f"{_C.GRAY}{r.elapsed_sec:>6.1f}s{_C.RESET}"
            name   = f"{_C.WHITE}{r.name:<25}{_C.RESET}"
            print(f"  [{icon}] {name} {timing}")

        print(_C.LINE)

        # 처리 통계
        if all_ok:
            print(
                f"  {_C.GRAY}총 소요 시간 : {_C.RESET}"
                f"{_C.BOLD}{_C.WHITE}{total_sec:.1f}s{_C.RESET}"
            )
            print(
                f"  {_C.GRAY}처리 프레임  : {_C.RESET}"
                f"{_C.WHITE}{report.frame_count:,}{_C.RESET}"
            )
            print(
                f"  {_C.GRAY}인사이트 수  : {_C.RESET}"
                f"{_C.WHITE}{report.insight_count}{_C.RESET}"
            )
            print(
                f"  {_C.GRAY}하이라이트   : {_C.RESET}"
                f"{_C.WHITE}{report.highlight_count}{_C.RESET}"
            )
            print(
                f"  {_C.GRAY}특징 데이터  : {_C.RESET}"
                f"{_C.CYAN}{report.features_path}{_C.RESET}"
            )
            print(
                f"  {_C.GRAY}인사이트     : {_C.RESET}"
                f"{_C.CYAN}{report.insights_path}{_C.RESET}"
            )
            print(
                f"  {_C.GRAY}하이라이트   : {_C.RESET}"
                f"{_C.CYAN}{report.highlights_path}{_C.RESET}"
            )
        else:
            # 실패한 스테이지의 오류 메시지를 별도 출력
            for r in report.stage_results:
                if not r.success:
                    print(f"  {_C.RED}오류 ({r.name}):{_C.RESET} {r.summary}")

        print(_C.DLINE)
        print()


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.WARNING,  # 오케스트레이터 레벨에서는 WARNING 이상만 출력
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="암묵지 추출 통합 파이프라인 오케스트레이터",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
실행 예시:
  # 실제 API/DB 사용 (전체 파이프라인)
  python pipeline.py sample.mp4 --ncs-file ncs.txt --master-name "김철수" --job-category "자동차 정비"

  # Gemini + DB 없이 흐름 검증 (빠른 테스트)
  python pipeline.py sample.mp4 --ncs-file ncs.txt --master-name "김철수" --job-category "자동차 정비" --mock

  # Gemini는 실제 호출, DB만 스킵
  python pipeline.py sample.mp4 --ncs-file ncs.txt --master-name "김철수" --job-category "자동차 정비" --skip-db
""",
    )

    parser.add_argument(
        "video_path",
        help="처리할 MP4 영상 파일 경로 (예: sample.mp4, /data/master_01.mp4)",
    )

    # NCS 매뉴얼 입력: 셋 중 하나 선택
    #   우선순위: --ncs-text > --ncs-file > DB 자동 조회 (--ncs-code 기반)
    #   아무것도 지정하지 않으면 --ncs-code로 ncs_modules 테이블을 조회한다.
    ncs_group = parser.add_mutually_exclusive_group(required=False)
    ncs_group.add_argument(
        "--ncs-file",
        metavar="PATH",
        help="NCS 표준 매뉴얼 텍스트 파일 경로 (.txt)",
    )
    ncs_group.add_argument(
        "--ncs-text",
        metavar="TEXT",
        help="NCS 표준 매뉴얼 내용을 문자열로 직접 전달",
    )

    parser.add_argument(
        "--master-name",
        required=True,
        metavar="NAME",
        help="명장 식별자 (예: 김철수_001)",
    )
    parser.add_argument(
        "--job-category",
        required=True,
        metavar="CATEGORY",
        help="직군 분류명 (예: 자동차 정비, 전통 도예)",
    )
    parser.add_argument(
        "--ncs-code",
        default=None,
        metavar="CODE",
        help="NCS 능력단위 코드 (미확정 시 생략 가능)",
    )

    # 실행 모드 플래그
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Gemini API + DB 모두 스킵하여 mock 데이터로 파이프라인 흐름만 검증",
    )
    mode_group.add_argument(
        "--skip-db",
        action="store_true",
        default=False,
        help="Gemini는 실제 호출하되 MySQL 적재 단계만 건너뜀",
    )

    args = parser.parse_args()

    # 영상 파일 존재 여부 사전 확인
    if not Path(args.video_path).is_file():
        print(f"\n{_C.RED}오류: 영상 파일을 찾을 수 없습니다 → {args.video_path}{_C.RESET}\n")
        sys.exit(1)

    # NCS 매뉴얼 텍스트 로드 (우선순위: --ncs-text > --ncs-file > DB 자동 조회)
    if args.ncs_text:
        ncs_manual_text = args.ncs_text
    elif args.ncs_file:
        ncs_path = Path(args.ncs_file)
        if not ncs_path.is_file():
            print(f"\n{_C.RED}오류: NCS 매뉴얼 파일을 찾을 수 없습니다 → {args.ncs_file}{_C.RESET}\n")
            sys.exit(1)
        ncs_manual_text = ncs_path.read_text(encoding="utf-8")
    else:
        # --ncs-text / --ncs-file 미지정 시 DB에서 자동 조회
        if not args.ncs_code:
            print(f"\n{_C.RED}오류: --ncs-text, --ncs-file, --ncs-code 중 하나는 반드시 지정해야 합니다.{_C.RESET}\n")
            sys.exit(1)
        print(f"  {_C.CYAN}[NCS] ncs_modules 테이블에서 '{args.ncs_code}' 조회 중...{_C.RESET}")
        try:
            ncs_manual_text = fetch_ncs_text_from_db(args.ncs_code)
            print(f"  {_C.GREEN}[NCS] 매뉴얼 텍스트 로드 완료 ({len(ncs_manual_text)}자){_C.RESET}")
        except Exception as e:
            print(f"\n{_C.RED}오류: DB에서 NCS 매뉴얼 조회 실패 → {e}{_C.RESET}\n")
            sys.exit(1)

    orchestrator = PipelineOrchestrator(
        mock=args.mock,
        skip_db=args.skip_db,
    )
    orchestrator.run(
        video_path=args.video_path,
        ncs_manual_text=ncs_manual_text,
        master_name=args.master_name,
        job_category=args.job_category,
        ncs_code=args.ncs_code or "",
    )


if __name__ == "__main__":
    main()
