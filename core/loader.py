"""
Persistence Phase - 데이터베이스 적재 워커
-------------------------------------------
파이프라인 위치: Knowledge Synthesis → [Persistence Phase]

역할:
    extractor.py가 생성한 특징 데이터(*_features.json)와
    synthesizer.py가 생성한 인사이트 데이터(*_insights.json)를 읽어
    MySQL 3개 테이블에 단일 트랜잭션으로 안전하게 적재한다.

    적재 순서 (FK 제약 준수):
        1. video_metadata           → 마스터 레코드 INSERT, video_id 획득
        2. frame_features           → video_id 매핑 후 BULK INSERT
        3. tacit_knowledge_insights → video_id 매핑 후 INSERT

실행 예시:
    python loader.py \\
        --master-name "김철수" \\
        --job-category "자동차 정비" \\
        --ncs-code "LM1501010106_19v4" \\
        --video-path "/data/videos/master_01.mp4" \\
        --features master_01_features.json \\
        --insights master_01_insights.json

의존 패키지:
    pip install pymysql python-dotenv
"""

import os
import json
import logging
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pymysql
import pymysql.cursors
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 벌크 인서트 1회 배치 크기.
# 너무 작으면 네트워크 왕복 횟수가 증가하고, 너무 크면 단일 패킷 크기가 MySQL의
# max_allowed_packet 한계를 초과할 수 있으므로 500을 기본값으로 사용한다.
BULK_BATCH_SIZE = 500


# ──────────────────────────────────────────────────────────────
# DB 접속 설정 로더
# ──────────────────────────────────────────────────────────────

def _load_db_config() -> dict:
    """
    .env 파일에서 MySQL 접속 정보를 로드하여 pymysql.connect() 인자 dict를 반환한다.
    필수 환경 변수가 하나라도 없으면 즉시 예외를 발생시켜 조용한 실패를 방지한다.

    필수 .env 항목:
        DB_HOST     : MySQL 서버 호스트 (예: localhost, 192.168.1.100)
        DB_USER     : DB 사용자 이름
        DB_PASSWORD : DB 비밀번호
        DB_NAME     : 대상 데이터베이스 이름

    선택 .env 항목:
        DB_PORT     : MySQL 포트 (미설정 시 기본값 3306 사용)
    """
    required_keys = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing = [k for k in required_keys if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f".env 파일에 다음 항목이 없습니다: {', '.join(missing)}\n"
            "프로젝트 루트의 .env 파일을 확인하세요. (.env.example 참고)"
        )

    return {
        "host":        os.environ["DB_HOST"],
        "port":        int(os.environ.get("DB_PORT", 3306)),
        "user":        os.environ["DB_USER"],
        "password":    os.environ["DB_PASSWORD"],
        "database":    os.environ["DB_NAME"],
        "charset":     "utf8mb4",
        # DictCursor: 쿼리 결과를 컬럼명 기반 dict로 반환 (디버깅 용이)
        "cursorclass": pymysql.cursors.DictCursor,
    }


# ──────────────────────────────────────────────────────────────
# DatabaseLoader 클래스
# ──────────────────────────────────────────────────────────────

class DatabaseLoader:
    """
    Persistence Phase 핵심 클래스.

    extractor.py와 synthesizer.py의 출력 JSON을 읽어
    MySQL 3개 테이블에 단일 트랜잭션으로 안전하게 적재한다.

    트랜잭션 보장:
        - 3개 테이블 모두 INSERT 성공 → COMMIT
        - 어느 단계에서든 예외 발생   → ROLLBACK 후 예외 재발생
          (부분 적재로 인한 데이터 불일치 상태를 원천 차단)

    사용 예시:
        loader = DatabaseLoader()
        loader.insert_pipeline_data(
            master_name="김철수",
            job_category="자동차 정비",
            ncs_code="LM1501010106_19v4",
            video_file_path="/data/videos/master_01.mp4",
            features_json_path="master_01_features.json",
            insights_json_path="master_01_insights.json",
        )
    """

    def __init__(self):
        # 초기화 시 접속 정보를 로드하여 환경 변수 누락을 조기에 감지한다.
        self._db_config = _load_db_config()

    @contextmanager
    def _get_connection(self):
        """
        pymysql 커넥션을 컨텍스트 매니저로 제공한다.

        autocommit 기본값은 False이므로 명시적 commit/rollback으로만 트랜잭션이 확정된다.
        with 블록이 어떤 이유로든 종료되면 connection을 자동으로 닫아 커넥션 누수를 방지한다.
        """
        conn = pymysql.connect(**self._db_config)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 퍼블릭 인터페이스
    # ------------------------------------------------------------------

    def insert_pipeline_data(
        self,
        master_name:           str,
        job_category:          str,
        ncs_code:              str,
        video_file_path:       str,
        features_json_path:    str,
        insights_json_path:    str,
        highlights_json_path:  Optional[str] = None,
    ) -> None:
        """
        파이프라인 전체 출력 데이터를 MySQL에 단일 트랜잭션으로 적재한다.

        적재 순서 (FK 제약 준수):
            1. video_metadata           : 마스터 레코드 INSERT → AUTO_INCREMENT video_id 획득
            2. frame_features           : video_id 매핑 후 BULK INSERT (BULK_BATCH_SIZE 단위)
            3. tacit_knowledge_insights : video_id 매핑 후 INSERT
            4. behavioral_highlights    : video_id 매핑 후 INSERT (경로 제공 시)

        트랜잭션 정책:
            - 전 단계 성공 시 COMMIT
            - 어느 단계에서든 예외 발생 시 ROLLBACK 후 예외 재발생

        Args:
            master_name:           명장 식별자 (예: "김철수_001")
            job_category:          직군 분류명 (예: "자동차 정비")
            ncs_code:              NCS 능력단위 코드 (미확정 시 빈 문자열 또는 None 허용)
            video_file_path:       원본 영상 스토리지 경로 (로컬 경로 또는 URI)
            features_json_path:    extractor.py 출력 *_features.json 경로
            insights_json_path:    synthesizer.py 출력 *_insights.json 경로
            highlights_json_path:  highlighter.py 출력 *_highlights.json 경로 (선택)
        """
        logger.info("=" * 60)
        logger.info("[Loader] Persistence Phase 시작")
        logger.info("=" * 60)

        frame_rows, insight_rows = self._load_source_files(
            features_json_path, insights_json_path
        )

        highlight_rows = []
        if highlights_json_path and Path(highlights_json_path).is_file():
            highlight_rows = self._load_highlights(highlights_json_path)

        with self._get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    # --- Step 1: video_metadata INSERT ---
                    video_id = self._insert_video_metadata(
                        cursor, master_name, job_category, ncs_code, video_file_path
                    )

                    # --- Step 2: frame_features BULK INSERT ---
                    self._bulk_insert_frame_features(cursor, video_id, frame_rows)

                    # --- Step 3: tacit_knowledge_insights INSERT ---
                    self._insert_insights(cursor, video_id, insight_rows)

                    # --- Step 4: behavioral_highlights INSERT ---
                    if highlight_rows:
                        self._insert_highlights(cursor, video_id, highlight_rows)

                conn.commit()
                logger.info(f"[Loader] 트랜잭션 COMMIT 완료 (video_id={video_id})")

            except Exception as exc:
                conn.rollback()
                logger.error(f"[Loader] 오류 발생 → 트랜잭션 ROLLBACK 실행: {exc}")
                raise

        logger.info(
            f"[Loader] 적재 요약: video_id={video_id}, "
            f"frame_features={len(frame_rows)}행, "
            f"insights={len(insight_rows)}행, "
            f"highlights={len(highlight_rows)}행"
        )
        logger.info("=" * 60)
        logger.info("[Loader] Persistence Phase 완료")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 내부 구현 메서드 - 소스 파일 로드
    # ------------------------------------------------------------------

    @staticmethod
    def _load_source_files(
        features_json_path: str,
        insights_json_path: str,
    ) -> tuple:
        """
        features JSON과 insights JSON을 읽어 DB 인서트용 튜플 리스트로 변환한다.

        frame_features 행 튜플 구조 (video_id는 INSERT 시점에 결합):
            (timestamp_sec, detected_tools_json_str, hand_movement_vector_json_str)

        tacit_knowledge_insights 행 튜플 구조:
            (step_number, standard_manual_text, tacit_knowledge_description)

        detected_tools와 hand_movement_vector는 Python dict/list를 JSON 문자열로
        직렬화하여 MySQL JSON 컬럼에 저장한다.
        """
        # features JSON 로드 및 검증
        feat_path = Path(features_json_path)
        if not feat_path.is_file():
            raise FileNotFoundError(
                f"features JSON 파일을 찾을 수 없습니다: {features_json_path}\n"
                "extractor.py를 먼저 실행하세요."
            )

        with open(feat_path, "r", encoding="utf-8") as f:
            features_data = json.load(f)

        raw_frames = features_data.get("frame_features", [])
        if not raw_frames:
            raise ValueError(
                f"'frame_features' 배열이 비어 있습니다: {features_json_path}"
            )

        frame_rows = [
            (
                frame["timestamp_sec"],
                # dict/list → JSON 문자열 변환 (MySQL JSON 컬럼 저장용)
                json.dumps(frame.get("detected_tools"),       ensure_ascii=False),
                json.dumps(frame.get("hand_movement_vector"), ensure_ascii=False),
            )
            for frame in raw_frames
        ]
        logger.info(f"[Loader] frame_features 행 변환 완료: {len(frame_rows)}개")

        # insights JSON 로드 및 검증
        ins_path = Path(insights_json_path)
        if not ins_path.is_file():
            raise FileNotFoundError(
                f"insights JSON 파일을 찾을 수 없습니다: {insights_json_path}\n"
                "synthesizer.py를 먼저 실행하세요."
            )

        with open(ins_path, "r", encoding="utf-8") as f:
            insights_data = json.load(f)

        if not isinstance(insights_data, list) or not insights_data:
            raise ValueError(
                f"insights JSON이 비어 있거나 배열 형식이 아닙니다: {insights_json_path}"
            )

        insight_rows = [
            (
                int(item["step_number"]),
                item.get("standard_manual_text"),          # NULL 허용
                item["tacit_knowledge_description"],
            )
            for item in insights_data
        ]
        logger.info(f"[Loader] tacit_knowledge_insights 행 변환 완료: {len(insight_rows)}개")

        return frame_rows, insight_rows

    # ------------------------------------------------------------------
    # 내부 구현 메서드 - DB INSERT
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_video_metadata(
        cursor,
        master_name:     str,
        job_category:    str,
        ncs_code:        str,
        video_file_path: str,
    ) -> int:
        """
        video_metadata 테이블에 마스터 레코드를 INSERT하고
        AUTO_INCREMENT로 발급된 video_id를 반환한다.

        이 video_id가 이후 두 테이블의 FK 컬럼에 매핑되므로
        반드시 가장 먼저 실행되어야 한다.
        """
        sql = """
            INSERT INTO video_metadata
                (master_name, job_category, ncs_code, file_path)
            VALUES
                (%s, %s, %s, %s)
        """
        # ncs_code가 빈 문자열인 경우 None으로 변환하여 DB에 NULL로 저장
        cursor.execute(sql, (master_name, job_category, ncs_code or None, video_file_path))
        video_id = cursor.lastrowid  # AUTO_INCREMENT로 발급된 PK 값
        logger.info(f"[Loader] video_metadata INSERT 완료 (video_id={video_id})")
        return video_id

    @staticmethod
    def _bulk_insert_frame_features(
        cursor,
        video_id:   int,
        frame_rows: list,
    ) -> None:
        """
        frame_features 테이블에 프레임 데이터를 BULK_BATCH_SIZE 단위로 벌크 인서트한다.

        단순 루프 INSERT vs executemany 배치 INSERT 성능 비교:
            - 단순 루프    : 프레임 1개당 네트워크 왕복 1회
                             → 10,000 프레임 = 10,000회 왕복
            - executemany : BULK_BATCH_SIZE(500)개를 1회 왕복으로 처리
                             → 10,000 프레임 / 500 = 20회 왕복으로 단축

        video_id는 _load_source_files 단계에서 아직 없으므로
        이 메서드에서 실제 발급된 video_id를 각 행과 결합한다.
        """
        sql = """
            INSERT INTO frame_features
                (video_id, frame_timestamp, detected_tools, hand_movement_vector)
            VALUES
                (%s, %s, %s, %s)
        """
        total = len(frame_rows)

        # video_id를 각 행의 첫 번째 컬럼으로 결합하여 최종 튜플 생성
        rows_with_id = [
            (video_id, row[0], row[1], row[2])
            for row in frame_rows
        ]

        # BULK_BATCH_SIZE 단위로 슬라이싱하여 executemany 반복 호출
        for batch_start in range(0, total, BULK_BATCH_SIZE):
            batch     = rows_with_id[batch_start : batch_start + BULK_BATCH_SIZE]
            batch_end = min(batch_start + BULK_BATCH_SIZE, total)
            cursor.executemany(sql, batch)
            logger.info(
                f"[Loader] frame_features 배치 인서트: "
                f"{batch_start + 1}~{batch_end} / {total}"
            )

        logger.info(f"[Loader] frame_features BULK INSERT 완료: 총 {total}개")

    @staticmethod
    def _load_highlights(highlights_json_path: str) -> list:
        """highlights JSON을 읽어 DB 인서트용 튜플 리스트로 변환한다."""
        with open(highlights_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = [
            (
                float(h['timestamp_sec']),
                h['highlight_type'],
                h['trigger_reason'],
                h.get('feature_description', ''),
                h.get('speech_quote'),
                json.dumps(h.get('detected_tools', []), ensure_ascii=False),
                int(bool(h.get('hand_active', False))),
            )
            for h in data
        ]
        logger.info(f"[Loader] behavioral_highlights 행 변환 완료: {len(rows)}개")
        return rows

    @staticmethod
    def _insert_highlights(cursor, video_id: int, highlight_rows: list) -> None:
        """behavioral_highlights 테이블에 하이라이트 데이터를 INSERT한다."""
        sql = """
            INSERT INTO behavioral_highlights
                (video_id, timestamp_sec, highlight_type, trigger_reason,
                 feature_description, speech_quote, detected_tools, hand_active)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        rows_with_id = [(video_id, *row) for row in highlight_rows]
        cursor.executemany(sql, rows_with_id)
        logger.info(f"[Loader] behavioral_highlights INSERT 완료: {len(rows_with_id)}개")

    @staticmethod
    def _insert_insights(
        cursor:       object,
        video_id:     int,
        insight_rows: list,
    ) -> None:
        """
        tacit_knowledge_insights 테이블에 Gemini 분석 인사이트를 INSERT한다.

        insight_embedding 컬럼은 현재 단계에서 NULL로 적재한다.
        추후 임베딩 모델(OpenAI Embeddings 등) 연동 시 별도 UPDATE 배치 또는
        이 메서드를 확장하여 벡터 데이터를 채운다.
        """
        sql = """
            INSERT INTO tacit_knowledge_insights
                (video_id, step_number, standard_manual_text,
                 tacit_knowledge_description, insight_embedding)
            VALUES
                (%s, %s, %s, %s, NULL)
        """
        rows_with_id = [
            (video_id, row[0], row[1], row[2])
            for row in insight_rows
        ]
        cursor.executemany(sql, rows_with_id)
        logger.info(
            f"[Loader] tacit_knowledge_insights INSERT 완료: {len(rows_with_id)}개"
        )


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
        description="암묵지 추출 시스템 - Persistence Phase DB 적재 워커",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--master-name",
        required=True,
        help="명장 식별자 (예: 김철수_001)",
    )
    parser.add_argument(
        "--job-category",
        required=True,
        help="직군 분류명 (예: 자동차 정비)",
    )
    parser.add_argument(
        "--ncs-code",
        default=None,
        help="NCS 능력단위 코드 (미확정 시 생략 가능)",
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="원본 영상 스토리지 경로 (로컬 절대 경로 또는 오브젝트 스토리지 URI)",
    )
    parser.add_argument(
        "--features",
        required=True,
        metavar="PATH",
        help="extractor.py 출력 *_features.json 파일 경로",
    )
    parser.add_argument(
        "--insights",
        required=True,
        metavar="PATH",
        help="synthesizer.py 출력 *_insights.json 파일 경로",
    )
    args = parser.parse_args()

    loader = DatabaseLoader()
    loader.insert_pipeline_data(
        master_name=args.master_name,
        job_category=args.job_category,
        ncs_code=args.ncs_code or "",
        video_file_path=args.video_path,
        features_json_path=args.features,
        insights_json_path=args.insights,
    )


if __name__ == "__main__":
    main()
