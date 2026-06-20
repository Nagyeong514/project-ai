-- ============================================================
-- 프로젝트 암묵지 추출 시스템 - 데이터베이스 초기화 스크립트
-- 참조 문서: 파이프라인 구조 설계안 Section 4 - DB 구조 설계안
-- 대상 DBMS : MySQL 8.0 이상
-- 문자셋    : utf8mb4 (한글 및 이모지 포함 4바이트 유니코드 완전 지원)
-- 작성 기준 : InnoDB 스토리지 엔진, 외래키(FK) + CASCADE 전략 적용
-- ============================================================

SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- 외래키 제약을 일시 비활성화하여 DROP 순서 의존성 제거
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS tacit_knowledge_insights;
DROP TABLE IF EXISTS frame_features;
DROP TABLE IF EXISTS video_metadata;

SET FOREIGN_KEY_CHECKS = 1;


-- ============================================================
-- [테이블 1] 수집 영상 메타데이터 테이블
-- 역할: 명장 촬영 원본 영상의 기본 정보 및 직군 분류 정보를 관리하는
--       마스터 테이블. frame_features 및 tacit_knowledge_insights
--       두 테이블이 이 테이블을 부모로 하는 FK 관계를 가짐.
-- ============================================================
CREATE TABLE video_metadata (
    video_id      INT UNSIGNED  NOT NULL AUTO_INCREMENT
                  COMMENT '영상 고유 식별자 (PK, 참조 영상 수가 수백만 건을 넘지 않을 것으로 예측되어 INT 사용)',

    master_name   VARCHAR(100)  NOT NULL
                  COMMENT '촬영 대상 명장의 이름 또는 내부 식별 코드 (예: 김철수_001)',

    job_category  VARCHAR(100)  NOT NULL
                  COMMENT '명장의 직군 분류명 (예: 자동차 정비, 전통 도예, 정밀기계 수리, 바리스타)',

    ncs_code      VARCHAR(50)   NULL
                  COMMENT '국가직무능력표준 능력단위 코드 - NCS 표준 매뉴얼과의 매핑 키. 미확정 시 NULL 허용',

    file_path     VARCHAR(1000) NOT NULL
                  COMMENT '원본 MP4 영상이 저장된 경로 (로컬 절대 경로 또는 오브젝트 스토리지 URI, 예: s3://bucket/path/video.mp4)',

    created_at    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
                  COMMENT '레코드 생성 일시 (영상 최초 인입 시각)',

    updated_at    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                  COMMENT '레코드 최종 수정 일시 (메타데이터 재분류 또는 ncs_code 갱신 시 자동 업데이트)',

    PRIMARY KEY (video_id)
)
ENGINE  = InnoDB
DEFAULT CHARSET  = utf8mb4
COLLATE  = utf8mb4_unicode_ci
COMMENT = '수집 영상 메타데이터 테이블 - 명장 촬영 원본 MP4의 식별 정보, 직군 분류, NCS 코드, 스토리지 경로를 통합 관리하는 마스터 테이블';


-- ============================================================
-- [테이블 2] 프레임별 특징 데이터 테이블
-- 역할: AI 모델(YOLO, MediaPipe)이 영상을 프레임 단위로 분석하여
--       추출한 공구 탐지 결과 및 손 관절 벡터를 타임스탬프 기준으로 저장.
--       단일 영상당 수천~수만 건의 레코드가 생성될 수 있으므로 PK를 BIGINT로 설계.
-- FK 전략:
--   - video_metadata 삭제 시 관련 프레임 데이터 전체 자동 삭제 (ON DELETE CASCADE)
--   - video_metadata의 video_id 변경 시 자동 동기화 (ON UPDATE CASCADE)
-- ============================================================
CREATE TABLE frame_features (
    feature_id             BIGINT UNSIGNED   NOT NULL AUTO_INCREMENT
                           COMMENT '프레임 특징 레코드 고유 식별자 (PK, 대용량 누적을 고려하여 BIGINT UNSIGNED 사용)',

    video_id               INT UNSIGNED      NOT NULL
                           COMMENT '원본 영상 참조 키 (FK → video_metadata.video_id)',

    frame_timestamp        DECIMAL(12, 3)    NOT NULL
                           COMMENT '프레임의 영상 내 시간 위치 (단위: 초, 소수점 3자리로 밀리초까지 표현. 예: 123.456)',

    detected_tools         JSON              NULL
                           COMMENT 'YOLO 객체 탐지 결과 - 해당 프레임에서 식별된 공구 및 부품 목록. JSON 배열로 저장 (예: {"tools": ["Spanner", "Bolt"], "bounding_boxes": [...]})',

    hand_movement_vector   JSON              NULL
                           COMMENT 'MediaPipe 손 관절 추적 결과 - 양손 21개 랜드마크의 3D 좌표(x, y, z)와 관절 각도 등 정형 수치 데이터. JSON 객체로 저장 (예: {"left_hand": [[x,y,z], ...], "right_hand": [[x,y,z], ...]})',

    created_at             TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP
                           COMMENT '레코드 생성 일시 (피처 추출 파이프라인 처리 완료 시각)',

    PRIMARY KEY (feature_id),

    -- 빈번한 조회 패턴에 대한 인덱스 정의
    INDEX idx_frame_features_video_id      (video_id)
          COMMENT '특정 영상의 전체 프레임 조회 시 풀스캔 방지용 인덱스',

    INDEX idx_frame_features_timestamp     (frame_timestamp)
          COMMENT '시간축 기반 범위 조회(특정 구간 내 프레임 탐색) 성능 최적화 인덱스',

    CONSTRAINT fk_frame_features_video_id
        FOREIGN KEY (video_id)
        REFERENCES video_metadata (video_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
)
ENGINE  = InnoDB
DEFAULT CHARSET  = utf8mb4
COLLATE  = utf8mb4_unicode_ci
COMMENT = '프레임별 특징 데이터 테이블 - YOLO 공구 탐지 결과와 MediaPipe 손 동작 벡터를 타임스탬프 단위로 저장. video_metadata와 N:1 관계';


-- ============================================================
-- [테이블 3] 추출된 암묵지 인사이트 테이블
-- 역할: Gemini VLM이 NCS 표준 매뉴얼(형식지)과 명장의 실제 행동을
--       대조 분석하여 도출한 핵심 암묵지 노하우 및 임베딩 벡터를 저장.
--       향후 신입 사원 대상 유사도 기반 교육 추천 시스템의 핵심 데이터 소스.
-- FK 전략:
--   - video_metadata 삭제 시 관련 인사이트 전체 자동 삭제 (ON DELETE CASCADE)
--   - video_metadata의 video_id 변경 시 자동 동기화 (ON UPDATE CASCADE)
-- ============================================================
CREATE TABLE tacit_knowledge_insights (
    insight_id                   BIGINT UNSIGNED  NOT NULL AUTO_INCREMENT
                                 COMMENT '암묵지 인사이트 레코드 고유 식별자 (PK)',

    video_id                     INT UNSIGNED     NOT NULL
                                 COMMENT '원본 영상 참조 키 (FK → video_metadata.video_id)',

    step_number                  SMALLINT UNSIGNED NOT NULL
                                 COMMENT '영상 내 공정 단계 번호 (1부터 시작하는 순서 인덱스. 예: 1=부품 분리, 2=세척, 3=조립)',

    standard_manual_text         TEXT             NULL
                                 COMMENT 'NCS 표준 매뉴얼 해당 공정 단계의 기준 지침 원문 (형식지 원본, Gemini 분석의 비교 기준으로 사용됨)',

    tacit_knowledge_description  TEXT             NOT NULL
                                 COMMENT 'Gemini VLM이 표준 매뉴얼과 명장 행동 비교 분석을 통해 정제한 암묵지 노하우 설명 (매뉴얼에 기재되지 않은 핵심 차이점 및 숙련 팁)',

    -- [벡터 임베딩 컬럼 설계 노트]
    -- 현재 MySQL 환경에서는 tacit_knowledge_description 텍스트의 임베딩 벡터를
    -- JSON 배열(부동소수점 배열) 형태로 저장함.
    -- 저장 형식 예시: [0.0231, -0.4512, 0.8823, ...]
    --   - OpenAI text-embedding-3-small 기준: 1536차원
    --   - OpenAI text-embedding-3-large 기준: 3072차원
    --
    -- 향후 확장 경로 (우선순위 순):
    --   1. MySQL 9.0+ VECTOR 타입 적용 (네이티브 ANN 인덱스 지원 시)
    --   2. PostgreSQL + pgvector 확장으로 마이그레이션 (ivfflat / hnsw 인덱스 활용)
    --   3. Qdrant, Weaviate, Pinecone 등 전용 벡터 DB에 insight_id를 키로 별도 저장
    --      후 이 컬럼을 NULL 처리하거나 제거
    insight_embedding            JSON             NULL
                                 COMMENT '암묵지 설명 텍스트의 임베딩 벡터 (JSON 부동소수점 배열). 사용자 의미 검색 및 신입 사원 교육용 유사도 추천에 활용. 추후 pgvector 또는 전용 벡터 DB로 확장 예정',

    created_at                   TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP
                                 COMMENT '레코드 생성 일시 (Gemini 분석 결과 최초 적재 시각)',

    updated_at                   TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                                 COMMENT '레코드 최종 수정 일시 (Gemini 재분석, 임베딩 모델 교체 등으로 인한 갱신 시 자동 업데이트)',

    PRIMARY KEY (insight_id),

    INDEX idx_insights_video_id    (video_id)
          COMMENT '특정 영상의 전체 인사이트 조회 시 풀스캔 방지용 인덱스',

    INDEX idx_insights_step_number (step_number)
          COMMENT '공정 단계 번호 기반 필터링 조회 성능 최적화 인덱스',

    CONSTRAINT fk_insights_video_id
        FOREIGN KEY (video_id)
        REFERENCES video_metadata (video_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
)
ENGINE  = InnoDB
DEFAULT CHARSET  = utf8mb4
COLLATE  = utf8mb4_unicode_ci
COMMENT = '추출된 암묵지 인사이트 테이블 - Gemini VLM이 NCS 표준 매뉴얼과 명장 행동을 대조하여 정제한 암묵지 노하우 및 의미 검색용 임베딩 벡터를 저장. video_metadata와 N:1 관계';
