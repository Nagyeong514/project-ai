# 암묵지 추출 파이프라인

명장(숙련 기술자) 영상에서 **문서화되지 않은 노하우(암묵지)** 를 AI로 자동 추출하여 데이터베이스에 정형화하는 시스템입니다.

---

## 개요

스마트 글래스(Meta Ray-Ban)로 촬영한 명장의 작업 영상을 입력받아, 음성·시각·동작 데이터를 복합 분석한 뒤 NCS(국가직무능력표준) 매뉴얼과 비교하여 초과 노하우를 자동으로 식별합니다.

---

## 파이프라인 구조

```
[MP4 영상]
    │
    ▼
[Stage 1] Feature Extraction       — Whisper + YOLOv8 + MediaPipe
    │         음성 대사, 공구 탐지, 손 관절 좌표 추출 → output/*_features.json
    ▼
[Stage 2] Behavioral Highlights    — 규칙 기반 (AI 없음)
    │         수치 발화·절차 키워드·손 동작 구간 분류 → output/*_highlights.json
    ▼
[Stage 3] Knowledge Synthesis      — Gemini 2.5 Flash
    │         NCS 매뉴얼 vs 영상 데이터 비교, 3단계 판정 → output/*_insights.json
    ▼
[Stage 4] Persistence              — MySQL 단일 트랜잭션
              5개 테이블에 일괄 적재
```

### 3단계 판정 기준 (Stage 3)

| 판정 | 의미 |
|------|------|
| `SUCCESS` | NCS 기준을 초과하는 정량적 노하우 확인 → 암묵지로 저장 |
| `STANDARD_COMPLIANCE` | 명장 행위가 NCS와 100% 일치, 초과 노하우 없음 |
| `MISMATCH` | 영상 도메인이 해당 NCS 능력단위와 무관 |

---

## 기술 스택

| 역할 | 모델 / 라이브러리 |
|------|------------------|
| 음성 인식 (STT) | OpenAI Whisper (base) |
| 객체 탐지 | Ultralytics YOLOv8n (COCO) |
| 손 관절 추적 | MediaPipe Tasks API — HandLandmarker |
| 암묵지 분석 | Google Gemini 2.5 Flash (`temperature=0.0`) |
| 데이터베이스 | MySQL 8.0 |
| 런타임 | Python 3.10 64-bit |

---

## 디렉토리 구조

```
project_ai/
├── pipeline.py          # 실행 진입점
├── requirements.txt     # Python 패키지 목록
├── .env.example         # 환경 변수 템플릿
│
├── core/                # 파이프라인 핵심 모듈
│   ├── extractor.py     # Stage 1: 특징 추출
│   ├── highlighter.py   # Stage 2: 행동 하이라이트
│   ├── synthesizer.py   # Stage 3: 지식 합성
│   └── loader.py        # Stage 4: DB 적재
│
├── db/
│   └── init.sql         # DB 스키마 초기화 SQL
│
├── models/              # AI 모델 바이너리 (별도 다운로드, git 제외)
│   └── hand_landmarker.task
│
├── output/              # 파이프라인 산출물 JSON (git 제외)
└── web/                 # 웹 인터페이스
    └── index.html
```

---

## 설치 및 실행

### 1. 환경 요구사항

- **Python 3.10 64-bit** 필수 (`mediapipe`가 32-bit 미지원)
- MySQL 8.0
- ffmpeg (Whisper 오디오 추출에 필요)

### 2. 패키지 설치

```bash
py -3.10-64 -m pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 항목을 채웁니다:

```env
GEMINI_API_KEY=your_gemini_api_key
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=tacit_knowledge_db
```

### 4. DB 초기화

MySQL Workbench 또는 CLI에서 실행:

```sql
source db/init.sql
```

### 5. AI 모델 다운로드

`models/` 디렉토리에 MediaPipe 핸드랜드마크 모델을 저장합니다:

```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'models/hand_landmarker.task'
)"
```

> YOLOv8 모델(`yolov8n.pt`)은 최초 실행 시 자동 다운로드됩니다.

### 6. 파이프라인 실행

```bash
# ffmpeg PATH 등록 (Windows)
$env:PATH = "C:\경로\ffmpeg\bin;$env:PATH"

# 파이프라인 실행
py -3.10-64 pipeline.py 영상파일.mp4 \
    --master-name "홍길동_명장" \
    --job-category "자동차 정비" \
    --ncs-code "LM1506030201_24v5"
```

#### 실행 옵션

| 옵션 | 설명 |
|------|------|
| `--ncs-code` | DB에서 NCS 매뉴얼 자동 조회 |
| `--ncs-file` | NCS 매뉴얼 텍스트 파일 직접 지정 |
| `--skip-db` | Gemini는 호출하되 DB 저장 생략 (테스트용) |
| `--mock` | AI·DB 모두 생략, 파이프라인 흐름만 확인 |

---

## 데이터베이스 스키마

| 테이블 | 설명 |
|--------|------|
| `video_metadata` | 영상 및 명장 식별 정보 |
| `frame_features` | 프레임별 YOLO + MediaPipe 원시 데이터 |
| `behavioral_highlights` | 수치 발화·절차 키워드 등 주목 구간 |
| `tacit_knowledge_insights` | Gemini 암묵지 판정 결과 |
| `ncs_modules` | NCS 표준 매뉴얼 마스터 데이터 |

---

## NCS 마스터 데이터 등록

새로운 NCS 능력단위를 추가하려면:

```sql
INSERT INTO ncs_modules (ncs_code, module_name, standard_manual_text)
VALUES ('LM1506030201_24v5', '엔진본체 정비', '매뉴얼 전문 텍스트...');
```

등록 후 `--ncs-code` 인자만으로 자동 조회됩니다.
