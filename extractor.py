"""
Feature Extraction Phase Worker
--------------------------------
파이프라인 위치: Data Ingestion → [Feature Extraction] → Knowledge Synthesis → DB 적재

입력 : MP4 영상 파일 경로
출력 : <영상명>_features.json  (Knowledge Synthesis Phase의 Gemini 입력 데이터)

실행 예시:
    python extractor.py sample.mp4
    python extractor.py sample.mp4 --output result.json

의존 패키지 (실제 모델 사용 시):
    pip install opencv-python openai-whisper ultralytics mediapipe
"""

import os
import json
import logging
import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2  # pip install opencv-python

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 데이터 모델
# ──────────────────────────────────────────────────────────────

@dataclass
class AudioSegment:
    """Whisper가 반환하는 단일 음성 구간 텍스트 및 타임스탬프."""
    start_sec: float
    end_sec: float
    text: str


@dataclass
class BoundingBox:
    """YOLO가 탐지한 단일 객체(공구/부품) 정보."""
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class HandLandmarks:
    """MediaPipe가 추출한 단일 손의 21개 관절 랜드마크."""
    handedness: str          # "Left" 또는 "Right"
    landmarks: list          # shape: [21, 3] — 각 원소는 [x, y, z] 정규화 좌표


@dataclass
class FrameFeature:
    """단일 프레임에서 추출된 비전 특징 데이터 (frame_features 테이블 1행에 대응)."""
    timestamp_sec: float
    detected_tools: list     # list[BoundingBox]
    left_hand: Optional[HandLandmarks]
    right_hand: Optional[HandLandmarks]


@dataclass
class IntegratedEntry:
    """오디오 컨텍스트와 비전 특징이 타임스탬프 기준으로 병합된 통합 항목."""
    timestamp_sec: float
    audio_context: Optional[str]
    detected_tools: list
    hand_movement_vector: dict


@dataclass
class ExtractionResult:
    """Feature Extraction Phase 전체 출력 결과."""
    file_path: str
    duration_sec: float
    fps: float
    total_frames: int
    audio_transcript: list   # list[AudioSegment]
    frame_features: list     # list[FrameFeature]
    integrated_timeline: list  # list[IntegratedEntry] — Gemini에 직접 전달되는 최종 데이터


# ──────────────────────────────────────────────────────────────
# Mock AI 모델 클래스
# 실제 배포 시 각 클래스 내 주석 처리된 코드 블록으로 교체
# ──────────────────────────────────────────────────────────────

class WhisperModel:
    """
    Whisper API를 통한 음성-텍스트 변환 모델.

    실제 구현 시 교체 코드:
        import whisper
        self._model = whisper.load_model("medium")

        def transcribe(self, audio_path: str) -> list[AudioSegment]:
            result = self._model.transcribe(audio_path, language="ko")
            return [
                AudioSegment(
                    start_sec=seg["start"],
                    end_sec=seg["end"],
                    text=seg["text"].strip(),
                )
                for seg in result["segments"]
            ]

    또는 OpenAI API 사용 시:
        from openai import OpenAI
        client = OpenAI()
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="verbose_json"
            )
    """

    def __init__(self):
        logger.info("[Whisper] 모델 초기화 완료 (mock)")

    def transcribe(self, audio_path: str) -> list:
        logger.info(f"[Whisper] 음성 변환 중: {audio_path} (mock)")
        return [
            AudioSegment(start_sec=0.0,  end_sec=3.5,  text="먼저 엔진 커버를 분리합니다."),
            AudioSegment(start_sec=3.5,  end_sec=8.2,  text="볼트를 풀 때는 반시계 방향으로 힘을 줍니다."),
            AudioSegment(start_sec=8.2,  end_sec=14.0, text="이 각도에서 스패너를 잡는 게 핵심 포인트입니다."),
            AudioSegment(start_sec=14.0, end_sec=20.0, text="손목을 고정하고 팔꿈치로 회전력을 전달하세요."),
        ]


class YoloModel:
    """
    YOLO v8/v10을 통한 공구 및 부품 객체 탐지 모델.

    실제 구현 시 교체 코드:
        from ultralytics import YOLO
        self._model = YOLO("best.pt")  # 공구 탐지 커스텀 학습 가중치

        def detect(self, frame) -> list[BoundingBox]:
            results = self._model(frame)[0]
            return [
                BoundingBox(
                    label=results.names[int(box.cls)],
                    confidence=round(float(box.conf), 4),
                    x1=int(box.xyxy[0][0]), y1=int(box.xyxy[0][1]),
                    x2=int(box.xyxy[0][2]), y2=int(box.xyxy[0][3]),
                )
                for box in results.boxes
            ]
    """

    _TOOL_POOL = [
        ("Spanner", 0.92), ("Bolt", 0.87), ("Screwdriver", 0.83),
        ("Wrench", 0.91), ("Socket", 0.79), ("Pliers", 0.85),
    ]

    def __init__(self):
        logger.info("[YOLO] 모델 로드 완료 (mock)")

    def detect(self, frame) -> list:
        k = random.randint(0, 2)
        sampled = random.sample(self._TOOL_POOL, k=k)
        return [
            BoundingBox(
                label=label, confidence=conf,
                x1=random.randint(50, 300), y1=random.randint(50, 300),
                x2=random.randint(350, 600), y2=random.randint(350, 600),
            )
            for label, conf in sampled
        ]


class MediaPipeModel:
    """
    MediaPipe Hands를 통한 손 관절 랜드마크 추출 모델.
    21개 랜드마크 × (x, y, z) 정규화 좌표를 반환.

    실제 구현 시 교체 코드:
        import mediapipe as mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
        )

        def extract_landmarks(self, frame) -> tuple[Optional[HandLandmarks], Optional[HandLandmarks]]:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(frame_rgb)
            left_hand, right_hand = None, None
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    side = handedness.classification[0].label  # "Left" or "Right"
                    parsed = HandLandmarks(
                        handedness=side,
                        landmarks=[[round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]
                                   for lm in hand_landmarks.landmark],
                    )
                    if side == "Left":
                        left_hand = parsed
                    else:
                        right_hand = parsed
            return left_hand, right_hand
    """

    def __init__(self):
        logger.info("[MediaPipe] Hands 모델 로드 완료 (mock)")

    def extract_landmarks(self, frame) -> tuple:
        def _random_landmarks() -> HandLandmarks:
            side = random.choice(["Left", "Right"])
            return HandLandmarks(
                handedness=side,
                landmarks=[[round(random.uniform(0.0, 1.0), 4) for _ in range(3)] for _ in range(21)],
            )

        left  = _random_landmarks() if random.random() > 0.35 else None
        right = _random_landmarks() if random.random() > 0.25 else None

        if left:
            left.handedness = "Left"
        if right:
            right.handedness = "Right"

        return left, right


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지 1: Audio Pipeline
# ──────────────────────────────────────────────────────────────

class AudioPipeline:
    """
    MP4 → 오디오 추출 → Whisper 텍스트 변환 → 타임스탬프별 AudioSegment 리스트 반환.

    실제 구현 시 FFmpeg으로 먼저 오디오만 추출:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"],
            check=True, capture_output=True,
        )
    """

    def __init__(self):
        self._whisper = WhisperModel()

    def run(self, video_path: str) -> list:
        logger.info("[AudioPipeline] 오디오 파이프라인 시작")
        segments = self._whisper.transcribe(video_path)
        logger.info(f"[AudioPipeline] {len(segments)}개 음성 구간 추출 완료")
        return segments


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지 2: Vision Pipeline
# ──────────────────────────────────────────────────────────────

class VisionPipeline:
    """
    MP4 → OpenCV 프레임 샘플링 → YOLO 공구 탐지 + MediaPipe 손 관절 추출
    → 타임스탬프별 FrameFeature 리스트 반환.
    """

    FRAME_SAMPLE_INTERVAL_SEC = 0.5  # 프레임 샘플링 간격 (초 단위)

    def __init__(self):
        self._yolo      = YoloModel()
        self._mediapipe = MediaPipeModel()

    def run(self, video_path: str) -> tuple:
        """
        반환값: (frame_features, fps, duration_sec, total_frames)
        """
        logger.info("[VisionPipeline] 비전 파이프라인 시작")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        sample_step  = max(1, int(fps * self.FRAME_SAMPLE_INTERVAL_SEC))

        logger.info(
            f"[VisionPipeline] FPS={fps:.2f}  총 프레임={total_frames}  "
            f"재생 시간={duration_sec:.2f}s  샘플 간격={sample_step} frames"
        )

        frame_features = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                timestamp_sec = round(frame_idx / fps, 3)
                bboxes               = self._yolo.detect(frame)
                left_hand, right_hand = self._mediapipe.extract_landmarks(frame)

                frame_features.append(FrameFeature(
                    timestamp_sec=timestamp_sec,
                    detected_tools=bboxes,
                    left_hand=left_hand,
                    right_hand=right_hand,
                ))

            frame_idx += 1

        cap.release()
        logger.info(f"[VisionPipeline] {len(frame_features)}개 샘플 프레임 특징 추출 완료")
        return frame_features, fps, duration_sec, total_frames


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지 3: Feature Integrator
# ──────────────────────────────────────────────────────────────

class FeatureIntegrator:
    """
    오디오 트랜스크립트 세그먼트와 프레임별 비전 특징을 타임스탬프 기준으로 병합.

    병합 전략: 각 프레임 타임스탬프가 AudioSegment의 [start_sec, end_sec) 구간에
    속하면 해당 텍스트를 audio_context로 매핑. 구간 밖이면 None.

    출력 JSON 구조 (Gemini 1.5 Pro 입력으로 직접 전달):
    {
        "timestamp_sec": 3.5,
        "audio_context": "볼트를 풀 때는 반시계 방향으로 힘을 줍니다.",
        "detected_tools": [{"label": "Spanner", "confidence": 0.92, "bbox": [...]}],
        "hand_movement_vector": {
            "left_hand":  [[x, y, z], ...],  // 21개 랜드마크
            "right_hand": [[x, y, z], ...]
        }
    }
    """

    def merge(self, audio_segments: list, frame_features: list) -> list:
        logger.info("[Integrator] 오디오 + 비전 타임스탬프 병합 시작")

        integrated = []
        for feat in frame_features:
            audio_ctx = self._find_audio_context(feat.timestamp_sec, audio_segments)

            integrated.append(IntegratedEntry(
                timestamp_sec=feat.timestamp_sec,
                audio_context=audio_ctx,
                detected_tools=[
                    {
                        "label":      b.label,
                        "confidence": b.confidence,
                        "bbox":       [b.x1, b.y1, b.x2, b.y2],
                    }
                    for b in feat.detected_tools
                ],
                hand_movement_vector={
                    "left_hand":  asdict(feat.left_hand)["landmarks"]  if feat.left_hand  else None,
                    "right_hand": asdict(feat.right_hand)["landmarks"] if feat.right_hand else None,
                },
            ))

        logger.info(f"[Integrator] {len(integrated)}개 통합 타임라인 항목 생성 완료")
        return integrated

    @staticmethod
    def _find_audio_context(timestamp_sec: float, audio_segments: list) -> Optional[str]:
        for seg in audio_segments:
            if seg.start_sec <= timestamp_sec < seg.end_sec:
                return seg.text
        return None


# ──────────────────────────────────────────────────────────────
# JSON 직렬화 헬퍼
# ──────────────────────────────────────────────────────────────

def _serialize(result: ExtractionResult) -> dict:
    """ExtractionResult를 JSON 직렬화 가능한 순수 dict로 변환."""
    return {
        "file_path":    result.file_path,
        "duration_sec": result.duration_sec,
        "fps":          result.fps,
        "total_frames": result.total_frames,

        # Stage 1 결과: 오디오 트랜스크립트 세그먼트
        "audio_transcript": [asdict(seg) for seg in result.audio_transcript],

        # Stage 2 결과: 프레임별 원시 비전 특징 (frame_features 테이블 적재용)
        "frame_features": [
            {
                "timestamp_sec":  ff.timestamp_sec,
                "detected_tools": [asdict(b) for b in ff.detected_tools],
                "hand_movement_vector": {
                    "left_hand":  asdict(ff.left_hand)["landmarks"]  if ff.left_hand  else None,
                    "right_hand": asdict(ff.right_hand)["landmarks"] if ff.right_hand else None,
                },
            }
            for ff in result.frame_features
        ],

        # Stage 3 결과: 타임스탬프 정렬 통합 데이터 (Gemini 입력으로 직접 사용)
        "integrated_timeline": [asdict(entry) for entry in result.integrated_timeline],
    }


# ──────────────────────────────────────────────────────────────
# 오케스트레이터
# ──────────────────────────────────────────────────────────────

class FeatureExtractionWorker:
    """
    Feature Extraction Phase 전체 오케스트레이터.

    실행 순서:
        Stage 1. AudioPipeline  : MP4 → Whisper → 타임스탬프별 텍스트 세그먼트
        Stage 2. VisionPipeline : MP4 → OpenCV 프레임 샘플링
                                      → YOLO 공구 탐지
                                      → MediaPipe 손 관절 추출
                                      → 프레임별 JSON
        Stage 3. Integrator     : 오디오 + 비전 타임스탬프 정렬 병합
        Stage 4. Persist        : JSON 파일 출력 (다음 단계 Gemini 입력)
    """

    def __init__(self):
        self._audio    = AudioPipeline()
        self._vision   = VisionPipeline()
        self._integrator = FeatureIntegrator()

    def run(self, video_path: str, output_path: Optional[str] = None) -> ExtractionResult:
        logger.info("=" * 60)
        logger.info(f"[Worker] Feature Extraction Phase 시작: {video_path}")
        logger.info("=" * 60)

        # Stage 1: Audio Pipeline
        audio_segments = self._audio.run(video_path)

        # Stage 2: Vision Pipeline
        frame_features, fps, duration_sec, total_frames = self._vision.run(video_path)

        # Stage 3: Feature Integration
        integrated_timeline = self._integrator.merge(audio_segments, frame_features)

        result = ExtractionResult(
            file_path=video_path,
            duration_sec=round(duration_sec, 3),
            fps=round(fps, 3),
            total_frames=total_frames,
            audio_transcript=audio_segments,
            frame_features=frame_features,
            integrated_timeline=integrated_timeline,
        )

        # Stage 4: JSON 파일 출력
        if output_path is None:
            output_path = str(Path(video_path).with_suffix("")) + "_features.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_serialize(result), f, ensure_ascii=False, indent=2)

        logger.info(f"[Worker] 출력 파일 저장 완료: {output_path}")
        logger.info(f"[Worker] 요약: 오디오 세그먼트={len(audio_segments)}  "
                    f"샘플 프레임={len(frame_features)}  "
                    f"통합 항목={len(integrated_timeline)}")
        logger.info("=" * 60)
        logger.info("[Worker] Feature Extraction Phase 완료")
        logger.info("=" * 60)
        return result


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
        description="암묵지 추출 시스템 - Feature Extraction Phase 워커",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "video_path",
        help="처리할 MP4 영상 파일 경로\n예: python extractor.py ./videos/master_01.mp4",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 JSON 파일 경로 (기본값: <영상명>_features.json)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {args.video_path}")

    worker = FeatureExtractionWorker()
    worker.run(args.video_path, args.output)


if __name__ == "__main__":
    main()
