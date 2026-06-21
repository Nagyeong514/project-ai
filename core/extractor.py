"""
Feature Extraction Phase Worker
--------------------------------
파이프라인 위치: Data Ingestion -> [Feature Extraction] -> Knowledge Synthesis -> DB 적재

입력 : MP4 영상 파일 경로
출력 : <영상명>_features.json

의존 패키지:
    pip install opencv-python openai-whisper ultralytics mediapipe
    ffmpeg 설치 필수 (Whisper 내부 오디오 추출에 사용)
    Windows: https://ffmpeg.org/download.html -> 압축 해제 후 PATH 등록
"""

import os
import json
import logging
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 데이터 모델
# ──────────────────────────────────────────────────────────────

@dataclass
class AudioSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class BoundingBox:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class HandLandmarks:
    handedness: str
    landmarks: list  # shape: [21, 3] — 각 원소는 [x, y, z] 정규화 좌표


@dataclass
class FrameFeature:
    timestamp_sec: float
    detected_tools: list
    left_hand: Optional[HandLandmarks]
    right_hand: Optional[HandLandmarks]


@dataclass
class IntegratedEntry:
    timestamp_sec: float
    audio_context: Optional[str]
    detected_tools: list
    hand_movement_vector: dict


@dataclass
class ExtractionResult:
    file_path: str
    duration_sec: float
    fps: float
    total_frames: int
    audio_transcript: list
    frame_features: list
    integrated_timeline: list


# ──────────────────────────────────────────────────────────────
# 실제 AI 모델 클래스
# ──────────────────────────────────────────────────────────────

class WhisperModel:
    """
    openai-whisper 로컬 모델을 사용한 실제 음성-텍스트 변환.

    model_size 옵션:
        tiny   : 가장 빠름, 정확도 낮음 (39MB)
        base   : 속도/정확도 균형 (74MB) -- 기본값
        small  : 더 높은 정확도 (244MB)
        medium : 높은 정확도, 느림 (769MB)

    ffmpeg 필수: Whisper가 내부적으로 ffmpeg을 사용하여 영상에서 오디오를 추출함.
    ffmpeg 미설치 시 "ffmpeg not found" 오류가 발생함.
    """

    def __init__(self, model_size: str = "base"):
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper 패키지가 설치되지 않았습니다.\n"
                "pip install openai-whisper 를 실행하세요."
            )
        logger.info(f"[Whisper] 모델 로드 중: {model_size} (최초 실행 시 다운로드)")
        self._model = whisper.load_model(model_size)
        logger.info(f"[Whisper] 모델 로드 완료: {model_size}")

    def transcribe(self, video_path: str) -> list:
        """
        영상 파일에서 오디오를 추출하여 전체 텍스트 및 타임스탬프 세그먼트를 반환한다.
        Whisper가 ffmpeg을 통해 직접 영상 파일을 처리하므로 별도 오디오 추출 불필요.
        언어는 자동 감지한다 (한국어, 영어 모두 처리 가능).
        """
        logger.info(f"[Whisper] 음성 변환 시작: {video_path}")
        result = self._model.transcribe(video_path, verbose=False)
        segments = [
            AudioSegment(
                start_sec=round(seg["start"], 3),
                end_sec=round(seg["end"], 3),
                text=seg["text"].strip(),
            )
            for seg in result["segments"]
            if seg["text"].strip()
        ]
        logger.info(f"[Whisper] 감지 언어: {result.get('language', 'unknown')}")
        logger.info(f"[Whisper] {len(segments)}개 음성 구간 추출 완료")
        return segments


class YoloModel:
    """
    YOLOv8 pretrained 모델을 사용한 실제 객체 탐지.

    사용 모델: yolov8n.pt (COCO 80개 클래스, 최초 실행 시 자동 다운로드 약 6MB)
    COCO 데이터셋은 자동차 정비 전문 공구(Spanner 등)를 포함하지 않으므로
    범용 객체(person, scissors 등)를 탐지한다.
    정밀 공구 분류가 필요한 경우 별도 학습된 커스텀 가중치(.pt) 파일로 교체한다.
    """

    CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, model_path: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics 패키지가 설치되지 않았습니다.\n"
                "pip install ultralytics 를 실행하세요."
            )
        logger.info(f"[YOLO] 모델 로드 중: {model_path}")
        self._model = YOLO(model_path)
        logger.info(f"[YOLO] 모델 로드 완료: {model_path}")

    def detect(self, frame) -> list:
        """
        단일 프레임에서 객체를 탐지하고 BoundingBox 리스트를 반환한다.
        confidence_threshold 미만의 탐지 결과는 제외한다.
        """
        results = self._model(frame, verbose=False, conf=self.CONFIDENCE_THRESHOLD)[0]
        return [
            BoundingBox(
                label=results.names[int(box.cls)],
                confidence=round(float(box.conf), 4),
                x1=int(box.xyxy[0][0]),
                y1=int(box.xyxy[0][1]),
                x2=int(box.xyxy[0][2]),
                y2=int(box.xyxy[0][3]),
            )
            for box in results.boxes
        ]


class MediaPipeModel:
    """
    MediaPipe Tasks API (0.10.x 이상)를 사용한 실제 손 관절 랜드마크 추출.
    21개 랜드마크 x (x, y, z) 정규화 좌표를 반환한다.

    model_path: hand_landmarker.task 파일 경로 (프로젝트 루트에 위치).
    최초 실행 전 아래 명령으로 모델 파일을 다운로드해야 한다:
        python -c "
        import urllib.request
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'hand_landmarker.task'
        )"
    """

    MODEL_PATH = str(Path(__file__).parent.parent / "models" / "hand_landmarker.task")

    def __init__(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            raise ImportError(
                "mediapipe 패키지가 설치되지 않았습니다.\n"
                "pip install mediapipe 를 실행하세요."
            )

        if not Path(self.MODEL_PATH).is_file():
            raise FileNotFoundError(
                f"hand_landmarker.task 모델 파일이 없습니다: {self.MODEL_PATH}\n"
                "models/ 디렉토리에 hand_landmarker.task 파일을 다운로드하세요."
            )

        base_options = mp_python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        logger.info("[MediaPipe] HandLandmarker 모델 로드 완료 (Tasks API)")

    def extract_landmarks(self, frame) -> tuple:
        """
        단일 프레임에서 최대 2개의 손 랜드마크를 추출한다.
        탐지 실패 시 해당 손은 None을 반환한다.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = self._detector.detect(mp_image)

        left_hand:  Optional[HandLandmarks] = None
        right_hand: Optional[HandLandmarks] = None

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            if i >= len(result.handedness):
                break
            side = result.handedness[i][0].category_name  # "Left" or "Right"
            parsed = HandLandmarks(
                handedness=side,
                landmarks=[
                    [round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]
                    for lm in hand_landmarks
                ],
            )
            if side == "Left":
                left_hand = parsed
            else:
                right_hand = parsed

        return left_hand, right_hand

    def close(self) -> None:
        """MediaPipe 리소스를 해제한다."""
        self._detector.close()


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지 1: Audio Pipeline
# ──────────────────────────────────────────────────────────────

class AudioPipeline:
    """MP4 -> Whisper -> 타임스탬프별 AudioSegment 리스트 반환."""

    def __init__(self, model_size: str = "base"):
        self._whisper = WhisperModel(model_size=model_size)

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
    MP4 -> OpenCV 프레임 샘플링 -> YOLO 객체 탐지 + MediaPipe 손 관절 추출
    -> 타임스탬프별 FrameFeature 리스트 반환.
    """

    FRAME_SAMPLE_INTERVAL_SEC = 0.5

    def __init__(self):
        self._yolo       = YoloModel()
        self._mediapipe  = MediaPipeModel()

    def run(self, video_path: str) -> tuple:
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
                timestamp_sec         = round(frame_idx / fps, 3)
                bboxes                = self._yolo.detect(frame)
                left_hand, right_hand = self._mediapipe.extract_landmarks(frame)

                frame_features.append(FrameFeature(
                    timestamp_sec=timestamp_sec,
                    detected_tools=bboxes,
                    left_hand=left_hand,
                    right_hand=right_hand,
                ))

            frame_idx += 1

        cap.release()
        self._mediapipe.close()

        logger.info(f"[VisionPipeline] {len(frame_features)}개 샘플 프레임 처리 완료")
        return frame_features, fps, duration_sec, total_frames


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지 3: Feature Integrator
# ──────────────────────────────────────────────────────────────

class FeatureIntegrator:
    """
    오디오 세그먼트와 프레임 비전 특징을 타임스탬프 기준으로 병합한다.
    각 프레임 타임스탬프가 AudioSegment의 [start_sec, end_sec) 구간에
    속하면 해당 텍스트를 audio_context로 매핑하고, 없으면 None으로 처리한다.
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
    return {
        "file_path":    result.file_path,
        "duration_sec": result.duration_sec,
        "fps":          result.fps,
        "total_frames": result.total_frames,
        "audio_transcript": [asdict(seg) for seg in result.audio_transcript],
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
        "integrated_timeline": [asdict(entry) for entry in result.integrated_timeline],
    }


# ──────────────────────────────────────────────────────────────
# 오케스트레이터
# ──────────────────────────────────────────────────────────────

class FeatureExtractionWorker:
    """
    Feature Extraction Phase 전체 오케스트레이터.

    Stage 1. AudioPipeline  : MP4 -> Whisper -> 실제 음성 세그먼트 + 타임스탬프
    Stage 2. VisionPipeline : MP4 -> OpenCV -> YOLOv8 탐지 + MediaPipe 손 관절
    Stage 3. Integrator     : 오디오 + 비전 타임스탬프 정렬 병합
    Stage 4. JSON 출력      : Gemini 입력용 _features.json 저장
    """

    def __init__(self, whisper_model_size: str = "base"):
        self._audio      = AudioPipeline(model_size=whisper_model_size)
        self._vision     = VisionPipeline()
        self._integrator = FeatureIntegrator()

    def run(self, video_path: str, output_path: Optional[str] = None) -> ExtractionResult:
        logger.info("=" * 60)
        logger.info(f"[Worker] Feature Extraction Phase 시작: {video_path}")
        logger.info("=" * 60)

        audio_segments = self._audio.run(video_path)
        frame_features, fps, duration_sec, total_frames = self._vision.run(video_path)
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

        if output_path is None:
            output_path = str(Path(video_path).with_suffix("")) + "_features.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_serialize(result), f, ensure_ascii=False, indent=2)

        logger.info(f"[Worker] 출력 파일 저장 완료: {output_path}")
        logger.info(
            f"[Worker] 요약: 오디오 세그먼트={len(audio_segments)}  "
            f"샘플 프레임={len(frame_features)}  "
            f"통합 항목={len(integrated_timeline)}"
        )
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
    parser.add_argument("video_path", help="처리할 MP4 파일 경로")
    parser.add_argument("--output", "-o", default=None, help="출력 JSON 파일 경로")
    parser.add_argument(
        "--whisper-model", default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper 모델 크기 (기본값: base)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {args.video_path}")

    worker = FeatureExtractionWorker(whisper_model_size=args.whisper_model)
    worker.run(args.video_path, args.output)


if __name__ == "__main__":
    main()
