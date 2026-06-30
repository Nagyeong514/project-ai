"""
Whisper large-v3-turbo STT 어댑터.

스펙 5.4:
  - segment 단위 timestamp(초) 포함 출력.
  - condition_on_previous_text=False 기본(할루시네이션/반복 루프 억제). config로 노출.
  - 결과를 transcripts/<video_id>.json 로 저장(스키마 transcript_ref 가 가리킴).

⚠️ 오늘은 서버가 없으므로 모델을 import/로딩하지 않는다. transcribe() 안에서 지연 import 한다.
   내일 `pip install -r requirements.txt` 후 그대로 동작.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..schema.intermediate import Transcript, Utterance, UtteranceTag


class WhisperTurboSTT:
    """faster-whisper(권장) 또는 openai-whisper 백엔드. registry 키: 'whisper_turbo'."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",  # Turing(sm75): fp16만. bf16/FP8 금지.
        language: str | None = "ko",
        condition_on_previous_text: bool = False,  # 스펙 기본값
        transcript_dir: str = "transcripts",
        backend: str = "faster_whisper",  # "faster_whisper" | "openai_whisper"
        **extra: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.condition_on_previous_text = condition_on_previous_text
        self.transcript_dir = transcript_dir
        self.backend = backend
        self.extra = extra
        self._model = None  # 지연 로딩

    def _load(self):
        """모델 지연 로딩(내일 서버에서 최초 호출 시)."""
        if self._model is not None:
            return
        if self.backend == "faster_whisper":
            from faster_whisper import WhisperModel  # noqa: 지연 import

            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        elif self.backend == "openai_whisper":
            import whisper  # noqa

            self._model = whisper.load_model(self.model_name, device=self.device)
        else:
            raise ValueError(f"알 수 없는 STT backend: {self.backend}")

    def transcribe(self, video_path: str, video_id: str) -> Transcript:
        self._load()
        utterances = []

        if self.backend == "faster_whisper":
            segments, info = self._model.transcribe(
                video_path,
                language=self.language,
                condition_on_previous_text=self.condition_on_previous_text,
                word_timestamps=False,
                **self.extra,
            )
            lang = getattr(info, "language", self.language)
            for seg in segments:
                text = seg.text.strip()
                utterances.append(
                    Utterance(
                        start=float(seg.start),
                        end=float(seg.end),
                        raw_text=text,
                        normalized_text=text,  # 정규화는 TranscriptRefiner가 채움
                        tags=[UtteranceTag.NONE],
                    )
                )
        else:  # openai_whisper
            result = self._model.transcribe(
                video_path,
                language=self.language,
                condition_on_previous_text=self.condition_on_previous_text,
                **self.extra,
            )
            lang = result.get("language", self.language)
            for seg in result.get("segments", []):
                text = seg["text"].strip()
                utterances.append(
                    Utterance(
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        raw_text=text,
                        normalized_text=text,
                        tags=[UtteranceTag.NONE],
                    )
                )

        transcript = Transcript(
            video_id=video_id, language=lang, model=self.model_name, utterances=utterances
        )
        self._save(transcript)
        return transcript

    def _save(self, transcript: Transcript) -> str:
        """transcripts/<video_id>.json 로 저장. 경로를 돌려준다(transcript_ref용)."""
        out_dir = Path(self.transcript_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{transcript.video_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)
        return str(path)
