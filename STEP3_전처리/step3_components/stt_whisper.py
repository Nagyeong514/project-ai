"""
Whisper large-v3-turbo STT 어댑터.

스펙 5.4:
  - segment 단위 timestamp(초) 포함 출력.
  - condition_on_previous_text=False 기본(할루시네이션/반복 루프 억제). config로 노출.
  - 결과를 transcripts/<video_id>.json 로 저장(스키마 transcript_ref 가 가리킴).

VAD (2026-07-02 갱신, `_내일_서버전환_가이드.md`의 옛 결정 "VAD는 병합 부작용 → OFF 유지"를
뒤집음): 00_사전연구/02_VAD재검증_v2에서 Silero로 발화 구간을 잘라 "이어붙이는" 방식은 결과 오디오가
짧아져 seg.start/end가 원본 영상 타임라인과 어긋난다는 걸 확인했다(aligner.py가 발화·행동
타임스탬프를 ±윈도우로 직접 매칭하므로 이 어긋남은 그대로 정렬 오류가 됨). 대신
faster-whisper 내장 vad_filter=True는 오디오를 자르지 않고 무음 구간만 스킵하면서
세그먼트 타임스탬프를 원본 기준으로 그대로 리포트한다 — CLIP1~4 실측으로 확인함
(`00_사전연구/02_VAD재검증_v2/results/vad_filter_builtin/`): CER은 병합 방식과 동급 이상(평균
0.2362→0.1813), RTF는 더 낮음(0.009~0.014), 세그먼트가 원본 오디오 전체 구간에 분포함
(트리밍후처럼 쏠리지 않음). vad_filter/vad_parameters를 config로 노출해 기본 켬.

⚠️ 오늘은 서버가 없으므로 모델을 import/로딩하지 않는다. transcribe() 안에서 지연 import 한다.
   내일 `pip install -r requirements.txt` 후 그대로 동작.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from tacit_common.schema.intermediate import Transcript, Utterance


class WhisperTurboSTT:
    """faster-whisper(권장) 또는 openai-whisper 백엔드. registry 키: 'whisper_turbo'."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",  # Turing(sm75): fp16만. bf16/FP8 금지.
        language: str | None = "ko",
        condition_on_previous_text: bool = False,  # 스펙 기본값
        vad_filter: bool = False,  # faster_whisper 전용. threshold는 vad_parameters로.
        vad_parameters: Dict[str, Any] | None = None,
        transcript_dir: str = "transcripts",
        backend: str = "faster_whisper",  # "faster_whisper" | "openai_whisper"
        **extra: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {}
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

    def unload(self) -> None:
        """GPU 메모리 해제. STT는 파이프라인 초반 1회만 쓰므로 끝나면 바로 비운다.

        2026-07-02: STT(cuda)가 언로드 안 된 채 VLM→LLM까지 이어지면 뒤 단계에서
        OOM 여유가 줄어드는 걸 실측함(특히 VLM이 실제 관찰을 많이 뽑아 LLM 입력이
        커진 경우) — orchestrator가 STT 직후 호출한다.
        """
        import gc

        self._model = None
        gc.collect()
        try:
            import torch  # noqa

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def transcribe(self, video_path: str, video_id: str) -> Transcript:
        self._load()
        utterances = []

        if self.backend == "faster_whisper":
            segments, info = self._model.transcribe(
                video_path,
                language=self.language,
                condition_on_previous_text=self.condition_on_previous_text,
                word_timestamps=False,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters or None,
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
