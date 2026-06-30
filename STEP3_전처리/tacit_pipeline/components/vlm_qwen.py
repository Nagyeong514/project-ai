"""
VLM 관찰 추출 어댑터 (Qwen3-VL-8B-Instruct, 4bit NF4 / BitsAndBytes). 스펙 5.3.

역할: '눈' — 관찰 가능한 사실만 기록(observations). 묶기/해석은 후속 LLM.
입력: 샘플 프레임 + 프레임별 YOLO 검출(위치 힌트) → 관찰 로그(JSON, 버전 A).

현재 제약: RTX 2080 8GB 단일 워크스테이션 → 4bit NF4 양자화 필수. 양자화 on/off는 config.
⚠️ Turing(sm75): fp16/4bit만. bf16/FP8 금지.
⚠️ 오늘은 모델 로딩 금지 — 지연 import/로딩.

팀 검증 기본 파라미터(config 기본값):
  max_pixels=192*192, repetition_penalty=1.2, max_new_tokens=4000, do_sample=False(greedy),
  fps=영상 길이별(짧으면 0.5, 길면 0.15까지) + config 오버라이드.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..interfaces.detector import FrameRef
from ..prompts.vlm_observation import (
    build_observation_messages,
    build_video_observation_messages,
)
from ..schema.intermediate import (
    ActionDescription,
    FrameDetections,
    FrameMeta,
    hhmmss_to_seconds,
    seconds_to_hhmmss,
)


class QwenVLActionExtractor:
    """Qwen3-VL 관찰 어댑터. registry 키: 'qwen3_vl'.

    backend 기본 'hf_transformers'(bnb 4bit는 transformers 경로 권장; vllm bnb는 Turing서 TP1 제약).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        backend: str = "hf_transformers",  # "hf_transformers" | "vllm"
        device: str = "cuda",
        # 입력 모드(결정됨): 본선 = 네이티브 비디오+fps. 프레임-리스트는 옵션.
        input_mode: str = "native_video",  # "native_video" | "frame_list"
        # ── 양자화(8GB 필수) ──
        quantization: str | None = "nf4",  # "nf4"(bnb 4bit) | "fp16" | None. config로 해제 가능.
        # ── 팀 검증 생성 파라미터 ──
        max_pixels: int = 192 * 192,  # 128px는 부품 오인 → 192로 상향(검증값)
        repetition_penalty: float = 1.2,  # 짧은 영상 동일 블록 무한반복 억제
        max_new_tokens: int = 4000,
        do_sample: bool = False,  # greedy — 재현성(프롬프트 실험 필수)
        # ── fps(네이티브 비디오 입력 모드용) ──
        fps_short: float = 0.5,
        fps_long: float = 0.15,
        long_video_threshold_sec: float = 120.0,  # 이 이상이면 long으로 간주
        fps_override: float | None = None,  # 주면 길이 무시하고 강제
        # ── 부품 주입(기법 2) ──
        part_injection: bool = True,  # [고정 사실] 블록 주입 on/off
        videos_map_path: Optional[str] = None,  # 영상→부품명 매핑 JSON. None이면 YOLO 검출로 자동
        tensor_parallel_size: int = 1,
        **extra: Any,
    ):
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.input_mode = input_mode
        self.quantization = quantization
        self.max_pixels = max_pixels
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.fps_short = fps_short
        self.fps_long = fps_long
        self.long_video_threshold_sec = long_video_threshold_sec
        self.fps_override = fps_override
        self.part_injection = part_injection
        self.videos_map_path = videos_map_path
        self.tensor_parallel_size = tensor_parallel_size
        self.extra = extra
        self._model = None
        self._processor = None
        self._videos_map = self._load_videos_map(videos_map_path)

    # ── 영상 길이별 fps 결정(검증 규칙) ──────────────────────────────────
    def fps_for_duration(self, duration_sec: float) -> float:
        """짧은 영상=고fps, 긴 영상=저fps(OOM/유사프레임 과다 방지). config override 우선."""
        if self.fps_override is not None:
            return self.fps_override
        return self.fps_long if duration_sec >= self.long_video_threshold_sec else self.fps_short

    def _load_videos_map(self, path: Optional[str]) -> Dict[str, List[str]]:
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}

    def _injected_parts_for(self, video_id: str, detected: List[str]) -> Optional[List[str]]:
        """부품 주입 대상 결정. videos_map에 있으면 그걸, 없으면 YOLO 검출 객체로 자동 주입.

        실험상 GPU 영상은 모델 자체 인식이 좋음 → 그런 경우 part_injection=False로 끌 수 있다.
        """
        if not self.part_injection:
            return None
        if video_id in self._videos_map:
            return self._videos_map[video_id]
        # 자동: 이 영상에서 검출된 고유 클래스명(위치 힌트 겸 고정사실)
        uniq = sorted(set(detected))
        return uniq or None

    def _load(self):
        if self._model is not None:
            return
        if self.backend == "hf_transformers":
            import torch  # noqa: 지연 import
            from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa

            quant_kwargs: Dict[str, Any] = {}
            if self.quantization == "nf4":
                from transformers import BitsAndBytesConfig  # noqa

                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,  # Turing: fp16
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quant_kwargs["torch_dtype"] = torch.float16

            # max_pixels는 processor에서 통제(과도한 토큰/메모리 방지)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True, max_pixels=self.max_pixels
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                device_map=self.device,
                trust_remote_code=True,
                **quant_kwargs,
            )
        elif self.backend == "vllm":
            # NOTE: bnb 4bit는 vllm Turing 경로서 제약(TP1). 가능하면 transformers 권장.
            from vllm import LLM  # noqa

            self._model = LLM(
                model=self.model_name,
                dtype="float16",
                quantization="bitsandbytes" if self.quantization == "nf4" else None,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                **self.extra,
            )
        else:
            raise ValueError(f"알 수 없는 VLM backend: {self.backend}")

    # ── 본선: 네이티브 비디오 모드 ──────────────────────────────────────
    def observe_video(
        self,
        video_path: str,
        meta: FrameMeta,
        injected_parts: Optional[List[str]] = None,
    ) -> List[ActionDescription]:
        """영상 전체를 fps로 샘플해 한 번에 관찰(본선 모드). 팀 검증 파라미터 기준.

        Qwen3-VL 네이티브 비디오 입력: fps_for_duration(영상길이)로 fps 결정 → processor가
        프레임을 만든다. timestamp는 모델이 영상 기준으로 붙인 값을 초로 되돌려 보관한다(5.6:
        자유 계산이 아니라 fps 그리드 시각).
        """
        self._load()
        duration = (meta.n_frames / meta.fps) if meta.fps else 0.0
        fps = self.fps_for_duration(duration)
        messages = build_video_observation_messages(injected_parts)
        raw = self._infer_video(video_path, fps, messages)
        out: List[ActionDescription] = []
        for obs in self._parse_observations(raw):
            ts = hhmmss_to_seconds(obs.get("timestamp"))
            out.append(
                ActionDescription(
                    timestamp=ts if ts is not None else 0.0,
                    actor=obs.get("actor"),
                    action=str(obs.get("action", "")).strip(),
                    objects=list(obs.get("objects_visible", [])),
                    raw=raw,
                )
            )
        return out

    # ── 옵션: 프레임-리스트 모드 ────────────────────────────────────────
    def describe_actions(
        self,
        frames: List[FrameRef],
        detections_by_frame: List[FrameDetections],
    ) -> List[ActionDescription]:
        self._load()
        det_map = {fd.frame_idx: fd for fd in detections_by_frame}
        out: List[ActionDescription] = []

        # 이 영상 전체에서 검출된 클래스(부품 자동 주입용)
        all_detected = [d.cls for fd in detections_by_frame for d in fd.detections]
        # video_id 추론(프레임엔 없으므로 detections에 기대지 않고 호출측 일임 → 자동주입은 검출 기반)
        injected = self._injected_parts_for(video_id="", detected=all_detected)

        for fr in frames:
            fd = det_map.get(fr.frame_idx)
            objs = (
                [
                    {"class": d.cls, "conf": round(d.conf, 3),
                     "bbox": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h]}
                    for d in fd.detections
                ]
                if fd
                else []
            )
            ts_label = seconds_to_hhmmss(fr.timestamp)
            messages = build_observation_messages(ts_label, objs, injected_parts=injected)
            raw = self._infer(fr, messages)
            for obs in self._parse_observations(raw):
                out.append(
                    ActionDescription(
                        timestamp=fr.timestamp,  # 샘플링 부여 시각 그대로(VLM 재계산 안 함)
                        actor=obs.get("actor"),
                        action=str(obs.get("action", "")).strip(),
                        objects=list(obs.get("objects_visible", [])),
                        raw=raw,
                    )
                )
        return out

    def _infer_video(self, video_path: str, fps: float, messages: List[Dict[str, str]]) -> str:
        """**본선** 네이티브 비디오 추론. video_path + fps → 관찰 JSON 문자열.

        TODO(impl): Qwen3-VL 비디오 입력 결합(qwen_vl_utils.process_vision_info 등):
            messages 에 {"type":"video","video":video_path,"fps":fps,"max_pixels":self.max_pixels}
            를 넣고 processor → model.generate(max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample, repetition_penalty=self.repetition_penalty).
        """
        raise NotImplementedError("내일 서버에서 네이티브 비디오 추론 구현. 오늘은 골격만.")

    def _infer(self, frame: FrameRef, messages: List[Dict[str, str]]) -> str:
        """(옵션) 프레임-리스트 모드 이미지 추론. backend별 멀티모달 입력 결합.

        TODO(impl): Qwen3-VL processor 로 frame.image(PIL/numpy)+messages 결합해 generate.
            generate 인자: max_new_tokens, do_sample, repetition_penalty 적용.
        """
        raise NotImplementedError("내일 서버에서 프레임-리스트 추론 구현(옵션 모드). 오늘은 골격만.")

    def unload(self) -> None:
        """GPU 메모리 해제 — 순차 실행(VLM→언로드→LLM)용. 단일 8GB라 필수.

        TODO(impl): 백엔드별 정리. transformers면 del model; torch.cuda.empty_cache().
        """
        try:
            import gc

            self._model = None
            self._processor = None
            gc.collect()
            try:
                import torch  # noqa

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            self._model = None
            self._processor = None

    @staticmethod
    def _parse_observations(raw: str) -> List[Dict[str, Any]]:
        """VLM JSON 응답에서 observations 배열 파싱. 실패 시 원문 1건으로(손실 방지)."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
        try:
            start, end = text.find("{"), text.rfind("}")
            obj = json.loads(text[start : end + 1])
            obs = obj.get("observations", [])
            return obs if isinstance(obs, list) else []
        except Exception:
            return [{"actor": None, "action": raw.strip(), "objects_visible": []}]
