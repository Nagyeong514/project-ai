"""
VLM 관찰 추출 어댑터 (Qwen3-VL-8B-Instruct, 4bit NF4 / transformers). 스펙 5.3.

역할: '눈' — 관찰 가능한 사실만 기록(observations). 묶기/해석은 후속 LLM.
입력: ffmpeg로 추출한 프레임(YOLO와 공용) → 관찰 로그(JSON 버전 A).

검증된 레시피(2026-06-30, [[step3-runtime-recipe]]):
  - 프레임은 ffmpeg CLI로 추출(이 env서 torchcodec/pyav/cv2 불안정) → PIL 리스트로 전달.
  - transformers + attn_implementation="sdpa"(Turing FA2 불가) + BitsAndBytesConfig nf4(compute fp16).
  - 팀 검증 파라미터: max_pixels=192², repetition_penalty=1.2, max_new_tokens=4000, do_sample=False.
  - 실측 VRAM 피크 ~6.9GB (8GB OK).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from step4_prompts.vlm_observation import build_video_observation_messages
from tacit_common.schema.intermediate import (
    ActionDescription,
    hhmmss_to_seconds,
    seconds_to_hhmmss,
)

logger = logging.getLogger(__name__)


def snap_to_grid(ts: float, grid: List[float]) -> float:
    """모델이 뱉은 시각 ts 를 실제 프레임 그리드 중 가장 가까운 값으로 교정.

    원칙 복원(5.6): 시간은 모델 추정이 아니라 우리가 부여한 그리드 값을 들고 간다.
    grid 가 비어 있으면 교정할 기준이 없으므로 ts 를 그대로 돌려준다.
    """
    if not grid:
        return ts
    return min(grid, key=lambda g: abs(ts - g))


class QwenVLActionExtractor:
    """Qwen3-VL 관찰 어댑터. registry 키: 'qwen3_vl'.

    프레임 추출(fps 정책 포함)은 STEP3 소관 — 이 클래스는 이미 추출된 frame_paths/times를
    받아 관찰만 한다(observe_frames). 예전엔 여기서 직접 ffmpeg를 불러 프레임을 뽑는
    observe_video()가 있었으나 orchestrator가 실제로 쓴 적 없는 죽은 경로였고(항상
    frame_extract.extract_frames를 직접 호출해 observe_frames로 넘겼음), STEP3/4 분리로
    fps 정책(frame_extraction: config)이 STEP3로 완전히 이동하면서 제거함.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        backend: str = "hf_transformers",
        device: str = "cuda:0",
        input_mode: str = "native_video",
        quantization: str | None = "nf4",
        # 생성 파라미터(팀 검증)
        max_pixels: int = 192 * 192,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 4000,
        do_sample: bool = False,
        # 부품 주입
        part_injection: bool = True,
        videos_map_path: Optional[str] = None,
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
        self.part_injection = part_injection
        self.videos_map_path = videos_map_path
        self.extra = extra
        self._model = None
        self._processor = None
        self._videos_map = self._load_videos_map(videos_map_path)

    # ── 부품주입 ───────────────────────────────────────────────────
    def _load_videos_map(self, path: Optional[str]) -> Dict[str, List[str]]:
        if not path:
            return {}
        import json
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}

    def _injected_parts_for(self, video_id: str, detected: List[str]) -> Optional[List[str]]:
        if not self.part_injection:
            return None
        if video_id in self._videos_map:
            return self._videos_map[video_id]
        uniq = sorted(set(detected))
        return uniq or None

    # ── 모델 로딩(지연) ──────────────────────────────────────────────────
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import (Qwen3VLForConditionalGeneration, AutoProcessor,
                                   BitsAndBytesConfig)
        quant_kwargs: Dict[str, Any] = {}
        if self.quantization == "nf4":
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        else:
            quant_kwargs["torch_dtype"] = torch.float16
        self._processor = AutoProcessor.from_pretrained(self.model_name, max_pixels=self.max_pixels)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name, device_map=self.device,
            attn_implementation="sdpa", **quant_kwargs)

    def unload(self) -> None:
        import gc
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 코어: 추출된 프레임(STEP3가 이미 뽑아둔 것) → 관찰 ─────────────────
    def observe_frames(self, frame_paths: List[str], times: List[float],
                       injected_parts: Optional[List[str]] = None) -> List[ActionDescription]:
        self._load()
        from PIL import Image
        frames = [Image.open(p).convert("RGB") for p in frame_paths]

        base = build_video_observation_messages(injected_parts)
        system_text, user_text = base[0]["content"], base[1]["content"]
        # 2026-07-02 실측: 55개 시각을 전부 나열해 "이 중에서 골라라"라고 강제하면 모델이
        # observations=[] 로 얼어붙는다(제약이 너무 빡빡함). 대략적인 초 단위만 요구하고,
        # 실제 그리드 스냅은 파싱 후 snap_to_grid()가 코드로 교정한다.
        user_text += (f"\n\n영상 길이는 약 {seconds_to_hhmmss(times[-1]) if times else '알 수 없음'}이다. "
                      "timestamp는 영상 시작(00:00:00) 기준 대략적인 초 단위로 적어라.")
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [{"type": "video", "video": frames},
                                         {"type": "text", "text": user_text}]},
        ]
        # 2026-07-04: video_metadata 없이 넘기면 processing_qwen3_vl.replace_video_token()이
        # metadata.fps=None을 만나 임의로 fps=24를 가정해버린다 — 실제로는 이 프레임들이
        # (times[-1]-times[0])/len(times) 간격(예: 6.67초)으로 퍼져 있는데 모델은 "30프레임
        # =1.25초"로 착각하게 됨(프롬프트의 "영상 길이 약 3분19초"와 모순 → 관찰 0건의
        # 유력한 원인, 00_사전연구 디버그 로그로 재현 확인). 우리가 이미 아는 실제 간격을
        # video_metadata로 명시해 이 오추정을 원천 차단한다.
        real_fps = (1.0 / (times[1] - times[0])) if len(times) > 1 else 1.0
        video_metadata = {
            "total_num_frames": len(frames),
            "fps": real_fps,
            "frames_indices": list(range(len(frames))),
            "duration": times[-1] if times else None,
        }
        raw = self._generate_mm(messages, video_metadata=video_metadata)
        print(f"[VLM-RAW] len={len(raw)}\n{'-'*70}\n{raw}\n{'-'*70}")  # 2026-07-04: 0건 디버깅용 임시 로그

        parsed = self._parse_observations(raw)
        n = len(parsed)
        out: List[ActionDescription] = []
        for i, obs in enumerate(parsed):
            ts = hhmmss_to_seconds(obs.get("timestamp"))
            if ts is not None:
                ts = snap_to_grid(ts, times)  # 모델 추정 시각 → 실제 그리드로 교정
            else:
                # 파싱 실패/누락: 0:00 떼몰림 대신 이 청크 grid를 관찰 순서대로 균등 배분.
                ts = self._even_grid_ts(i, n, times)
            out.append(ActionDescription(
                timestamp=ts,
                actor=obs.get("actor"),
                action=str(obs.get("action", "")).strip(),
                objects=list(obs.get("objects_visible", [])),
                raw=raw))

        # 순서 검증: timestamp 단조증가로 안정 정렬(동시각은 입력순 유지). 내용은 안 버린다.
        ordered = sorted(out, key=lambda a: a.timestamp)  # Python sort = stable
        moved = sum(1 for a, b in zip(out, ordered) if a is not b)
        if moved:
            logger.warning(f"[vlm] reordered {moved} observations by timestamp")
        return ordered

    def describe_actions(self, frames, detections_by_frame):
        """[옵션] 프레임-리스트 모드. 본선은 observe_video. (미사용)"""
        raise NotImplementedError("프레임-리스트 모드는 옵션 — 본선은 observe_video.")

    # ── 멀티모달 생성 ────────────────────────────────────────────────────
    def _generate_mm(self, messages: List[Dict[str, Any]],
                      video_metadata: Optional[Dict[str, Any]] = None) -> str:
        import torch
        extra_kwargs: Dict[str, Any] = {}
        if video_metadata is not None:
            extra_kwargs["video_metadata"] = video_metadata
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            do_sample_frames=False,  # 2026-07-02: 안 주면 transformers가 55프레임을 24fps로 오인해
                                      # 자체 재샘플링 → 4프레임만 남음(video_grid_thw T=2 실측).
                                      # False로 우리가 이미 뽑은 프레임을 그대로 다 쓰게 강제.
            **extra_kwargs,
        ).to(self.device)
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample, repetition_penalty=self.repetition_penalty)
        return self._processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    @staticmethod
    def _even_grid_ts(idx: int, total: int, grid: List[float]) -> float:
        """timestamp 없는/파싱실패 관찰: 0:00 떼몰림 대신 grid를 관찰 순서대로 균등 배분.

        idx 번째(총 total개) 관찰에 grid 를 등간격으로 매핑. 범위 밖이면 마지막 grid 값.
        """
        if not grid:
            return 0.0
        if total <= 1:
            return grid[0]
        pos = int(round(idx * (len(grid) - 1) / (total - 1)))
        pos = max(0, min(pos, len(grid) - 1))
        return grid[pos]

    @staticmethod
    def _salvage_observations(text: str) -> List[Dict[str, Any]]:
        """잘린/깨진 JSON에서 '완성된' observation 객체만 brace-매칭으로 건진다.

        max_new_tokens 초과로 출력이 중간에 잘리면 전체 json.loads는 실패한다.
        그래도 앞쪽 관찰 객체들은 온전하므로, 균형 잡힌 {...} 조각을 개별 파싱해 살린다.
        잘린 마지막 객체는 닫히지 않아 자연히 버려진다.
        """
        import json
        out: List[Dict[str, Any]] = []
        stack: List[int] = []
        for i, ch in enumerate(text):
            if ch == "{":
                stack.append(i)
            elif ch == "}" and stack:
                frag = text[stack.pop():i + 1]
                try:
                    d = json.loads(frag)
                except Exception:
                    continue
                if isinstance(d, dict) and "action" in d:  # observation 객체만(외곽 객체 제외)
                    out.append(d)
        return out

    @staticmethod
    def _parse_observations(raw: str) -> List[Dict[str, Any]]:
        import json
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text[:4].lower() == "json":
                text = text[4:]
        try:
            start, end = text.find("{"), text.rfind("}")
            obj = json.loads(text[start:end + 1])
            obs = obj.get("observations")
            if isinstance(obs, list):
                return obs  # 정상 파싱 — 빈 배열([])도 그대로(VLM이 관찰 없다고 한 것). 정크 금지.
        except Exception:
            pass
        # 보강: 전체 파싱 실패(주로 토큰 잘림) → 완성된 관찰만 부분 복구
        salvaged = QwenVLActionExtractor._salvage_observations(text)
        if salvaged:
            logger.warning(f"[vlm] JSON 불완전 — 완성된 관찰 {len(salvaged)}건만 부분 복구(나머지 잘림)")
            return salvaged
        # 최후: 아무 것도 못 건짐 → 통째로 1건(기존 동작)
        return [{"actor": None, "action": raw.strip(), "objects_visible": []}]
