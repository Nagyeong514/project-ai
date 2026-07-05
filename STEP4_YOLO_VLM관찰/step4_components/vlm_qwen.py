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
        # 청크 분할 관찰(2026-07-05 구조 변경): 영상을 이 길이(초)의 구간으로 잘라
        # 구간마다 generate()를 따로 한다. 0/None이면 예전처럼 영상 전체 1회 생성(비권장).
        chunk_sec: Optional[float] = 40.0,
        # 멀티 GPU 배분(device="auto"일 때만 의미 있음). 숫자 하나=균등, dict({0: 21, 1: 17})=비대칭
        max_memory_gib: Optional[float | Dict[int, float]] = None,
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
        self.chunk_sec = chunk_sec
        self.max_memory_gib = max_memory_gib
        self.extra = extra
        self._model = None
        self._processor = None
        self._videos_map = self._load_videos_map(videos_map_path)
        # 직전 observe_frames() 호출의 청크별 VLM 원출력. 관찰 객체에 raw를 복제 저장하던
        # 방식(청크 원문이 관찰 건수만큼 중복되는 실측 사고)을 폐기하고, 러너가 이 값을
        # 읽어 산출물 파일 최상위 raw_by_chunk 로 청크당 1건만 저장한다.
        self.last_raw_by_chunk: Dict[str, str] = {}

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
        import os
        # 2026-07-04: OOM 에러가 직접 제안한 옵션 — reserved-but-unallocated 파편화 완화.
        # torch가 CUDA 컨텍스트를 실제로 초기화하는 시점(첫 .cuda() 호출)에 읽으므로 늦어도
        # 여기서 설정하면 늦지 않음(shell에서 export해도 되지만 안전망으로 코드에도 둠).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
        # 2026-07-04: device="auto"가 레이어를 파라미터 개수 기준으로만 나눠서 GPU 1장에
        # 쏠리고 OOM 나던 문제 — GPU당 실제 용량보다 낮게 상한을 걸어 더 고르게 분산되도록
        # 강제한다(활성화/KV캐시용 여유를 의도적으로 남겨둠).
        # 2026-07-05: 균등 배분(19GiB×2)으로도 GPU1이 21.05GiB까지 쌓임(생성 중 KV캐시가
        # GPU1 쪽에 더 쌓이는 걸로 실측됨) — 비대칭으로 걸어서 레이어를 GPU0으로 더 밀고
        # GPU1에 KV캐시 여유를 만든다. dict로 주면 {장치idx: GiB} 비대칭, 숫자 하나면 균등.
        if self.device == "auto" and self.max_memory_gib is not None and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if isinstance(self.max_memory_gib, dict):
                # 2-GPU 기준으로 적어둔 config({0:21, 1:17})가 1-GPU 할당에서도 죽지 않게
                # 실존 디바이스만 남긴다(없는 장치 idx를 accelerate에 주면 로딩이 깨짐).
                # 청크 관찰(chunk_sec) 이후로는 호출당 피크가 작아 1장으로도 돈다(2026-07-05).
                mm = {int(k): v for k, v in self.max_memory_gib.items() if int(k) < n_gpus}
                dropped = {k: v for k, v in self.max_memory_gib.items() if int(k) >= n_gpus}
                if dropped:
                    print(f"[VLM-DEVICE] max_memory 중 없는 장치 제외: {dropped} (가시 GPU {n_gpus}장)")
                quant_kwargs["max_memory"] = {k: f"{v}GiB" for k, v in mm.items()}
            else:
                quant_kwargs["max_memory"] = {i: f"{self.max_memory_gib}GiB" for i in range(n_gpus)}
            print(f"[VLM-DEVICE] max_memory 강제: {quant_kwargs['max_memory']}")
        self._processor = AutoProcessor.from_pretrained(self.model_name, max_pixels=self.max_pixels)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name, device_map=self.device,
            attn_implementation="sdpa", **quant_kwargs)
        # 2026-07-04: device="auto"로 GPU 2장을 실제로 잡았는지 확인용 — YOLO를 서브프로세스로
        # 격리해서 CUDA_VISIBLE_DEVICES 오염을 없앤 뒤에도 실제로 여러 장에 분산됐는지는
        # hf_device_map으로 직접 봐야 확실하다(nvidia-smi는 총 사용량만 보여줘서 간접적).
        hf_device_map = getattr(self._model, "hf_device_map", None)
        if hf_device_map is not None:
            devices_used = sorted(set(str(d) for d in hf_device_map.values()))
            print(f"[VLM-DEVICE] hf_device_map 사용 GPU: {devices_used} "
                  f"(2개 이상이면 멀티GPU 분산 확인됨)")
        else:
            print(f"[VLM-DEVICE] hf_device_map 없음 — device={self.device!r}로 단일 디바이스 로딩된 것")

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
                       injected_parts: Optional[List[str]] = None,
                       detections: Optional[List[Any]] = None) -> List[ActionDescription]:
        """영상을 chunk_sec 단위 시간 구간으로 잘라 구간마다 따로 관찰한다.

        2026-07-05 구조 변경(핵심). 예전엔 3분 영상 전체를 generate() 1회로 뽑았는데,
        출력이 길어질수록 4가지가 같이 무너졌다(4클립 전부 실측):
          (a) ~15번째 관찰 이후 앞 문장의 자기복제 루프로 붕괴(greedy+rep_penalty로 못 막음)
          (b) timestamp가 장면이 아니라 텍스트 패턴(+4/+6 등간격 그리드)으로 날조됨
          (c) 아예 observations:[] 로 얼어붙음(CLIP4 0건)
          (d) 시퀀스가 길어져 KV캐시 OOM 문턱을 넘음(CLIP1/CLIP4 21.97GiB OOM)
        구간당 프레임 ~20장/관찰 한 자릿수로 묶으면 넷 다 구조적으로 사라진다.

        detections: STEP4 YOLO 검출(Detection 리스트). 주면 구간마다 '그 구간에서 실제로
        검출된 클래스'만 [고정 사실]로 주입한다(전 구간 고정 주입은 그 구간에 없는 부품
        환각을 유도). injected_parts는 videos_map 손지정/폴백용 전역 목록.
        """
        self._load()
        from PIL import Image
        if not times:
            return []

        ranges = self._chunk_ranges(times, self.chunk_sec)
        print(f"[VLM-INPUT] observe_frames(): 프레임 {len(frame_paths)}개, "
              f"{times[0]:.1f}~{times[-1]:.1f}s → 청크 {len(ranges)}개(chunk_sec={self.chunk_sec})")

        out: List[ActionDescription] = []
        self.last_raw_by_chunk = {}
        for ci, (s, e) in enumerate(ranges):
            c_paths, c_times = frame_paths[s:e], times[s:e]
            t0 = c_times[0]
            local = [t - t0 for t in c_times]  # 구간을 독립된 짧은 영상처럼(00:00부터)
            parts = self._parts_for_chunk(injected_parts, detections, c_times[0], c_times[-1])
            frames = [Image.open(p).convert("RGB") for p in c_paths]

            base = build_video_observation_messages(parts)
            system_text, user_text = base[0]["content"], base[1]["content"]
            # 2026-07-02 실측: 시각을 전부 나열해 "이 중에서 골라라"라고 강제하면 모델이
            # observations=[] 로 얼어붙는다(제약이 너무 빡빡함). 대략적인 시각만 요구하고,
            # 실제 그리드 스냅은 파싱 후 snap_to_grid()가 코드로 교정한다.
            # 2026-07-05(4-F): SYSTEM/few-shot은 HH:MM:SS를 요구하는데 이 문장만 "초 단위"라
            # 지시가 상충, 실측 CLIP4에서 두 형식이 공존 → 파서(hhmmss_to_seconds)의 일차
            # 기대 형식인 HH:MM:SS로 통일 + 청크 시작/끝을 명시(전체영상 기준과의 혼란 제거).
            user_text += (f"\n\n이 프레임들은 전체 영상 중 {c_times[0]:.0f}초부터 "
                          f"{c_times[-1]:.0f}초까지의 구간이다. 다만 첨부된 구간 영상 자체는 "
                          f"00:00:00부터 시작한다. timestamp는 이 구간 영상 기준 "
                          f"00:00:00~{seconds_to_hhmmss(local[-1])} 범위 안의 값을 "
                          "HH:MM:SS 형식으로 적어라.")
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": [{"type": "video", "video": frames},
                                             {"type": "text", "text": user_text}]},
            ]
            # 2026-07-04: video_metadata 없이 넘기면 processing_qwen3_vl.replace_video_token()이
            # metadata.fps=None을 만나 임의로 fps=24를 가정해버린다(관찰 0건의 유력 원인,
            # 00_사전연구 디버그 로그로 재현 확인). 실제 프레임 간격을 명시해 원천 차단.
            real_fps = (1.0 / (local[1] - local[0])) if len(local) > 1 else 1.0
            video_metadata = {
                "total_num_frames": len(frames),
                "fps": real_fps,
                "frames_indices": list(range(len(frames))),
                "duration": local[-1] if local else None,
            }
            print(f"[VLM-CHUNK {ci+1}/{len(ranges)}] {c_times[0]:.1f}~{c_times[-1]:.1f}s "
                  f"프레임 {len(frames)}개, 주입 부품={parts}")
            raw = self._generate_mm(messages, video_metadata=video_metadata)
            print(f"[VLM-RAW chunk {ci+1}] len={len(raw)}\n{'-'*70}\n{raw}\n{'-'*70}")
            self.last_raw_by_chunk[f"{c_times[0]:.0f}-{c_times[-1]:.0f}"] = raw

            parsed = self._parse_observations(raw)
            if not parsed:
                logger.warning(f"[vlm] chunk {ci+1}/{len(ranges)} "
                               f"({c_times[0]:.0f}~{c_times[-1]:.0f}s) 관찰 0건")
            n = len(parsed)
            for i, obs in enumerate(parsed):
                ts = hhmmss_to_seconds(obs.get("timestamp"))
                if ts is not None:
                    ts = snap_to_grid(ts, local)  # 구간 내 실제 프레임 그리드로 교정
                else:
                    # 파싱 실패/누락: 0:00 떼몰림 대신 이 청크 grid를 관찰 순서대로 균등 배분.
                    ts = self._even_grid_ts(i, n, local)
                out.append(ActionDescription(
                    timestamp=t0 + ts,
                    actor=obs.get("actor"),
                    action=str(obs.get("action", "")).strip(),
                    objects=list(obs.get("objects_visible", [])),
                    chunk=f"{c_times[0]:.0f}-{c_times[-1]:.0f}"))

        self._log_gpu_peak()  # GPU 0/1 각각 peak 확인용(전 청크 누적 최대)

        # dedup(4-B): 전체 청크 통합 후 시간순 정렬 → 최근 2건 비교로 접기(청크 경계
        # 넘는 반복 + A,B 교대 패턴 포착). 빈 action은 정보가 없으므로 제외.
        before = len(out)
        merged = self.dedup_merge([o for o in out if o.action])
        if len(merged) != before:
            print(f"[VLM-DEDUP] 관찰 {before}건 → {len(merged)}건 "
                  f"(동일 문장 {before - len(merged)}건을 repeat_count로 압축)")
        return merged

    @staticmethod
    def _norm_action(s: str) -> str:
        """dedup 비교용 정규화 — 공백/비단어문자 제거 + 소문자화(사소한 표기 차이 흡수)."""
        import re
        return re.sub(r"[\s\W]+", "", s).lower()

    @staticmethod
    def dedup_merge(observations: List[ActionDescription]) -> List[ActionDescription]:
        """전체 청크 통합본을 시간순 정렬 후, 직전 유지분 '최근 2건'과 정규화 action이
        같으면 접는다(2026-07-05, 4-B — tacit2 dedup_merge 이식).

        예전 _dedup_chunk(청크 안에서만 + 완전 동일 문자열만 + action 접미사 표기)의 한계:
        청크를 넘나드는 반복(CLIP1 60~190초)과 A,B,A,B 교대 반복을 못 잡았다. 접을 때
        삭제가 아니라 repeat_count 증가 + end_timestamp 갱신(별도 필드) — 반복 사실은
        STEP5 직렬화(_serialize)가 LLM에 전달.

        비교 창은 최근 4건(2026-07-05 실측으로 2→4 확대): CLIP1 160~196s에서 모델이
        4문장(위/좌/우/아래 방향만 바꾼 동일 템플릿, objects 완전 동일, 2초 기계적
        등간격)을 순환 재활용하는 퇴화가 2건 창을 통과함. 정규화 완전일치만 접으므로
        창을 넓혀도 오접합 위험은 낮다(진짜 재등장이 접혀도 repeat_count/end_timestamp
        로 횟수·구간이 보존됨).
        """
        obs = sorted(observations, key=lambda o: o.timestamp)
        kept: List[ActionDescription] = []
        for o in obs:
            merged = False
            for prev in kept[-4:]:
                if QwenVLActionExtractor._norm_action(prev.action) == \
                        QwenVLActionExtractor._norm_action(o.action):
                    prev.repeat_count += o.repeat_count
                    prev.end_timestamp = max(prev.end_timestamp or prev.timestamp, o.timestamp)
                    merged = True
                    break
            if not merged:
                kept.append(o)
        return QwenVLActionExtractor._collapse_screen_runs(kept)

    @staticmethod
    def _is_screen_gaze(o: ActionDescription) -> bool:
        """objects가 monitor 단독이고 actor가 '시선'인 순수 화면 상태 관찰인지."""
        objs = [str(x).strip().lower() for x in (o.objects or [])]
        return objs == ["monitor"] and "시선" in str(o.actor or "")

    @staticmethod
    def _has_concrete_info(action: str) -> bool:
        """화면 관찰 문장에 구체 수치/용량/버전/모델명이 담겨 있는지(4-E 예외).

        "65536 MB", "DDR4", "BIOS Version 2.6.0"처럼 숫자가 낀 관찰은 STEP5가 뽑아야
        할 실제 정보라 접으면 안 된다. 숫자 포함이면 보존(과잉 보존 쪽으로 오차) —
        접는 대상은 "로고 떴다/창 바뀌었다"류 무숫자 화면전환 관찰만.
        """
        import re
        return bool(re.search(r"\d", action))

    @staticmethod
    def _collapse_screen_runs(kept: List[ActionDescription]) -> List[ActionDescription]:
        """같은 청크 안에서 '화면 상태 관찰'(monitor 단독 + 시선)이 연달아 여러 건이면
        대표 1건으로 접는다(2026-07-05, 4-E — dedup_merge 확장, tacit2에도 없는 신규 로직).

        정적/저정보 구간(화면 꺼짐·부팅 루프)에서 VLM이 "화면에 X가 뜬다"류 문장을 조금씩
        바꿔가며 다발로 지어내는 실측 사고(CLIP4 120~154초, Windows/Ubuntu/Dell 창 혼재
        환각) 대응 — 글자가 매번 달라 정규화 dedup은 통과한다. action 핵심 명사가 서로
        달라도 화면 관찰 연속 run 이면 첫 건을 대표로 남기고 나머지는 repeat_count/
        end_timestamp 로 접는다. 다른 관찰이 사이에 끼거나 청크가 바뀌면 run 이 끊긴다.

        예외: 구체 수치/용량/버전이 담긴 화면 관찰(_has_concrete_info — 예: "65536 MB",
        "DDR4", "BIOS Version 2.6.0")은 STEP5가 뽑아야 할 실제 정보이므로 접지 않고
        개별 보존한다(run 을 끊는 독립 관찰로 취급).
        """
        out: List[ActionDescription] = []
        run: List[ActionDescription] = []

        def flush():
            if not run:
                return
            head = run[0]
            if len(run) > 1:
                head.repeat_count += sum(o.repeat_count for o in run[1:])
                last = run[-1]
                head.end_timestamp = max(head.end_timestamp or head.timestamp,
                                         last.end_timestamp or last.timestamp)
                logger.warning(f"[vlm] 화면 상태 관찰 연속 {len(run)}건을 대표 1건으로 압축 "
                               f"(chunk {head.chunk}, {head.timestamp:.0f}~{head.end_timestamp:.0f}s)")
            out.append(head)
            run.clear()

        for o in kept:
            collapsible = (QwenVLActionExtractor._is_screen_gaze(o)
                           and not QwenVLActionExtractor._has_concrete_info(o.action))
            if collapsible and (not run or o.chunk == run[0].chunk):
                run.append(o)
            else:
                flush()
                if collapsible:
                    run.append(o)
                else:
                    out.append(o)
        flush()
        return out

    @staticmethod
    def _chunk_ranges(times: List[float], chunk_sec: Optional[float]) -> List[tuple]:
        """times를 chunk_sec 길이의 연속 구간 [start_idx, end_idx) 리스트로 자른다.

        chunk_sec가 None/0 이하면 전체 1구간(예전 동작). 마지막 꼬리 구간이 3프레임
        미만이면 직전 구간에 병합한다(프레임 1~2장짜리 '영상'은 관찰 의미도 없고
        processor의 비디오 처리도 불안정).
        """
        if not chunk_sec or chunk_sec <= 0:
            return [(0, len(times))]
        ranges: List[tuple] = []
        start = 0
        for i, t in enumerate(times):
            if t - times[start] >= chunk_sec:
                ranges.append((start, i))
                start = i
        ranges.append((start, len(times)))
        if len(ranges) >= 2 and ranges[-1][1] - ranges[-1][0] < 3:
            s, _ = ranges[-2]
            ranges[-2] = (s, ranges[-1][1])
            ranges.pop()
        return ranges

    def _parts_for_chunk(self, global_parts: Optional[List[str]],
                         detections: Optional[List[Any]],
                         t0: float, t1: float) -> Optional[List[str]]:
        """이 구간의 [고정 사실] 주입 목록. 구간 내 YOLO 검출 클래스 > 전역 목록 순."""
        if not self.part_injection:
            return None
        if detections:
            cls = sorted({d.cls for d in detections if t0 <= d.timestamp <= t1})
            if cls:
                return cls
        return global_parts

    def describe_actions(self, frames, detections_by_frame):
        """[옵션] 프레임-리스트 모드. 본선은 observe_video. (미사용)"""
        raise NotImplementedError("프레임-리스트 모드는 옵션 — 본선은 observe_video.")

    @staticmethod
    def _log_gpu_peak() -> None:
        """GPU 0/1 각각의 peak 메모리를 찍는다(2026-07-05: max_new_tokens를 올린 뒤
        2-GPU 각각 얼마나 쓰는지 확인용 — torch.cuda.memory_allocated()는 인자 없이 쓰면
        현재 디바이스 하나만 보여서 멀티GPU에선 오해를 부른다)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.max_memory_allocated(i) / 1024**3
                reserved = torch.cuda.max_memory_reserved(i) / 1024**3
                print(f"[VLM-GPU-PEAK] GPU{i}: peak_allocated={alloc:.2f}GiB peak_reserved={reserved:.2f}GiB")
        except Exception as e:
            print(f"[VLM-GPU-PEAK] 측정 실패: {e}")

    def _log_vision_input(self, inputs: Dict[str, Any]) -> None:
        """model.generate() 직전 최종 input에 비디오 토큰이 실제로 몇 개 들어갔는지 찍는다.

        2026-07-04: fps를 0.15/0.3/0.5로 바꿔가며 돌려도 raw 출력이 완전히 동일하다는 제보 —
        프레임 밀도 문제가 아니라 프레임이 애초에 모델 입력에 안 들어가고 있을 가능성 확인용.
        `video_grid_thw`/`pixel_values_videos`가 없거나 vision 토큰 카운트가 0/극소수면
        원인이 여기(입력 단계)에 있는 것이고, 정상 범위인데도 raw가 동일하면 원인은 다른 곳
        (예: generate()의 캐시/샘플링 설정)에 있다는 뜻 — 이 로그로 둘을 구분할 것.
        """
        input_ids = inputs.get("input_ids")
        print(f"[VLM-INPUT] keys={list(inputs.keys())}")
        if input_ids is not None:
            print(f"[VLM-INPUT] input_ids.shape={tuple(input_ids.shape)}")
        video_grid_thw = inputs.get("video_grid_thw")
        if video_grid_thw is not None:
            print(f"[VLM-INPUT] video_grid_thw={video_grid_thw.tolist()}")
        else:
            print("[VLM-INPUT] video_grid_thw 없음 — 비디오가 아예 인식 안 됐을 가능성")
        pixel_values_videos = inputs.get("pixel_values_videos")
        if pixel_values_videos is not None:
            print(f"[VLM-INPUT] pixel_values_videos.shape={tuple(pixel_values_videos.shape)}")
        else:
            print("[VLM-INPUT] pixel_values_videos 없음 — 비디오가 아예 인식 안 됐을 가능성")
        video_token_id = getattr(self._processor, "video_token_id", None)
        if input_ids is not None and video_token_id is not None:
            n_video_tokens = int((input_ids == video_token_id).sum().item())
            print(f"[VLM-INPUT] video_token_id={video_token_id}, 실제 개수={n_video_tokens} "
                  f"(0이거나 프레임 수 대비 극소수면 vision 입력이 안 들어간 것)")
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            nonzero = int((mm_token_type_ids != 0).sum().item())
            print(f"[VLM-INPUT] mm_token_type_ids 중 비-텍스트(멀티모달) 토큰 개수={nonzero}")

    # ── 멀티모달 생성 ────────────────────────────────────────────────────
    def _generate_mm(self, messages: List[Dict[str, Any]],
                      video_metadata: Optional[Dict[str, Any]] = None) -> str:
        import torch
        extra_kwargs: Dict[str, Any] = {}
        if video_metadata is not None:
            extra_kwargs["video_metadata"] = video_metadata
        # device="auto"(멀티 GPU 분산)면 ".to('auto')"가 안 통함 → 모델의 첫 레이어 디바이스로
        # 보낸다(llm_fusion.py의 동일 패턴 참고). accelerate가 forward 중 내부적으로 레이어별
        # 디바이스 이동을 알아서 처리하므로 입력은 첫 디바이스에만 올려두면 된다.
        target_device = self._model.device if self.device == "auto" else self.device
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            do_sample_frames=False,  # 2026-07-02: 안 주면 transformers가 55프레임을 24fps로 오인해
                                      # 자체 재샘플링 → 4프레임만 남음(video_grid_thw T=2 실측).
                                      # False로 우리가 이미 뽑은 프레임을 그대로 다 쓰게 강제.
            **extra_kwargs,
        ).to(target_device)
        self._log_vision_input(inputs)  # 2026-07-04: fps 올려도 raw가 동일하다는 제보 — vision
                                          # 토큰이 실제로 몇 개 들어갔는지 매 호출마다 찍어서 확인.
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
        # 최후: 아무 것도 못 건짐 → 빈 결과 + 경고(2026-07-05 변경). 예전엔 raw 전문을
        # action 1건으로 밀어넣었는데, 그 비정형 덩어리가 STEP5 융합 LLM 입력까지 그대로
        # 흘러가 오염시킨다. raw는 어차피 [VLM-RAW] 로그와 산출물 raw_by_chunk에 남는다.
        logger.warning(f"[vlm] JSON 파싱 완전 실패 — 이 청크 관찰 0건 처리(raw {len(raw)}자)")
        return []
