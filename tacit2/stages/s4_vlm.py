"""S4: VLM 관찰 — 청크(30초) 단위 호출. 이 파일이 v2 의 심장이다.

세 증상의 공통 뿌리였던 "영상 통짜 1회 호출"을 폐기한다:
  - OOM        → 호출당 프레임 ≤15장 상한이라 vision 토큰이 작고 일정. GPU 1장 고정.
  - 관찰 0건    → 실패해도 청크 하나로 국소화. 빈 청크는 1회 재시도(temperature 샘플링).
  - 반복 퇴화   → 생성 길이가 짧아 루프에 빠질 활주로 자체가 없음 + 병합 시 dedup 안전망.

방어 계측(조용한 실패 금지):
  - 청크마다 video_grid_thw 의 T 값을 assert — 프레임 축소 버그(55→4) 재발 시 즉사.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from core.config import Cfg
from core.gpu import log_mem, release, set_alloc_conf
from core.paths import Contract, clip_id
from core.schema import Observation
from core.timeline import even_spread, parse_ts, snap_to_grid
from prompts.vlm_prompt import build_chunk_messages


# ─────────────────────────── 청크 분할 ───────────────────────────

def make_chunks(times: List[float], chunk_sec: float, overlap: float,
                max_frames: int) -> List[Tuple[float, float, List[int]]]:
    """[(start, end, frame_indices), ...]. 창당 프레임이 상한을 넘으면 균등 서브샘플."""
    if not times:
        return []
    duration = times[-1]
    chunks = []
    t0 = 0.0
    step = max(chunk_sec - overlap, 1.0)
    while t0 <= duration:
        t1 = t0 + chunk_sec
        idxs = [i for i, t in enumerate(times) if t0 <= t < t1]
        if idxs:
            if len(idxs) > max_frames:
                stride = len(idxs) / max_frames
                idxs = [idxs[int(k * stride)] for k in range(max_frames)]
            chunks.append((t0, min(t1, duration), idxs))
        t0 += step
    return chunks


# ─────────────────────────── 파싱 (3단 폴백) ───────────────────────────

def _strip_fence(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).strip()


def _salvage(text: str) -> List[dict]:
    """brace 스택 매칭으로 잘린 출력에서 '완성된' 관찰 객체만 부분 복구.

    중첩을 전부 추적한다 — 바깥 wrapper 가 잘려 안 닫혀도,
    내부에 온전히 닫힌 {"action": ...} 객체들은 각각 건진다.
    """
    out, starts = [], []
    for i, ch in enumerate(text):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            s = starts.pop()
            try:
                d = json.loads(text[s:i + 1])
                if isinstance(d, dict) and "action" in d:
                    out.append(d)
            except Exception:
                pass
    return out


def parse_observations(raw: str) -> List[dict]:
    text = _strip_fence(raw)
    try:
        s, e = text.find("{"), text.rfind("}")
        obj = json.loads(text[s:e + 1])
        obs = obj.get("observations", [])
        if isinstance(obs, list):
            return [d for d in obs if isinstance(d, dict)]  # 빈 배열도 그대로 인정
    except Exception:
        pass
    return _salvage(text)


# ─────────────────────────── dedup 병합 (2차 방어) ───────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[\s\W]+", "", s).lower()


def dedup_merge(observations: List[Observation]) -> List[Observation]:
    """시간순 정렬 후, 직전 유지분과 동일 문장이면 접고 지속 구간만 늘린다.

    반복 2문장 교대(A,B,A,B,...) 패턴도 잡기 위해 직전 '2건'과 비교한다.
    """
    obs = sorted(observations, key=lambda o: o.timestamp)
    kept: List[Observation] = []
    for o in obs:
        merged = False
        for prev in kept[-2:]:
            if _norm(prev.action) == _norm(o.action):
                prev.repeat_count += 1
                prev.end_timestamp = max(prev.end_timestamp or prev.timestamp, o.timestamp)
                merged = True
                break
        if not merged:
            kept.append(o)
    return kept


# ─────────────────────────── 모델 어댑터 ───────────────────────────

class VlmRunner:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        v = self.cfg.vlm
        kwargs = {"device_map": v.device}  # 한 장 고정 — auto 줄다리기 폐지
        if v.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,  # Turing: fp16 고정
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        print(f"[S4] loading VLM {v.model_name} → {v.device}")
        self.model = AutoModelForImageTextToText.from_pretrained(v.model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(v.model_name, max_pixels=v.max_pixels)
        log_mem("S4 loaded")

    def _generate(self, messages, n_frames: int, do_sample: bool) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        v = self.cfg.vlm
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        try:
            imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)
        except TypeError:  # 구버전 qwen_vl_utils
            imgs, vids = process_vision_info(messages)
            vkw = {}
        inputs = self.processor(
            text=[text], images=imgs, videos=vids,
            do_sample_frames=False,  # 리샘플링 차단(55→4 축소 버그)
            return_tensors="pt", **vkw,
        )

        # ★ 방어 계측: T 값 assert — 프레임이 조용히 잘려나가면 즉사시킨다
        if "video_grid_thw" in inputs:
            T = int(inputs["video_grid_thw"][0][0])
            expected_min = max(1, n_frames // 2)  # temporal patch=2
            print(f"[VLM-INPUT] frames={n_frames} grid_T={T}")
            assert T >= expected_min, (
                f"프레임 축소 버그 재발: frames={n_frames}인데 grid_T={T} "
                f"(기대 최소 {expected_min}). do_sample_frames/fps 설정 확인.")

        inputs = inputs.to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=v.max_new_tokens,
            repetition_penalty=v.repetition_penalty,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=v.retry_temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def observe_chunk(self, frame_paths, t0, t1, parts, fps) -> Tuple[List[dict], str]:
        messages = build_chunk_messages(frame_paths, t0, t1, parts, fps)
        raw = self._generate(messages, len(frame_paths), do_sample=False)
        obs = parse_observations(raw)
        if not obs:  # 빈 청크 → 1회 재시도(그리디 루프 탈출용 샘플링)
            print(f"[S4] chunk {t0:.0f}-{t1:.0f}s empty → retry (temp={self.cfg.vlm.retry_temperature})")
            raw = self._generate(messages, len(frame_paths), do_sample=True)
            obs = parse_observations(raw)
        return obs, raw

    def unload(self):
        release(self.model, self.processor)
        self.model = self.processor = None
        log_mem("S4 released")


# ─────────────────────────── 스테이지 진입 ───────────────────────────

def _load_detections(det_path: Path) -> List[dict]:
    """[{timestamp, cls}, ...] — 청크별 필터용."""
    if not det_path.exists():
        return []
    try:
        return json.loads(det_path.read_text(encoding="utf-8")).get("detections", [])
    except Exception:
        return []


def _parts_for_chunk(dets: List[dict], t0: float, t1: float) -> Optional[List[str]]:
    """이 청크 시간대에 실제 검출된 클래스만 주입(구간에 없는 부품명 주입 → 환각 유도 차단)."""
    classes = sorted({d["cls"] for d in dets if t0 <= d.get("timestamp", -1) < t1})
    return classes or None


def run(cfg: Cfg, force: bool = False):
    set_alloc_conf()
    contract = Contract(cfg.paths.output_dir)
    contract.ensure_dirs()

    todo = []
    for v in cfg.paths.videos:
        cid = clip_id(v)
        if not force and contract.observations(cid).exists():
            print(f"[S4] skip (exists): {cid}")
            continue
        if not contract.times(cid).exists():
            print(f"[S4] WARN: {cid} 프레임 없음 — S2 먼저 실행 필요. 건너뜀")
            continue
        todo.append(cid)
    if not todo:
        return

    runner = VlmRunner(cfg)
    runner.load()
    try:
        for cid in todo:
            meta = json.loads(contract.times(cid).read_text(encoding="utf-8"))
            times, fps = meta["times"], meta["fps"]
            fdir = contract.frames_dir(cid)
            frames = sorted(str(p) for p in fdir.glob("frame_*.jpg"))
            dets = _load_detections(contract.detections(cid))

            chunks = make_chunks(times, cfg.vlm.chunk_sec,
                                 cfg.vlm.chunk_overlap_sec, cfg.vlm.max_frames_per_chunk)
            print(f"[S4] {cid}: {len(frames)} frames → {len(chunks)} chunks")

            all_obs: List[Observation] = []
            raw_log = {}
            for (t0, t1, idxs) in chunks:
                chunk_frames = [frames[i] for i in idxs]
                chunk_grid = [times[i] for i in idxs]
                parts = _parts_for_chunk(dets, t0, t1)
                obs_dicts, raw = runner.observe_chunk(chunk_frames, t0, t1, parts, fps)
                raw_log[f"{t0:.0f}-{t1:.0f}"] = raw

                missing_ts = [d for d in obs_dicts if d.get("timestamp") in (None, "", "null")]
                fills = iter(even_spread(len(missing_ts), t0, t1))
                for d in obs_dicts:
                    ts = d.get("timestamp")
                    sec = next(fills) if d in missing_ts else parse_ts(ts)
                    sec = min(max(sec, t0), t1)          # 청크 범위로 클램프
                    sec = snap_to_grid(sec, chunk_grid)   # 실제 프레임 시각으로 스냅
                    all_obs.append(Observation(
                        timestamp=sec,
                        actor=str(d.get("actor", "")),
                        action=str(d.get("action", "")).strip(),
                        objects=[str(x) for x in d.get("objects_visible", []) or []],
                        chunk=f"{t0:.0f}-{t1:.0f}",
                    ))
                print(f"[S4]   chunk {t0:.0f}-{t1:.0f}s: {len(obs_dicts)} obs")

            before = len(all_obs)
            merged = dedup_merge([o for o in all_obs if o.action])
            print(f"[S4] {cid}: dedup {before} → {len(merged)}")

            out = contract.observations(cid)
            out.write_text(json.dumps({
                "video_id": cid,
                "n_observations": len(merged),
                "n_raw": before,
                "observations": [o.model_dump() for o in merged],
                "raw_by_chunk": raw_log,  # 디버깅용 원문 보존
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[S4] saved {out}")
    finally:
        runner.unload()
