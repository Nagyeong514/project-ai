"""S3: YOLO 검출 — 서브프로세스 완전 격리.

왜 서브프로세스인가(5일 삽질의 근본 원인):
  ultralytics 가 .predict() 최초 호출 시 os.environ["CUDA_VISIBLE_DEVICES"] 를
  무조건 덮어써서, 같은 프로세스의 이후 VLM/LLM 이 GPU 를 못 잡는다.
  자식 프로세스는 환경변수 '복사본'만 받으므로 오염이 부모로 안 샌다.

cuDNN 로딩 실패(CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED) 처방:
  자식 env 의 LD_LIBRARY_PATH 에 venv 안 nvidia cudnn/cublas lib 경로를 '앞에 추가'
  (덮어쓰기 금지 — libnvrtc.so.13 유실 사건, 체크리스트 2단계 함정).

실패해도 파이프라인은 계속 간다: 검출 0건 = 빈 detections.json.
YOLO 는 부품주입 보조라 없어도 VLM/LLM 은 돈다.
"""
from __future__ import annotations

import json
import os
import site
import subprocess
import sys
from pathlib import Path

from core.config import Cfg
from core.paths import Contract, clip_id

ENTRY = Path(__file__).parent / "s3_yolo_entry.py"


def _child_env() -> dict:
    env = os.environ.copy()
    # venv 의 pip nvidia 패키지 lib 들을 LD_LIBRARY_PATH 앞에 추가
    lib_dirs = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia = Path(sp) / "nvidia"
        if nvidia.is_dir():
            lib_dirs += [str(p) for p in nvidia.glob("*/lib") if p.is_dir()]
    if lib_dirs:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ((":" + prev) if prev else "")
    return env


def run(cfg: Cfg, force: bool = False):
    contract = Contract(cfg.paths.output_dir)
    contract.ensure_dirs()
    if not cfg.yolo.enabled:
        print("[S3] yolo.enabled=false — 전 클립 빈 검출로 통과")
        for v in cfg.paths.videos:
            cid = clip_id(v)
            if force or not contract.detections(cid).exists():
                contract.detections(cid).write_text(
                    json.dumps({"video_id": cid, "detections": []}, ensure_ascii=False),
                    encoding="utf-8")
        return

    env = _child_env()
    for video in cfg.paths.videos:
        cid = clip_id(video)
        out = contract.detections(cid)
        if not force and out.exists():
            print(f"[S3] skip (exists): {cid}")
            continue
        if not contract.times(cid).exists():
            print(f"[S3] WARN: {cid} 프레임 없음 — S2 먼저. 빈 검출로 통과")
            out.write_text(json.dumps({"video_id": cid, "detections": []}), encoding="utf-8")
            continue

        cmd = [
            sys.executable, str(ENTRY),
            "--frames-dir", str(contract.frames_dir(cid)),
            "--times-json", str(contract.times(cid)),
            "--weights", cfg.paths.yolo_weights,
            "--out", str(out),
            "--imgsz", str(cfg.yolo.imgsz),
            "--conf", str(cfg.yolo.conf),
            "--batch", str(cfg.yolo.batch),
        ]
        print(f"[S3] detect (subprocess): {cid}")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[S3] WARN: YOLO 실패 — 빈 검출로 계속 진행\n--- stderr tail ---\n{r.stderr[-2000:]}")
            out.write_text(json.dumps({"video_id": cid, "detections": [],
                                       "error": r.stderr[-500:]}, ensure_ascii=False),
                           encoding="utf-8")
        else:
            n = json.loads(out.read_text(encoding="utf-8")).get("n", "?")
            print(f"[S3] saved {out} ({n} detections)")
