"""tacit2 엔트리포인트 — 스테이지-우선 실행.

클립-우선(클립 하나를 끝까지 → 다음 클립)이 아니라 스테이지-우선:
  STT 로드 → 전 클립 전사 → 언로드
  프레임 추출 → 전 클립 (모델 없음)
  YOLO 서브프로세스 → 전 클립 → 프로세스 소멸
  VLM 로드 → 전 클립 관찰(청크) → 언로드
  LLM 로드 → 전 클립 융합 → 언로드

→ 모델당 로드 정확히 1회. 무거운 모델 2개가 GPU 에 공존하는 순간이 구조적으로 없음.
→ unload 누락으로 배치가 전멸하는 버그 클래스 소멸.
→ 각 스테이지는 출력 파일이 있으면 skip(재실행/재개 공짜). --force 로 무시.

사용:
  python run_pipeline.py                          # 전체 (all)
  python run_pipeline.py --stages vlm fusion      # 특정 스테이지만
  python run_pipeline.py --clips CLIP4.mp4        # 특정 클립만
  python run_pipeline.py --smoke                  # 스모크: 첫 클립만 + 결과 assert
  python run_pipeline.py --force --stages vlm     # VLM 만 강제 재실행
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # 어디서 실행해도 패키지 잡히게

from core.config import load_cfg  # noqa: E402  (가벼움: yaml+pydantic 만)
from core.gpu import set_alloc_conf  # noqa: E402
from preflight import run_preflight  # noqa: E402

STAGE_ORDER = ["stt", "frames", "yolo", "vlm", "fusion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stages", nargs="+", choices=STAGE_ORDER + ["all"], default=["all"])
    ap.add_argument("--clips", nargs="+", default=None,
                    help="처리할 클립 파일명(부분 일치). 미지정 시 config 의 전체")
    ap.add_argument("--force", action="store_true", help="skip-if-exists 무시하고 재실행")
    ap.add_argument("--smoke", action="store_true",
                    help="스모크 테스트: 첫 클립 1개만 + 산출물 내용까지 assert")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    if args.clips:  # 클립 필터 (부분 일치)
        cfg.paths.videos = [v for v in cfg.paths.videos
                            if any(c in v for c in args.clips)]
        if not cfg.paths.videos:
            print(f"[run] --clips {args.clips} 에 맞는 영상 없음")
            sys.exit(1)
    if args.smoke:
        cfg.paths.videos = cfg.paths.videos[:1]
        print(f"[run] SMOKE MODE — {cfg.paths.videos[0]} 하나만")

    stages = STAGE_ORDER if "all" in args.stages else \
        [s for s in STAGE_ORDER if s in args.stages]

    run_preflight(cfg, stages)  # 무거운 import 는 이 뒤에서만
    set_alloc_conf()

    results = {}
    for stage in stages:
        t0 = time.time()
        print(f"\n{'=' * 60}\n[run] STAGE: {stage}\n{'=' * 60}")
        try:
            if stage == "stt":
                from stages import s1_stt
                s1_stt.run(cfg, force=args.force or args.smoke)
            elif stage == "frames":
                from stages import s2_frames
                s2_frames.run(cfg, force=args.force or args.smoke)
            elif stage == "yolo":
                from stages import s3_yolo
                s3_yolo.run(cfg, force=args.force or args.smoke)
            elif stage == "vlm":
                from stages import s4_vlm
                s4_vlm.run(cfg, force=args.force or args.smoke)
            elif stage == "fusion":
                from stages import s5_fusion
                s5_fusion.run(cfg, force=args.force or args.smoke)
            results[stage] = f"OK ({time.time() - t0:.0f}s)"
        except Exception as e:
            results[stage] = f"FAIL: {e}"
            print(f"[run] STAGE {stage} FAILED: {e}")
            import traceback

            traceback.print_exc()
            break  # 뒤 스테이지는 앞 산출물이 필요하므로 중단

    print(f"\n{'=' * 60}\n[SUMMARY]")
    for s, r in results.items():
        print(f"  {s:8s}: {r}")

    if args.smoke:
        _smoke_assert(cfg)

    if any(r.startswith("FAIL") for r in results.values()):
        sys.exit(1)


def _smoke_assert(cfg):
    """스모크는 '에러 없이 통과'로 부족하다 — 산출물 내용까지 assert 해야 진짜.

    (exit 0 인데 결과만 텅 비던 '관찰 0건' 사건의 재발 방지.)
    """
    import json

    from core.paths import Contract, clip_id

    contract = Contract(cfg.paths.output_dir)
    cid = clip_id(cfg.paths.videos[0])
    print(f"\n[SMOKE] 산출물 검증: {cid}")

    tr = json.loads(contract.transcript(cid).read_text(encoding="utf-8"))
    assert tr["utterances"], "SMOKE FAIL: 발화 0건"
    print(f"  transcript: {len(tr['utterances'])} utterances ✓")

    times = json.loads(contract.times(cid).read_text(encoding="utf-8"))
    assert times["n_frames"] > 0, "SMOKE FAIL: 프레임 0장"
    print(f"  frames: {times['n_frames']} ✓")

    obs = json.loads(contract.observations(cid).read_text(encoding="utf-8"))
    assert obs["n_observations"] > 0, "SMOKE FAIL: 관찰 0건 (조용한 실패!)"
    actions = [o["action"] for o in obs["observations"]]
    # 반복 퇴화 검사: dedup 후에도 동일 문장 3연속이면 실패
    import re
    norm = [re.sub(r"[\s\W]+", "", a).lower() for a in actions]
    for i in range(len(norm) - 2):
        assert not (norm[i] == norm[i + 1] == norm[i + 2]), \
            f"SMOKE FAIL: 동일 문장 3연속 — dedup 이 안 먹음: {actions[i][:40]}"
    print(f"  observations: {obs['n_observations']}건, 반복 3연속 없음 ✓")

    tac = json.loads(contract.tacit(cid).read_text(encoding="utf-8"))
    assert tac["candidates"], "SMOKE FAIL: 후보 0건"
    for c in tac["candidates"]:
        for s in c["knowledge"]["diagnostic_steps"]:
            if s["evidence"] == "utterance":
                assert s["source_utterance"], \
                    "SMOKE FAIL: evidence=utterance 인데 source_utterance 없음"
            if s["evidence"] == "action_only":
                assert s["source_utterance"] is None, \
                    "SMOKE FAIL: action_only 인데 source_utterance 있음"
    print(f"  tacit: {len(tac['candidates'])} candidates, evidence 정합 ✓")
    print("[SMOKE] ALL PASS")


if __name__ == "__main__":
    main()
