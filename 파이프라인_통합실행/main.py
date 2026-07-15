# -*- coding: utf-8 -*-
"""
파이프라인 원샷 오케스트레이터 (2026-07-14).

클립을 넣으면 STEP3→4→5→adapter→STEP6 을 한 번에 돌린다. "여기서 뭘 돌리면 전체가
도느냐"의 정답 진입점.

왜 subprocess 인가:
  STEP3~5(통합러너) / STEP6 는 venv 가 다르다(langchain·transformers 버전 충돌, CLAUDE.md
  규칙). 한 파이썬 프로세스로 못 합치므로, 이 main 은 각 단계를 '그 단계의 venv 파이썬'으로
  subprocess 호출한다. 단계 사이 데이터는 전부 파일로 주고받는다(통합러너 설계 원칙 그대로).

포함 범위: STEP3~7 (후보 생성 → 품질검증 → 벡터DB 적재).
  STEP6→STEP7 은 step7_adapter 가 스키마 변환(decision→routing, confidence→{score}).
제외:
  - STEP8: FastAPI+uvicorn 상주 웹 서비스라 배치 체인 대상 아님(끝에 기동 명령만 안내).

실행:
    # GPU 노드에서 (STEP3~5 통합러너가 GPU 필요)
    srun -p RTX6000 -w n5 --gres=gpu:2 파이프라인_통합실행/.venv/bin/python3 main.py
    # 출력 태그 지정
    python3 main.py --run-tag myrun
    # STEP6 판정자 오버라이드(격리 실험)
    python3 main.py --judge-model Qwen/Qwen2.5-14B-Instruct
    # DB 적재 생략(STEP6까지만)
    python3 main.py --skip-db
    # DB에 accept만 적재(기본: 전건 + routing 태그 보존)
    python3 main.py --accept-only-db
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

ROOT = "/home/ai_user/team_a2/members/안나경/project-ai"

PIPE_PY = os.path.join(ROOT, "파이프라인_통합실행", ".venv", "bin", "python3")
STEP6_PY = os.path.join(ROOT, "STEP6_암묵지_품질검증", ".venv", "bin", "python3")
STEP7_PY = os.path.join(ROOT, "STEP7_DB", ".venv", "bin", "python3")

TACIT_DIR = os.path.join(ROOT, "STEP5_LLM_암묵지_후보생성", "output", "tacit_json")
ADAPTER_PY = os.path.join(ROOT, "STEP5_LLM_암묵지_후보생성", "step6_adapter.py")
STEP7_ADAPTER_PY = os.path.join(ROOT, "STEP6_암묵지_품질검증", "step7_adapter.py")
STEP7_BUILD_PY = os.path.join(ROOT, "STEP7_DB", "build_db.py")


def _run(cmd: list[str], cwd: str, env: dict | None = None) -> None:
    """단계 하나를 실행. 실패(비정상 종료)하면 전체 중단."""
    print(f"\n$ (cwd={cwd})\n  {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        print(f"[FAIL] 종료코드 {r.returncode} — 파이프라인 중단", flush=True)
        sys.exit(r.returncode)


def _count(pattern: str) -> int:
    return len(glob.glob(pattern))


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP3~6 원샷 파이프라인")
    ap.add_argument("--run-tag", default="e2e_run", help="adapter/STEP6 출력 폴더 태그")
    ap.add_argument("--judge-model", default=None,
                    help="STEP6 판정자 오버라이드(예: Qwen/Qwen2.5-14B-Instruct). 미지정 시 config 기본값")
    ap.add_argument("--skip-db", action="store_true", help="STEP7 DB 적재 생략(STEP6까지만)")
    ap.add_argument("--accept-only-db", action="store_true", help="DB에 accept만 적재(기본: 전건+routing 태그)")
    args = ap.parse_args()

    adapter_out = os.path.join(ROOT, "STEP5_LLM_암묵지_후보생성", "output",
                               "step6_adapter_input", args.run_tag)
    step6_out = os.path.join("output", args.run_tag)  # STEP6 cwd 기준 상대경로

    # ── [1/3] STEP3→4→5 통합러너 (후보 생성) ──────────────────────────────
    print("=" * 60 + "\n[1/3] STEP3→4→5 통합러너 (후보 생성)\n" + "=" * 60)
    _run([PIPE_PY, "run_all_clips.py"], cwd=os.path.join(ROOT, "파이프라인_통합실행"))
    n_tacit = _count(os.path.join(TACIT_DIR, "*.tacit.json"))
    print(f"  → tacit_json 산출: {n_tacit}건")
    if n_tacit == 0:
        print("[FAIL] STEP5 후보 0건 — 중단"); sys.exit(1)

    # ── [2/3] STEP5→STEP6 어댑터 (입력 형식 변환) ─────────────────────────
    print("=" * 60 + "\n[2/3] STEP5→STEP6 어댑터\n" + "=" * 60)
    # step6_adapter.py 는 표준 라이브러리만 써서 어느 venv로도 실행 가능(파이프라인 venv 재사용).
    _run([PIPE_PY, ADAPTER_PY, "--tacit-dir", TACIT_DIR, "--output", adapter_out],
         cwd=os.path.join(ROOT, "파이프라인_통합실행"))
    n_adapt = _count(os.path.join(adapter_out, "*.json"))
    print(f"  → 변환 후보: {n_adapt}건 ({adapter_out})")
    if n_adapt == 0:
        print("[FAIL] 어댑터 변환 0건 — 중단"); sys.exit(1)

    # ── [3/3] STEP6 품질검증 (accept/hold/reject) ─────────────────────────
    print("=" * 60 + "\n[3/3] STEP6 품질검증\n" + "=" * 60)
    step6_env = dict(os.environ)
    if args.judge_model:
        step6_env["STEP6_LLM_MODEL"] = args.judge_model  # config.py 가 이 env 를 읽어 판정자 교체
    step6_dir = os.path.join(ROOT, "STEP6_암묵지_품질검증")
    _run([STEP6_PY, "main.py", "--input", adapter_out, "--output_dir", step6_out],
         cwd=step6_dir, env=step6_env)
    step6_out_abs = os.path.join(step6_dir, step6_out)
    n_verified = _count(os.path.join(step6_out_abs, "*_verified.json"))
    print(f"  → 판정 완료: {n_verified}건 ({step6_out_abs})")

    if args.skip_db:
        print("\n" + "=" * 60)
        print(f"[DONE] STEP3~6 완료 (--skip-db) — 후보 {n_tacit} → 변환 {n_adapt} → 판정 {n_verified}")
        print("=" * 60)
        return

    # ── [4/4] STEP6→STEP7 어댑터 + STEP7 DB 적재 ──────────────────────────
    print("=" * 60 + "\n[4/4] STEP6→STEP7 어댑터 + STEP7 벡터DB 적재\n" + "=" * 60)
    step7_in = os.path.join(ROOT, "STEP7_DB", "_step7_input", args.run_tag)
    step7_dir = os.path.join(ROOT, "STEP7_DB")
    # 어댑터(표준 라이브러리만) → STEP6 판정 결과를 STEP7 입력 스키마로 변환
    adp_cmd = [PIPE_PY, STEP7_ADAPTER_PY, "--input", step6_out_abs, "--output", step7_in]
    if args.accept_only_db:
        adp_cmd.append("--accept-only")
    _run(adp_cmd, cwd=step7_dir)
    n_db_in = _count(os.path.join(step7_in, "*.json"))
    print(f"  → STEP7 입력 변환: {n_db_in}건 ({step7_in})")
    if n_db_in == 0:
        print("[FAIL] STEP7 적재 대상 0건 — 중단"); sys.exit(1)
    # STEP7 build_db (자기 venv)
    _run([STEP7_PY, STEP7_BUILD_PY, "--input_dir", step7_in], cwd=step7_dir)

    print("\n" + "=" * 60)
    print(f"[DONE] STEP3~7 원샷 완료 — 후보 {n_tacit} → 변환 {n_adapt} → 판정 {n_verified} → DB적재 {n_db_in}")
    print("  STEP8(RAG 서비스)은 상주 웹서버라 배치에 미포함. 별도 기동:")
    print("    cd STEP8_RAG서비스 && .venv/bin/uvicorn app:app --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
