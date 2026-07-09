"""
융합 LLM 선정 벤치 실행기 — 모델 1개 × (웜업 1 + 본 실행 N회) (2026-07-09).

사용(GPU 노드, 모델별 순차 — 동시 실행 금지):
    python3 run_bench.py --model qwen3vl [--precision fp16] [--runs 3]

원칙(docs/실행전_방어_체크리스트.md):
  - 무거운 import(torch/transformers)는 프리플라이트 통과 후로 미룬다.
  - 판단 기준은 로딩 크기가 아니라 forward peak — 로드 직후(정적)와 생성 중 peak를
    구분 기록(torch.cuda.max_memory_allocated + nvidia-smi 교차).
  - OOM 시: 예외 객체 보관 금지(메시지만), gc+empty_cache 후 다음 윈도우 계속
    (한 윈도우 실패로 잡 전체를 죽이지 않는다 — 실패도 데이터다).
  - 결과 덮어쓰기 금지: results/{model}/{precision}/run{N} 존재 시 즉사.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "STEP5_LLM으로_암묵지_후보생성"))

import yaml  # noqa: E402


# ── 프리플라이트(가볍게 — 5초 안에 죽을 수 있는 검증만) ──────────────

def preflight(cfg: dict, model_key: str, precision: str) -> dict:
    problems = []
    if model_key not in cfg["models"]:
        problems.append(f"모델 키 없음: {model_key} (등록: {list(cfg['models'])})")
    for f in ("windows_labels.json", "frozen_windows.json"):
        if not (HERE / f).exists():
            problems.append(f"{f} 없음 — prepare_windows.py 먼저")
    import importlib
    for mod in ["torch", "transformers"] + (["bitsandbytes"] if precision == "nf4" else []):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            problems.append(f"{mod} import 실패: {e}")
    import torch
    if not torch.cuda.is_available():
        problems.append("CUDA 사용 불가(로그인 노드에서 실행했나?)")
    if problems:
        raise SystemExit("[PREFLIGHT FAIL]\n  - " + "\n  - ".join(problems))
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": importlib.import_module("transformers").__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def nvidia_smi_used() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


# ── 모델 로딩(전 모델 동일 스택: transformers + sdpa + device_map=auto) ──

def load_model(repo: str, loader: str, precision: str):
    import torch
    from transformers import AutoTokenizer
    kwargs: dict = {"device_map": "auto", "attn_implementation": "sdpa"}
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        raise ValueError(f"precision {precision}?")
    if loader == "causal":
        from transformers import AutoModelForCausalLM as M
    elif loader == "image_text":
        from transformers import AutoModelForImageTextToText as M
    else:
        raise ValueError(f"loader {loader}?")
    tok = AutoTokenizer.from_pretrained(repo)
    model = M.from_pretrained(repo, **kwargs)
    return tok, model


def apply_template(tok, messages):
    """gemma 계열은 system 롤을 거부할 수 있음 → 첫 user에 병합하는 일반 폴백."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged, sys_txt = [], ""
        for m in messages:
            if m["role"] == "system":
                sys_txt += m["content"] + "\n\n"
            else:
                merged.append(dict(m))
        if merged and sys_txt:
            merged[0]["content"] = sys_txt + merged[0]["content"]
        return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


# ── 1회 생성 + 즉석 검증(형식만 — 충실도 지표는 aggregate.py 몫) ─────

def generate_one(tok, model, messages, dec: dict) -> dict:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    text = apply_template(tok, messages)
    device = getattr(model, "device", "cuda:0")
    inputs = tok(text, return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        gen = model.generate(
            **inputs, max_new_tokens=dec["max_new_tokens"],
            do_sample=dec["do_sample"] and dec["temperature"] > 0,
            temperature=dec["temperature"], top_p=dec["top_p"])
    torch.cuda.synchronize()
    latency = time.time() - t0
    out_ids = gen[0, n_in:]
    raw = tok.decode(out_ids, skip_special_tokens=True)
    peak = {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
            for i in range(torch.cuda.device_count())}
    return {"latency_s": round(latency, 2), "in_tokens": int(n_in),
            "gen_tokens": int(out_ids.numel()),
            "tokens_per_s": round(out_ids.numel() / latency, 2) if latency > 0 else None,
            "peak_alloc_gib": peak, "raw": raw}


def validate_format(raw: str) -> dict:
    """③ 지시 이행 즉석 판정: 파싱/스키마/규칙. (충실도 ④는 aggregate.py)"""
    from step5_components.llm_fusion import QwenLLMFusion
    import re
    r = {"parse_ok": False, "schema_ok": False, "rule_flags": [], "n_candidates": None,
         "reasoning_origins": []}
    if QwenLLMFusion._looks_derailed(raw):
        r["rule_flags"].append("derail_chars")  # 한자/대체문자 — fp16 오버플로 탈선 신호
    try:
        draft = QwenLLMFusion._parse_draft(raw)  # 펜스 제거+json+FusionDraft(pydantic)
        r["parse_ok"] = True  # _parse_draft는 파싱+스키마 일체 — 단계 구분 위해 아래서 재파싱
        r["schema_ok"] = True
        r["n_candidates"] = len(draft.candidates)
        r["reasoning_origins"] = [c.knowledge.reasoning_origin.value for c in draft.candidates]
        narr = " ".join(f"{c.knowledge.situation} {c.knowledge.tacit_insight} "
                        f"{c.knowledge.reasoning or ''}" for c in draft.candidates)
        if len(re.findall(r"[가-힣]", narr)) < 10:
            r["rule_flags"].append("not_korean")
        if not draft.candidates:
            r["rule_flags"].append("empty_candidates")
    except json.JSONDecodeError:
        r["rule_flags"].append("json_parse_fail")
    except Exception as e:
        # json은 됐는데 스키마 실패인지 구분
        try:
            t = raw.strip().strip("`")
            json.loads(t[t.find("{"): t.rfind("}") + 1])
            r["parse_ok"] = True
            r["rule_flags"].append(f"schema_fail({type(e).__name__})")
        except Exception:
            r["rule_flags"].append("json_parse_fail")
    for key in ("diagnostic_steps", "source_utterance", '"timestamp"'):
        if key in raw:
            r["rule_flags"].append(f"forbidden_field({key.strip(chr(34))})")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default=None, help="기본: bench_config.yaml 값")
    ap.add_argument("--runs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "bench_config.yaml").read_text(encoding="utf-8"))
    precision = args.precision or cfg["precision"]
    n_runs = args.runs or cfg["runs"]
    versions = preflight(cfg, args.model, precision)
    mcfg = cfg["models"][args.model]
    dec = cfg["decoding"]

    out_root = HERE / "results" / args.model / precision
    for n in range(1, n_runs + 1):
        if (out_root / f"run{n}").exists():
            raise SystemExit(f"[ABORT] {out_root}/run{n} 이미 존재 — 덮어쓰기 금지(폴더를 지우거나 이름 바꿔라)")

    labels = json.loads((HERE / "windows_labels.json").read_text(encoding="utf-8"))
    frozen = json.loads((HERE / "frozen_windows.json").read_text(encoding="utf-8"))
    from step5_prompts.llm_fusion_prompt import build_fusion_messages

    import gc
    import torch
    print(f"[BENCH] {args.model} ({mcfg['repo_id']}) precision={precision} "
          f"runs={n_runs}+warmup — versions {versions}")
    smi_before = nvidia_smi_used()
    t0 = time.time()
    tok, model = load_model(mcfg["repo_id"], mcfg["loader"], precision)
    load_s = round(time.time() - t0, 1)
    static_alloc = {i: round(torch.cuda.memory_allocated(i) / 2**30, 2)
                    for i in range(torch.cuda.device_count())}
    smi_static = nvidia_smi_used()
    print(f"[BENCH] 로드 {load_s}s — 정적 alloc {static_alloc} GiB / nvidia-smi {smi_static} MiB "
          f"(로드 전 {smi_before})")

    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "repo_id": mcfg["repo_id"], "loader": mcfg["loader"],
            "precision": precision, "decoding": dec, "versions": versions,
            "load_s": load_s, "static_alloc_gib": static_alloc,
            "nvidia_smi_mib": {"before_load": smi_before, "after_load": smi_static},
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": cfg["prompt"], "n_windows": len(labels)}

    def one_window(rec_label):
        tid = rec_label["id"]
        fw = frozen[tid]
        messages = build_fusion_messages(fw["video_id"], [fw["payload"]])
        try:
            g = generate_one(tok, model, messages, dec)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            del e  # OOM traceback이 텐서 물지 않게 즉시 해제
            gc.collect()
            torch.cuda.empty_cache()
            return {"id": tid, "case": rec_label["case"], "error": err}
        v = validate_format(g["raw"])
        return {"id": tid, "case": rec_label["case"], **g, **v}

    # 웜업(첫 윈도우 1회, 집계 제외 — 커널 컴파일/캐시 워밍 비용 분리)
    w = one_window(labels[0])
    (out_root / "warmup.json").write_text(json.dumps(w, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"[BENCH] 웜업: {w.get('latency_s')}s peak={w.get('peak_alloc_gib')} "
          f"err={w.get('error')}")
    # ★ 정밀도 판단 재료(첫 모델 첫 실행): 여기서 peak가 한도(~22GiB/장) 근접이면 중단하고
    #   nf4 전환을 결정한다 — sbatch가 이 로그 라인을 보고 사람이/에이전트가 판단.
    if w.get("error") and "OutOfMemory" in w["error"]:
        meta["verdict_hint"] = "웜업부터 OOM — fp16 폐기, 셋 다 nf4 전환 필요"
        (out_root / "runmeta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        raise SystemExit("[ABORT] 웜업 OOM — precision 재결정 필요")

    for n in range(1, n_runs + 1):
        run_dir = out_root / f"run{n}"
        run_dir.mkdir()
        t_run = time.time()
        records = []
        for lab in labels:
            rec = one_window(lab)
            records.append(rec)
            tag = (f"{rec['latency_s']}s {rec['gen_tokens']}tok "
                   f"{'OK' if rec.get('schema_ok') else 'FAIL:' + ','.join(rec.get('rule_flags', []))}"
                   if "error" not in rec else f"ERR {rec['error'][:60]}")
            print(f"[BENCH] run{n} {rec['id']:12s} ({rec['case']}) {tag}", flush=True)
        with (run_dir / "outputs.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (run_dir / "run_config.json").write_text(json.dumps(
            {**meta, "run": n, "run_elapsed_s": round(time.time() - t_run, 1),
             "nvidia_smi_after_run_mib": nvidia_smi_used()},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[BENCH] run{n} 완료 ({round(time.time() - t_run, 1)}s)")

    (out_root / "runmeta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    # 스모크 원칙: 빈 결과로 exit 0 금지
    assert any((out_root / f"run{n}" / "outputs.jsonl").stat().st_size > 0
               for n in range(1, n_runs + 1)), "결과 0건"
    print(f"[OK] {args.model} 완료 → {out_root}")


if __name__ == "__main__":
    main()
