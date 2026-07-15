"""
융합 LLM 선정 — 정량 지표 ①~④ 자동 집계 (2026-07-09).

입력: results/{model}/{precision}/run{1..3}/outputs.jsonl (+ runmeta.json)
출력: summary.md(종합표+케이스 분해+환각 플래그 목록), metrics.json

지표 정의(지시서 그대로, 현행 프롬프트에 맞춘 각주 2건):
 ① Latency: 윈도우당 생성 시간, 웜업 제외 전체×3회 평균±σ. 보조 tokens/s.
 ② VRAM: 정적(로드 직후)/생성 peak(torch max_memory_allocated), nvidia-smi 교차.
 ③ 지시 이행률 = (파싱O & 스키마O & 규칙플래그 0) / (윈도우×3회).
    ★각주1: 현행 프롬프트(1.4 재설계)는 timestamp/diagnostic_steps '출력 금지'가 규칙 —
    "timestamp 인용 필드 채움" 대신 '금지 필드 미출력'을 규칙 준수로 센다.
 ④ 사실적 충실도(휴리스틱) = 환각 플래그 없는 출력 비율.
    - 시각 접지: 출력에 시각 표기가 '있으면' 윈도우 범위(±5s) 실존 검사(없으면 통과 — 각주1).
    - 키워드 접지: kiwipiepy 명사 겹침. 출력 명사 중 입력(STT+VLM) 등장 비율 <50% → 창작 의심.
    - 케이스 위반: B(행동만)인데 utterance 태깅/발화 인용 → 환각. C(발화만)인데 행동 서술은
      ★각주2: 신뢰 가능한 자동 판별이 없어 키워드 접지에 위임(근사) — 정성 재확인 목록에 포함.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["qwen3vl", "llama31", "gemma3"]

_kiwi = None


def nouns(text: str) -> set:
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    out = {t.form for t in _kiwi.tokenize(text or "") if t.tag.startswith("NN") and len(t.form) >= 2}
    out |= {m.lower() for m in re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", text or "")}
    return out


TS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})|(\d{1,2}):(\d{2})(?!\d)|(\d+(?:\.\d+)?)\s*초")


def ts_in_text(text: str) -> list:
    vals = []
    for m in TS_RE.finditer(text or ""):
        if m.group(1) is not None:
            vals.append(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
        elif m.group(4) is not None:
            vals.append(int(m.group(4)) * 60 + int(m.group(5)))
        else:
            vals.append(float(m.group(6)))
    return vals


def narrative_of(raw: str) -> str:
    """출력 JSON에서 서술 필드만 모은다(실패 시 raw 전체 — 보수적)."""
    try:
        t = raw.strip().strip("`")
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
        parts = []
        for c in obj.get("candidates", []):
            k = c.get("knowledge", {})
            parts += [k.get("situation", ""), k.get("tacit_insight", ""), k.get("reasoning") or ""]
            m = c.get("metadata", {})
            parts += m.get("keywords", []) + [m.get("scenario_title") or ""]
        return " ".join(p for p in parts if p)
    except Exception:
        return raw or ""


def input_texts_of(payload: dict) -> str:
    parts = []
    for a in payload.get("actions", []):
        parts.append(a.get("action", ""))
        parts += [f"{a.get('actor') or ''}"] + list(a.get("objects_visible", []))
    for u in payload.get("utterances", []):
        parts.append(u.get("raw_text", ""))
    return " ".join(parts)


def faithfulness_flags(rec: dict, label: dict, payload: dict) -> list:
    flags = []
    raw = rec.get("raw", "")
    narr = narrative_of(raw)
    # (1) 시각 접지 — 표기가 있을 때만 검사
    lo, hi = label["window_start"] - 5, label["window_end"] + 5
    bad_ts = [v for v in ts_in_text(narr) if not (lo <= v <= hi)]
    if bad_ts:
        flags.append(f"ts_hallucination({bad_ts[:3]})")
    # (2) 키워드 접지
    out_n, in_n = nouns(narr), nouns(input_texts_of(payload))
    if out_n:
        ratio = len(out_n & in_n) / len(out_n)
        rec["keyword_grounding"] = round(ratio, 3)
        if ratio < 0.5:
            flags.append(f"keyword_grounding<0.5({ratio:.2f})")
    # (3) 케이스 위반
    if label["case"] == "B":
        if "utterance" in rec.get("reasoning_origins", []):
            flags.append("case_B_utterance_origin")
        if "(발화)" in narr or "라고 말" in narr:
            flags.append("case_B_speech_citation")
    return flags


def main() -> None:
    labels = {r["id"]: r for r in json.loads((HERE / "windows_labels.json").read_text(encoding="utf-8"))}
    frozen = json.loads((HERE / "frozen_windows.json").read_text(encoding="utf-8"))
    precision = sys.argv[1] if len(sys.argv) > 1 else \
        __import__("yaml").safe_load((HERE / "bench_config.yaml").read_text(encoding="utf-8"))["precision"]

    metrics, flag_lists = {}, {}
    for model in MODELS:
        root = HERE / "results" / model / precision
        runs = sorted(root.glob("run*/outputs.jsonl"))
        if not runs:
            print(f"[agg] {model}: 결과 없음 — 건너뜀")
            continue
        meta = json.loads((root / "runmeta.json").read_text(encoding="utf-8"))
        recs = []
        for rp in runs:
            for line in rp.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                r["run"] = rp.parent.name
                recs.append(r)
        ok_recs = [r for r in recs if "error" not in r]
        lat = [r["latency_s"] for r in ok_recs]
        tps = [r["tokens_per_s"] for r in ok_recs if r.get("tokens_per_s")]
        peak = max((max(r["peak_alloc_gib"].values()) for r in ok_recs), default=None)

        n_total = len(recs)
        compliant, faithful = 0, 0
        case_stats = {c: {"n": 0, "compliant": 0, "faithful": 0} for c in "ABC"}
        flags_all = []
        for r in recs:
            lab = labels[r["id"]]
            cs = case_stats[lab["case"]]
            cs["n"] += 1
            is_comp = ("error" not in r) and r.get("parse_ok") and r.get("schema_ok") \
                and not r.get("rule_flags")
            compliant += is_comp
            cs["compliant"] += is_comp
            if "error" in r:
                flags_all.append({"id": r["id"], "run": r["run"], "case": lab["case"],
                                  "flags": [f"exec_error({r['error'][:60]})"]})
                continue
            fl = faithfulness_flags(r, lab, frozen[r["id"]]["payload"])
            if fl:
                flags_all.append({"id": r["id"], "run": r["run"], "case": lab["case"], "flags": fl})
            else:
                faithful += 1
                cs["faithful"] += 1
        by_type = {}
        for f in flags_all:
            for x in f["flags"]:
                key = x.split("(")[0]
                by_type[key] = by_type.get(key, 0) + 1
        metrics[model] = {
            "repo_id": meta["repo_id"], "precision": precision, "versions": meta["versions"],
            "load_s": meta["load_s"], "static_alloc_gib": meta["static_alloc_gib"],
            "nvidia_smi_mib": meta["nvidia_smi_mib"],
            "n_records": n_total, "n_exec_errors": n_total - len(ok_recs),
            "latency_mean_s": round(st.mean(lat), 2) if lat else None,
            "latency_sd_s": round(st.stdev(lat), 2) if len(lat) > 1 else None,
            "tokens_per_s_mean": round(st.mean(tps), 1) if tps else None,
            "peak_alloc_gib": peak,
            "compliance_pct": round(compliant / n_total * 100, 1) if n_total else None,
            "faithful_pct": round(faithful / n_total * 100, 1) if n_total else None,
            "flag_counts_by_type": by_type,
            "case_breakdown": {c: {
                "n": s["n"],
                "compliance_pct": round(s["compliant"] / s["n"] * 100, 1) if s["n"] else None,
                "faithful_pct": round(s["faithful"] / s["n"] * 100, 1) if s["n"] else None,
            } for c, s in case_stats.items()},
        }
        flag_lists[model] = flags_all

    (HERE / "metrics.json").write_text(json.dumps(
        {"precision": precision, "models": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── summary.md ──
    L = ["# 융합 LLM 선정 — 정량 집계 (자동)", "",
         f"- precision: **{precision}** (전 모델 동일) / 윈도우 15개(A6/B5/C4) × 본 실행 3회",
         "- ★ 통제실험이 아니라 **실사용 비교**: 프롬프트=STEP5 현행 동결본, 입력=실전 산출물.",
         "- 최종 판정 없음 — 정성(블라인드) 채점 합산 전까지 보류.", ""]
    if metrics:
        v = next(iter(metrics.values()))["versions"]
        L += [f"- 환경: transformers {v['transformers']} / torch {v['torch']} / CUDA {v['cuda']} / {', '.join(set(v['gpus']))}", ""]
    L += ["| 모델 | latency(s) ±σ | tok/s | VRAM 정적(GiB) | VRAM peak | 이행률 | 충실도 | 실행오류 |",
          "|---|---|---|---|---|---|---|---|"]
    for m, x in metrics.items():
        L.append(f"| {m} ({x['repo_id'].split('/')[-1]}) | {x['latency_mean_s']}±{x['latency_sd_s']} "
                 f"| {x['tokens_per_s_mean']} | {x['static_alloc_gib']} | {x['peak_alloc_gib']} "
                 f"| {x['compliance_pct']}% | {x['faithful_pct']}% | {x['n_exec_errors']} |")
    L += ["", "## 케이스별 분해 (이행률% / 충실도%)", "",
          "| 모델 | A(융합) | B(행동만·침묵) | C(발화만) |", "|---|---|---|---|"]
    for m, x in metrics.items():
        cb = x["case_breakdown"]
        L.append("| " + m + " | " + " | ".join(
            f"{cb[c]['compliance_pct']} / {cb[c]['faithful_pct']}" for c in "ABC") + " |")
    L += ["", "## 환각/위반 플래그 유형별 건수", ""]
    for m, x in metrics.items():
        L.append(f"- **{m}**: {x['flag_counts_by_type'] or '없음'}")
    L += ["", "## 플래그 상세(정성 재확인 대상)", ""]
    for m, fl in flag_lists.items():
        L.append(f"### {m}")
        for f in fl:
            L.append(f"- {f['run']} {f['id']} ({f['case']}): {', '.join(f['flags'])}")
        if not fl:
            L.append("- 없음")
        L.append("")
    (HERE / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] metrics.json / summary.md 저장 — 모델 {list(metrics)}")


if __name__ == "__main__":
    main()
