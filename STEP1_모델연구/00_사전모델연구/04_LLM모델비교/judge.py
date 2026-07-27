"""judge.py — LLM-as-judge 자동 채점 (계획서 3.5 / 5절).

후보와 다른 계열의 심판 모델로 6항목을 1~5점 루브릭(prompts/rubric.md)으로 채점한다.
6항목: Faithfulness(충실도) / Accuracy(정확성) / Usefulness(유용성) /
       CodeSwitch(한·영 혼용 이해) / Fluency(자연스러움) / Format(형식 준수).

통제(계획서 3.5):
- reference-free 채점(정답 일치가 아니라 근거·타당성 기준) — 채점 프롬프트에 명시.
- 제시 순서 무작위화(position bias 완화): 샘플별 모델 처리 순서와 후보 제시 순서를 셔플.
- Format(형식 준수)은 심판 LLM이 채점하지 않고 **기계가 자동 산출**한다(계획서 3.3/3.5):
  parse_ok(JSON 유효) + 스키마 준수(후보 N개 일치, 4개 필수필드 구비, 자기신뢰도 0~1)를 1~5점화.
  → 심판 LLM은 나머지 5항목(충실도/정확성/유용성/한영혼용/한국어)만 채점.
- 심판 출력은 JSON 안전 파싱(실패 시 재시도 1회).

입력: results/generations/*.jsonl, config.yaml(judge, weights, generation.seed),
      data/golden_set/golden_set.jsonl(모범답안), prompts/judge_prompt.txt, prompts/rubric.md
출력: results/scores.csv
      (컬럼: sample_id, model, faithfulness, accuracy, usefulness, codeswitch, fluency, format, judge_comment)

사용:
  python judge.py              # 실제 심판 엔드포인트 호출
  python judge.py --dry-run    # 엔드포인트 없이 가짜 점수로 흐름 점검
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml

# CSV/최종 점수 6항목. format 은 기계 자동 산출(계획서 3.3/3.5), 나머지 5항목은 심판 LLM 채점.
ITEMS = ["faithfulness", "accuracy", "usefulness", "codeswitch", "fluency", "format"]
ITEMS_LLM = ["faithfulness", "accuracy", "usefulness", "codeswitch", "fluency"]


# ──────────────────────────────────────────────────────────────────────────
# 로드
# ──────────────────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_generations(gen_dir: Path, model_keys: list[str]) -> list[dict]:
    """results/generations/{model}.jsonl 들을 모아 한 리스트로. 모델별 파일이 없으면 건너뜀."""
    rows: list[dict] = []
    for key in model_keys:
        p = gen_dir / f"{key}.jsonl"
        if not p.exists():
            print(f"  [경고] 생성결과 없음, 건너뜀: {p.name}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def read_records(path: Path) -> list[dict]:
    """골든셋 파일을 레코드 리스트로 로드. JSON 배열([...])과 JSONL 모두 자동 인식."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return list(json.loads(text))
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def record_id(rec: dict, fallback) -> str:
    """id 별칭 허용: id / sample_id / 없으면 fallback (generate.py 와 동일 규칙)."""
    return str(rec.get("id") or rec.get("sample_id") or fallback)


def load_references(path: Path) -> dict[str, str]:
    """골든셋에서 sample_id → 모범답안(reference) 매핑. 없으면 빈 dict(reference-free라 무방).

    reference 별칭 허용: reference / ideal_tacit_candidates / input.reference.
    """
    refs: dict[str, str] = {}
    if not path.exists():
        return refs
    for i, rec in enumerate(read_records(path), 1):
        sid = record_id(rec, i)
        ref = (rec.get("reference")
               or rec.get("ideal_tacit_candidates")
               or rec.get("input", {}).get("reference"))
        refs[sid] = json.dumps(ref, ensure_ascii=False) if ref is not None else ""
    return refs


def load_inputs(path: Path) -> dict[str, dict]:
    """골든셋에서 sample_id → {vlm_result, transcript}. 없으면 빈 dict."""
    inputs: dict[str, dict] = {}
    if not path.exists():
        return inputs
    for i, rec in enumerate(read_records(path), 1):
        sid = record_id(rec, i)
        inp = rec.get("input", rec)
        inputs[sid] = {
            "vlm_result": inp.get("vlm_result", ""),
            "transcript": inp.get("transcript", ""),
        }
    return inputs


# ──────────────────────────────────────────────────────────────────────────
# 채점 입력 구성 / 파싱
# ──────────────────────────────────────────────────────────────────────────
def candidate_text(row: dict, rng: random.Random, randomize: bool) -> str:
    """후보 답안을 채점 프롬프트에 넣을 문자열로. parse 성공 시 후보 순서를 셔플(position bias)."""
    parsed = row.get("parsed")
    if isinstance(parsed, list):
        cands = list(parsed)
        if randomize:
            rng.shuffle(cands)
        return json.dumps(cands, ensure_ascii=False, indent=2)
    # 파싱 실패물은 원문 그대로 보여줌(형식 점수 평가 위해)
    return row.get("raw_output") or "(빈 응답)"


def build_messages(template: str, rubric: str, glossary: str, inp: dict,
                   reference: str, candidate: str) -> list[dict]:
    user = template.format(
        rubric=rubric,
        glossary=glossary,
        vlm_result=inp.get("vlm_result", ""),
        transcript=inp.get("transcript", ""),
        reference=reference or "(모범답안 미제공 — reference-free로 평가)",
        candidate=candidate,
    )
    return [{"role": "user", "content": user}]


def parse_scores(raw: str) -> dict | None:
    """심판 응답에서 6항목+comment JSON 파싱. 실패 시 None."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                obj = json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for it in ITEMS_LLM:  # format 제외 — 심판은 5항목만 채점
        v = obj.get(it)
        if not isinstance(v, (int, float)):
            return None
        out[it] = int(max(1, min(5, round(v))))  # 1~5 클램프
    out["judge_comment"] = str(obj.get("judge_comment", "")).replace("\n", " ").strip()
    return out


def compute_format_score(row: dict, required: list[str], n_expected: int) -> int:
    """형식 준수(Format)를 기계로 자동 산출 (계획서 3.3/3.5) → 1~5 정수.

    구성: parse_ok(JSON 유효) + 스키마 준수(후보 N개 일치, 4개 필수필드 구비,
          자기신뢰도 0~1 수치). 심판 LLM이 채점하지 않는다.
    """
    if not row.get("parse_ok"):
        return 1  # JSON 파싱 불가 = 형식 미준수
    parsed = row.get("parsed")
    if not isinstance(parsed, list) or not parsed:
        return 1  # 배열 아님/빈 배열

    def cand_valid(c: object) -> bool:
        if not isinstance(c, dict):
            return False
        if any(k not in c for k in required):
            return False
        conf = c.get("자기신뢰도")
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            return False
        return True

    frac_valid = sum(cand_valid(c) for c in parsed) / len(parsed)
    # "최대 N개" 정책(과잉생성 함정 대응): 1~N개면 개수 적정. 0개는 위에서 1점 처리됨.
    count_ok = (1 <= len(parsed) <= n_expected)
    # 2점(유효 JSON 배열) + 최대 2점(필수필드 충실도) + 1점(개수 적정: 1~N)
    score = 2 + 2 * frac_valid + (1 if count_ok else 0)
    return int(max(1, min(5, round(score))))


# ──────────────────────────────────────────────────────────────────────────
# 심판 호출
# ──────────────────────────────────────────────────────────────────────────
def call_judge(judge_cfg: dict, messages: list[dict], max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=judge_cfg["base_url"], api_key=judge_cfg.get("api_key", "EMPTY"))
    resp = client.chat.completions.create(
        model=judge_cfg["name"],
        messages=messages,
        temperature=0,          # 채점은 결정적으로
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def fake_scores(row: dict, rng: random.Random) -> dict:
    """--dry-run 가짜 점수: 심판 5항목만 생성(format 은 자동 산출). 모델별로 살짝 다르게."""
    bias = {"qwen": 1, "exaone": 0, "llama": 0}.get(row.get("model"), 0)
    out = {it: max(1, min(5, 3 + bias + rng.randint(-1, 1))) for it in ITEMS_LLM}
    out["judge_comment"] = f"[dry-run] {row.get('model')} 더미 채점(5항목)"
    return out


# ──────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge 채점 (계획서 3.5)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="엔드포인트 없이 가짜 점수로 흐름 점검")
    args = ap.parse_args()

    root = Path(args.config).resolve().parent
    cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    judge_cfg = cfg["judge"]
    randomize = bool(judge_cfg.get("position_randomization", True))
    seed = cfg.get("generation", {}).get("seed", 42)
    rng = random.Random(seed)

    template = read_text(root / paths["judge_prompt"])
    rubric = read_text(root / paths["rubric"])
    glossary = read_text(root / paths["glossary"])

    # 형식 자동 산출용: 스키마의 필수필드 + 기대 후보 개수 (계획서 3.3/3.5)
    schema = json.loads(read_text(root / paths["output_schema"]))
    required = schema.get("items", {}).get("required",
                                           ["암묵지내용", "근거", "자기신뢰도", "관련작업단계"])
    n_expected = int(cfg["generation"]["n_candidates"])

    model_keys = [m["key"] for m in cfg["models"]]
    rows = load_generations(root / paths["generations_dir"], model_keys)
    if not rows:
        sys.exit("[오류] 채점할 생성결과가 없습니다. 먼저 generate.py 를 실행하세요.")

    refs = load_references(root / paths["golden_set"])
    inputs = load_inputs(root / paths["golden_set"])

    # 샘플별로 묶어 모델 제시 순서 무작위화 (position bias 완화 — 계획서 3.5)
    by_sample: dict[str, list[dict]] = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], []).append(r)

    print(f"샘플 {len(by_sample)}개 × 모델출력 {len(rows)}건 채점 "
          f"(judge={judge_cfg['name']}, 순서무작위화={randomize}) "
          f"{'[DRY-RUN]' if args.dry_run else ''}")

    out_path = root / paths["scores_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id", "model"] + ITEMS + ["judge_comment"]
    n_ok = n_total = 0

    with open(out_path, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=header)
        writer.writeheader()

        for sid in by_sample:
            sample_rows = list(by_sample[sid])
            if randomize:
                rng.shuffle(sample_rows)  # 모델 제시 순서 셔플
            for row in sample_rows:
                n_total += 1
                cand = candidate_text(row, rng, randomize)
                # 행별 기대 후보 개수(생성 기록 우선, 없으면 config)
                row_n = int(row.get("n_candidates", n_expected))

                if args.dry_run:
                    scores = fake_scores(row, rng)
                    n_ok += 1
                else:
                    inp = inputs.get(sid, {})
                    ref = refs.get(sid, "")
                    messages = build_messages(template, rubric, glossary, inp, ref, cand)
                    scores = None
                    for attempt in range(2):  # 안전 파싱: 실패 시 재시도 1회
                        try:
                            raw = call_judge(judge_cfg, messages, cfg["generation"]["max_tokens"])
                        except Exception as e:
                            print(f"  [호출실패] {row['model']}/{sid} (attempt {attempt+1}): {e}")
                            continue
                        scores = parse_scores(raw)
                        if scores is not None:
                            break
                        print(f"  [파싱실패] {row['model']}/{sid} (attempt {attempt+1}) 재시도…")
                    if scores is None:  # 끝내 실패 → 5항목 중립 3점
                        scores = {it: 3 for it in ITEMS_LLM}
                        scores["judge_comment"] = "심판 파싱 실패 — 중립 3점 처리"
                    else:
                        n_ok += 1

                # format 은 심판과 무관하게 기계 자동 산출(계획서 3.3/3.5)
                scores["format"] = compute_format_score(row, required, row_n)
                writer.writerow({"sample_id": sid, "model": row["model"], **scores})

    print(f"  ✔ {out_path.relative_to(root)}  (채점성공 {n_ok}/{n_total})")


if __name__ == "__main__":
    main()
