"""generate.py — 암묵지 후보 생성 (계획서 3.1·4·5절).

골든셋(입력: VLM 결과 + 정제 Transcript)을 후보 모델 3종(Qwen2.5-72B / EXAONE-3.5-32B /
Llama-3.3-70B)에 **완전히 동일한 조건**으로 입력해 구조화(JSON) 암묵지 후보를 생성·저장한다.

동일 조건(계획서 4 통제): System/User 프롬프트, Glossary 주입, Output Schema,
temperature·seed·max_tokens·n_candidates, 4bit 양자화(2.1절)를 전 모델 고정.
→ 모델별로 들어간 입력이 동일함을 검증할 수 있도록 **프롬프트 해시**를 로그에 남긴다.

서빙: vLLM OpenAI 호환 API (openai 파이썬 클라이언트로 base_url 지정 호출).
입력: config.yaml(models, generation, paths), data/golden_set/golden_set.jsonl, prompts/
출력: results/generations/{model_key}.jsonl
      (각 줄: sample_id, model, raw_output, parsed, parse_ok, prompt_hash, …)

사용:
  python generate.py                # 실제 vLLM 엔드포인트 호출
  python generate.py --dry-run      # 엔드포인트 없이 가짜 응답으로 흐름 점검
  python generate.py --config config.yaml --models qwen,llama
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


# ──────────────────────────────────────────────────────────────────────────
# 로드 헬퍼
# ──────────────────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_records(path: Path) -> list[dict]:
    """골든셋 파일을 레코드 리스트로 로드. JSON 배열([...])과 JSONL 모두 자동 인식."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":  # JSON 배열
        data = json.loads(text)
        return list(data)
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]  # JSONL


def record_id(rec: dict, fallback) -> str:
    """id 별칭 허용: id / sample_id / 없으면 fallback."""
    return str(rec.get("id") or rec.get("sample_id") or fallback)


def load_golden_set(path: Path, dry_run: bool) -> list[dict]:
    """골든셋 로드. 각 레코드에서 id·vlm_result·transcript 를 꺼낸다.

    형식 허용: JSON 배열 또는 JSONL. 입력은 input.{...} 중첩이든 top-level이든 OK.
    파일이 없고 --dry-run이면 흐름 점검용 합성 샘플 2개로 대체(실행 중단 방지).
    """
    if not path.exists():
        if dry_run:
            print(f"[dry-run] 골든셋 파일 없음({path}) → 합성 샘플 2개로 대체")
            return _synthetic_samples()
        sys.exit(
            f"[오류] 골든셋이 없습니다: {path}\n"
            f"       config.paths.golden_set 의 파일을 채운 뒤 실행하거나 --dry-run 으로 점검하세요."
        )

    samples: list[dict] = []
    for i, rec in enumerate(read_records(path), 1):
        inp = rec.get("input", rec)  # input 래핑이 없으면 레코드 자체에서 읽음
        samples.append(
            {
                "id": record_id(rec, i),
                "vlm_result": inp.get("vlm_result", ""),
                "transcript": inp.get("transcript", ""),
            }
        )
    if not samples:
        sys.exit(f"[오류] 골든셋이 비어 있습니다: {path}")
    return samples


def _synthetic_samples() -> list[dict]:
    return [
        {
            "id": "synthetic-001",
            "vlm_result": "작업자가 밸브에 잠금장치를 부착하고 압력계를 가리킨다.",
            "transcript": "여기 valve를 lockout 하고 pressure 확인부터 해야 돼. 잔압 남으면 위험해.",
        },
        {
            "id": "synthetic-002",
            "vlm_result": "롤러 표면을 손으로 만지며 온도를 살핀다.",
            "transcript": "roller가 좀 hot하다 싶으면 바로 멈추고 cooling 돌려. 감으로 알아.",
        },
    ]


# ──────────────────────────────────────────────────────────────────────────
# 프롬프트 구성 (전 모델 동일 — 계획서 4)
# ──────────────────────────────────────────────────────────────────────────
def build_messages(system_prompt: str, user_template: str, glossary: str,
                   sample: dict, n_candidates: int) -> list[dict]:
    user = user_template.format(
        glossary=glossary,
        vlm_result=sample["vlm_result"],
        transcript=sample["transcript"],
        n_candidates=n_candidates,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def prompt_hash(messages: list[dict], gen: dict) -> str:
    """샘플별 입력(프롬프트 + 생성 파라미터)의 해시.

    모델 key는 일부러 제외 → 같은 샘플이면 세 모델의 해시가 동일해야 한다(동일조건 검증).
    """
    payload = {
        "messages": messages,
        "temperature": gen["temperature"],
        "seed": gen["seed"],
        "max_tokens": gen["max_tokens"],
        "n_candidates": gen["n_candidates"],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────
# 출력 파싱 (형식 준수 지표 — parse_ok)
# ──────────────────────────────────────────────────────────────────────────
def parse_output(raw: str):
    """모델 원응답에서 JSON 배열을 파싱. 실패해도 예외를 던지지 않는다.

    반환: (parsed | None, parse_ok). 코드펜스/머리말이 섞여도 첫 '['~마지막 ']'를 시도.
    """
    if raw is None:
        return None, False
    text = raw.strip()
    # ```json ... ``` 코드펜스 제거
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, TypeError):
        pass
    # 첫 배열 구간만 추출 재시도
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), True
        except json.JSONDecodeError:
            pass
    return None, False


# ──────────────────────────────────────────────────────────────────────────
# 모델 호출
# ──────────────────────────────────────────────────────────────────────────
def call_model(model_cfg: dict, messages: list[dict], gen: dict) -> str:
    """vLLM OpenAI 호환 엔드포인트 호출 → 원응답 문자열 반환."""
    from openai import OpenAI

    client = OpenAI(base_url=model_cfg["base_url"], api_key=model_cfg.get("api_key", "EMPTY"))
    resp = client.chat.completions.create(
        model=model_cfg["name"],
        messages=messages,
        temperature=gen["temperature"],
        seed=gen["seed"],
        max_tokens=gen["max_tokens"],
    )
    return resp.choices[0].message.content


def fake_response(n_candidates: int, sample: dict, model_key: str) -> str:
    """--dry-run 가짜 응답: 스키마에 맞는 JSON 배열 문자열."""
    cands = [
        {
            "암묵지내용": f"[{model_key}] 더미 암묵지 후보 {i + 1} (샘플 {sample['id']})",
            "근거": f"Transcript: '{sample['transcript'][:24]}…'",
            "자기신뢰도": round(0.5 + 0.1 * i, 2),
            "관련작업단계": "정비 전 에너지 차단(LOTO)",
        }
        for i in range(n_candidates)
    ]
    return json.dumps(cands, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="암묵지 후보 생성 (계획서 3.1·4·5)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="엔드포인트 없이 가짜 응답으로 흐름 점검")
    ap.add_argument("--models", default="", help="쉼표구분 model key 부분집합 (기본: 전체)")
    args = ap.parse_args()

    root = Path(args.config).resolve().parent
    cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    gen = cfg["generation"]

    system_prompt = read_text(root / paths["system_prompt"])
    user_template = read_text(root / paths["user_prompt_template"])
    glossary = read_text(root / paths["glossary"])
    schema = json.loads(read_text(root / paths["output_schema"]))  # 로드해 존재/유효 확인

    samples = load_golden_set(root / paths["golden_set"], args.dry_run)

    models = cfg["models"]
    if args.models:
        wanted = {k.strip() for k in args.models.split(",") if k.strip()}
        models = [m for m in models if m["key"] in wanted]
        if not models:
            sys.exit(f"[오류] --models 와 일치하는 모델 없음: {wanted}")

    out_dir = root / paths["generations_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    n = gen["n_candidates"]
    print(f"샘플 {len(samples)}개 × 모델 {len(models)}종 × 후보 {n}개 "
          f"(temp={gen['temperature']}, seed={gen['seed']}) "
          f"{'[DRY-RUN]' if args.dry_run else ''}")

    # 동일조건 검증용: 샘플별 프롬프트 해시 (모델 무관, 같아야 정상)
    sample_hashes: dict[str, str] = {}

    for m in models:
        out_path = out_dir / f"{m['key']}.jsonl"
        ok = total = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for s in samples:
                messages = build_messages(system_prompt, user_template, glossary, s, n)
                phash = prompt_hash(messages, gen)
                sample_hashes.setdefault(s["id"], phash)

                if args.dry_run:
                    raw = fake_response(n, s, m["key"])
                else:
                    try:
                        raw = call_model(m, messages, gen)
                    except Exception as e:  # 호출 실패도 기록 후 계속
                        raw = None
                        print(f"  [호출실패] {m['key']} / {s['id']}: {e}")

                parsed, parse_ok = parse_output(raw)
                rec = {
                    "sample_id": s["id"],
                    "model": m["key"],
                    "model_name": m["name"],
                    "prompt_hash": phash,
                    "n_candidates": n,
                    "raw_output": raw,
                    "parsed": parsed,
                    "parse_ok": parse_ok,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                ok += int(parse_ok)
        print(f"  ✔ {m['key']:7s} → {out_path.relative_to(root)}  "
              f"(parse_ok {ok}/{total})")

    # 동일조건 로그: 모델 간 입력이 동일했는지(샘플별 해시가 1종이어야 함)
    print("\n[동일조건 검증] 샘플별 프롬프트 해시 (모델 무관, 동일해야 정상):")
    for sid, h in sample_hashes.items():
        print(f"  - {sid}: {h}")
    print(f"  · 적용 system_prompt 길이={len(system_prompt)}, glossary 길이={len(glossary)}, "
          f"schema items={'array' if schema.get('type') == 'array' else '?'}")


if __name__ == "__main__":
    main()
