"""
정성 평가용 블라인드 파일 생성 (2026-07-09).

세 모델의 run1 출력을 모델명 제거 후 무작위 A/B/C로 익명화:
  - blind_eval.md      : 윈도우별 3모델 출력 나란히 — 사람이 Claude에게 채점 루브릭
                         (행동-발화 매칭/두괄식/어조/존대, 1~5점)으로 넘길 파일.
  - mapping_secret.json: 익명 문자 ↔ 모델 매핑. blind_eval.md에는 절대 노출 금지.
★ 채점은 이 스크립트/에이전트가 하지 않는다 — 파일 생성까지가 역할.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["qwen3vl", "llama31", "gemma3"]


def narrative_block(raw: str) -> str:
    try:
        t = raw.strip().strip("`")
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
        lines = []
        for c in obj.get("candidates", []):
            k = c.get("knowledge", {})
            lines += [
                f"- situation: {k.get('situation')}",
                f"- tacit_insight: {k.get('tacit_insight')}",
                f"- reasoning: {k.get('reasoning')}",
                f"- reasoning_origin: {k.get('reasoning_origin')}, conflict: {k.get('conflict')}",
            ]
        return "\n".join(lines) if lines else "(후보 0건)"
    except Exception:
        return f"(JSON 파싱 실패 — 원문 앞부분)\n```\n{(raw or '')[:400]}\n```"


def main() -> None:
    precision = sys.argv[1] if len(sys.argv) > 1 else \
        __import__("yaml").safe_load((HERE / "bench_config.yaml").read_text(encoding="utf-8"))["precision"]
    labels = json.loads((HERE / "windows_labels.json").read_text(encoding="utf-8"))
    frozen = json.loads((HERE / "frozen_windows.json").read_text(encoding="utf-8"))

    outputs = {}
    for m in MODELS:
        p = HERE / "results" / m / precision / "run1" / "outputs.jsonl"
        if not p.exists():
            raise SystemExit(f"[ABORT] {p} 없음 — 세 모델 run1이 다 있어야 블라인드 생성")
        outputs[m] = {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()}

    letters = ["A", "B", "C"]
    rng = random.SystemRandom()
    shuffled = MODELS[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(letters, shuffled))  # 파일 전체 고정 매핑(윈도우 간 일관 채점)
    (HERE / "mapping_secret.json").write_text(
        json.dumps({"precision": precision, "mapping": mapping}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    L = ["# 융합 LLM 블라인드 정성 채점지 (모델 익명화: 출력 A/B/C)", "",
         "채점 루브릭(각 1~5점): ① 행동-발화 매칭(크로스모달 융합 품질) ② 두괄식(핵심 먼저)",
         "③ 어조·존대 일관성 ④ 서술 자연스러움. 윈도우마다 A/B/C 각각 채점.",
         "입력에 없는 내용을 서술하면 감점이 아니라 **환각으로 별도 표기**할 것.",
         "(매핑은 mapping_secret.json — 채점 완료 전 열람 금지)", ""]
    for lab in labels:
        tid = lab["id"]
        pl = frozen[tid]["payload"]
        L += [f"## {tid} — 케이스 {lab['case']} "
              f"({'행동+발화' if lab['case'] == 'A' else '행동만(침묵)' if lab['case'] == 'B' else '발화만'})", "",
              "**입력**"]
        for a in pl.get("actions", []):
            L.append(f"- 행동 @{a['timestamp']}: {a['action']}")
        for u in pl.get("utterances", []):
            L.append(f"- 발화 @{u['timestamp']}: “{u['raw_text']}”")
        L.append("")
        for letter in letters:
            rec = outputs[mapping[letter]].get(tid, {})
            L += [f"**출력 {letter}**", ""]
            if "error" in rec:
                L.append(f"(실행 오류: {rec['error'][:80]})")
            else:
                L.append(narrative_block(rec.get("raw", "")))
            L.append("")
        L.append("---\n")
    (HERE / "blind_eval.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] blind_eval.md ({len(labels)}윈도우 × 3출력) / mapping_secret.json(별도 보관) 저장")


if __name__ == "__main__":
    main()
