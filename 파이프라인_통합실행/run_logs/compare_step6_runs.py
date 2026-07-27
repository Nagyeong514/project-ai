"""STEP6 두 run의 판정 결과 전/후 비교 (2026-07-08, motion_run2 vs motion_run3용).

후보 id 기준으로 confidence/판정/점수 4항목/접지 상태를 대조해
① 판정 분포 변화 ② 후보별 confidence 변화표 ③ 접지 전이(model_inferred→utterance)
④ 과잉 접지 신호(rg 0인데 utterance 태깅 = judge가 접지 불인정)를 출력한다.

사용: python3 compare_step6_runs.py --before <run2 폴더> --after <run3 폴더>
주의: 후보 id는 재실행마다 순번이 밀릴 수 있다(윈도우→후보 병합이 비결정적).
      id 일치만으로 같은 지식이라 단정하지 말고 window_ids 도 함께 본다.
"""
import argparse
import glob
import json
import os


def load_run(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_verified.json"))):
        j = json.load(open(f))
        k = j["knowledge"]
        v = j["verification"]
        out[j["id"]] = {
            "windows": ",".join(j.get("window_ids", [])),
            "origin": k.get("reasoning_origin"),
            "n_rs": len(k.get("reasoning_source", [])),
            "scores": v["scores"],
            "conf": v["confidence"],
            "dec": v["decision"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    b, a = load_run(args.before), load_run(args.after)

    for name, run in (("before", b), ("after", a)):
        dist = {}
        for r in run.values():
            dist[r["dec"]] = dist.get(r["dec"], 0) + 1
        n_utt = sum(1 for r in run.values() if r["origin"] == "utterance")
        overg = sum(1 for r in run.values()
                    if r["origin"] == "utterance" and r["scores"]["reasoning_grounding"] == 0.0)
        print(f"[{name}] {len(run)}건 | 판정 {dist} | utterance 태깅 {n_utt}건 "
              f"| 과잉접지 의심(utterance인데 rg=0) {overg}건")

    print(f"\n{'id':42s} {'win(b→a)':14s} {'origin b→a':26s} {'rg b→a':12s} {'conf b→a':16s} 판정 b→a")
    ids = sorted(set(b) | set(a))
    for i in ids:
        rb, ra = b.get(i), a.get(i)
        f = lambda r, k: (f"{r['scores'][k]:.2f}" if r else " -- ")
        c = lambda r: (f"{r['conf']:.3f}" if r else "  --  ")
        d = lambda r: (r["dec"] if r else "--")
        o = lambda r: (f"{r['origin']}({r['n_rs']})" if r else "--")
        w = lambda r: (r["windows"] if r else "--")
        mark = ""
        if rb and ra:
            if ra["conf"] > rb["conf"] + 1e-9:
                mark = " ↑"
            elif ra["conf"] < rb["conf"] - 1e-9:
                mark = " ↓"
            if rb["dec"] != ra["dec"]:
                mark += f"  ★{rb['dec']}→{ra['dec']}"
        print(f"{i:42s} {w(rb)}→{w(ra):7s} {o(rb)}→{o(ra):14s} "
              f"{f(rb,'reasoning_grounding')}→{f(ra,'reasoning_grounding')} "
              f"{c(rb)}→{c(ra)}{mark}")


if __name__ == "__main__":
    main()
