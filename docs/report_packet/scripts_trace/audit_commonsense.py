
"""CLIP1~4 모션런 최종 후보 24건 전수: 서술 필드 vs 입력(발화+관찰) 문자열 대조.

방법: 후보 서술 필드(situation/tacit_insight/reasoning/task/keywords/scenario_title)의
어절을 조사·어미를 대충 벗겨 2~4자 어간으로 만들고, (a) 그 후보 멤버 윈도우의
발화+관찰 코퍼스, (b) 클립 전체 코퍼스(전 발화+전 관찰)에서 부분문자열 검색.
어간이 (a)에도 (b)에도 없으면 'LLM 외부 유입' 후보로 플래그.
"""
import json, re, sys
sys.path.insert(0,"/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0,"/home/ai_user/team_a2/members/안나경/project-ai/STEP5_LLM으로_암묵지_후보생성")
from tacit_common import artifacts

BASE="/home/ai_user/team_a2/members/안나경/project-ai"
CLIPS=["CLIP1_정상조립과정.mp4","CLIP2.mp4","CLIP3_재부팅시도 전까지.mp4","CLIP4_재부팅및BIOS.mp4"]

def norm(s): return re.sub(r"[\s\W]+","",s or "")

# 조사/어미 크루드 스트리퍼: 뒤에서 1~3자씩 깎아가며 2자 이상 어간 후보 생성
def stems(word):
    w=re.sub(r"[^\w가-힣]","",word)
    out=set()
    for cut in range(0,3):
        if len(w)-cut>=2: out.add(w[:len(w)-cut])
    return out

def tokens(text):
    return [w for w in re.split(r"[\s,./·—\-()\[\]'\"‘’]+",text or "") if len(re.sub(r"[^\w가-힣]","",w))>=2]

report={}
for vid in CLIPS:
    T=json.load(open(f"{BASE}/STEP3_전처리/transcripts/{vid}.json"))
    O=json.load(open(f"{BASE}/STEP4_YOLO_VLM관찰/output_motion/vlm_observations_original_ts/{vid}.observations.json"))
    ws=artifacts.load_aligned_windows(f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/aligned_windows_motion_obs",vid)
    D=json.load(open(f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/tacit_json_motion_obs/{vid}.tacit.json"))
    clip_corpus=norm(" ".join([u["raw_text"]+u["normalized_text"] for u in T["utterances"]]
        +[o["action"]+(o.get("actor") or "")+" ".join(o["objects"]) for o in O["observations"]]))
    wmap={f"W{i:02d}":w for i,w in enumerate(ws,1)}
    for c in D["candidates"]:
        win_corpus=norm(" ".join(
            [u.raw_text+u.normalized_text for wid in c["window_ids"] for u in wmap[wid].utterances]
            +[a.action+(a.actor or "")+" ".join(a.objects) for wid in c["window_ids"] for a in wmap[wid].actions]))
        fields={"situation":c["knowledge"]["situation"],"tacit_insight":c["knowledge"]["tacit_insight"],
                "reasoning":c["knowledge"]["reasoning"] or "","task":c["metadata"]["task"] or "",
                "keywords":" ".join(c["metadata"]["keywords"]),"scenario_title":c["metadata"]["scenario_title"] or ""}
        flags=[]
        for fname,ftext in fields.items():
            for w in tokens(ftext):
                st=stems(w)
                in_win=any(s in win_corpus for s in st)
                in_clip=any(s in clip_corpus for s in st)
                if not in_win:
                    flags.append((fname,w,"클립내O" if in_clip else "클립내X"))
        # dedup by word
        seen=set(); uniq=[]
        for f in flags:
            if f[1] in seen: continue
            seen.add(f[1]); uniq.append(f)
        report[c["id"]]={"window_ids":c["window_ids"],"origin":c["knowledge"]["reasoning_origin"],
                         "flags":uniq,"fields":fields,
                         "win_utts":[u.raw_text for wid in c["window_ids"] for u in wmap[wid].utterances],
                         "win_acts":[a.action for wid in c["window_ids"] for a in wmap[wid].actions]}

for cid,r in report.items():
    print(f"\n### {cid} {r['window_ids']} origin={r['origin']}")
    print("  윈도우 발화:", r["win_utts"])
    x=[f for f in r["flags"] if f[2]=="클립내X"]
    o=[f for f in r["flags"] if f[2]=="클립내O"]
    if x: print("  [입력 어디에도 없음]:", [(f[0],f[1]) for f in x])
    if o: print("  [윈도우엔 없고 클립 다른곳엔 있음]:", [(f[0],f[1]) for f in o])
