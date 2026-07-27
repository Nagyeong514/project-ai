
"""CLIP4(잡 2437) STAGE5~10 추적: 게이트 실연·Pass2 입력 재구성·토큰 실측."""
import json, os, re, sys
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0,"/home/ai_user/team_a2/members/안나경/project-ai")
sys.path.insert(0,"/home/ai_user/team_a2/members/안나경/project-ai/STEP5_LLM으로_암묵지_후보생성")
from tacit_common import artifacts
from step5_components.llm_fusion import QwenLLMFusion
from step5_prompts.llm_fusion_prompt import build_fusion_messages
from step5_prompts.llm_grouping_prompt import build_grouping_messages
from step5_schema.tacit_schema import TacitKnowledgeDocument, FusionDraft

BASE="/home/ai_user/team_a2/members/안나경/project-ai"
VID="CLIP4_재부팅및BIOS.mp4"
windows=artifacts.load_aligned_windows(f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/aligned_windows_motion_obs",VID)
P=json.load(open(f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/tacit_json_motion_obs/{VID}.pass1.json"))
doc=TacitKnowledgeDocument.model_validate(P)
fus=QwenLLMFusion.__new__(QwenLLMFusion)

# ── (1) 접지 게이트 실연: 후보 003 ──
c3=doc.candidates[2]
print("== 접지 게이트(_ungrounded_utterance_claims) 후보 003 실연")
print("reasoning:",c3.knowledge.reasoning)
quotes=re.findall(r"[‘'\"]([^‘'\"]{4,60})[’'\"]", c3.knowledge.reasoning or "")
print("인용문 추출:",quotes)
wmap={fus._window_id(i):w for i,w in enumerate(windows,start=1)}
utts=[u.raw_text for wid in c3.window_ids for u in wmap[wid].utterances if not u.repeat_hallucination]
print("대조 대상 발화(윈도우 W03∪W04, dedup 전):",utts)
for q in quotes:
    qn=fus._norm_txt(q)[:12]
    hits=[u for u in utts if qn in fus._norm_txt(u)]
    print(f"  qn={qn!r} → 매칭 발화: {hits}")
bad=fus._ungrounded_utterance_claims(doc,windows)
print("전체 문서 접지 위반:",[c.window_ids for c in bad])

# 후보 002(W02): 최종본은 model_inferred라 게이트 스킵 — 만약 utterance였다면?
c2=doc.candidates[1]
print("\n== 후보 002(W02) — 최종 reasoning:",repr(c2.knowledge.reasoning),"origin:",c2.knowledge.reasoning_origin.value)
u2=[u.raw_text for u in wmap["W02"].utterances]
print("W02 발화:",u2)
rn=fus._norm_txt(c2.knowledge.reasoning or "")
q2=re.findall(r"[‘'\"]([^‘'\"]{4,60})[’'\"]", c2.knowledge.reasoning or "")
print("인용문:",q2)
grounded=False
for u in u2:
    un=fus._norm_txt(u)
    ok=[un[i:i+6] for i in range(0,max(1,len(un)-6),3) if un[i:i+6] in rn]
    print(f"  6자 공통부분 검사({u!r}): {ok}")
    if ok: grounded=True
print("→ utterance 태깅이었다면 접지 판정:",grounded)

# ── (2) 발화 커버리지(_missing_utterances) ──
input_utts=[u.raw_text for w in windows for u in w.utterances if not u.repeat_hallucination]
missing=fus._missing_utterances(doc,input_utts)
print(f"\n== 발화 커버리지: 입력 {len(input_utts)}건(중복 포함), 미반영 {len(missing)}건: {missing}")
print("허용 문턱: len(missing) >", f"{len(input_utts)}*(1-0.5) = {len(input_utts)*0.5}")

# ── (3) 귀속 게이트가 후보 003에서 검사한 값 ──
print("\n== 귀속 게이트 실연(_check_window_binding)")
seen={}
for c in doc.candidates:
    for wid in c.window_ids: seen[wid]=seen.get(wid,0)+1
print("귀속 카운트:",dict(sorted(seen.items())),"| 누락:",[f"W{i:02d}" for i in range(1,10) if f"W{i:02d}" not in seen])
print("후보 003: len(window_ids)=",len(c3.window_ids),"인접검사 i1-i0 =",int(c3.window_ids[1][1:])-int(c3.window_ids[0][1:]))
print("후보수 하한: ceil(9/2)=5 ≤ 후보 8건")
fus._check_window_binding(doc,len(windows)); print("게이트 통과(예외 없음)")

# ── (4) Pass2 입력 재구성 + 후보 c03 payload ──
pay2=fus._serialize_candidates(doc,windows)
print("\n== Pass2 payload c03:")
print(json.dumps(pay2[2],ensure_ascii=False,indent=1))
msg2=build_grouping_messages(VID,pay2)
print("Pass2 user 앞 400자:\n",msg2[1]["content"][:400])

# ── (5) 재시도 메시지 재구성(attempt2·3에 append된 원문) ──
err="ValueError: 접지 위장 의심 1건(window_ids [['W02']]): reasoning_origin=utterance인데 reasoning이 그 구간 발화와 무접점 — 발화 표현을 직접 인용해 reasoning을 다시 쓰거나, 발화에 '왜'가 없으면 reasoning_origin을 model_inferred로 정직 태깅하라"
fb={"role":"user","content":
    f"출력 검증 실패: {err[:300]}. 모든 후보의 knowledge에 "
    f"situation/tacit_insight/reasoning/reasoning_origin/conflict를 "
    f"빠짐없이 채우고, 출력 스키마(candidates[]: window_ids/metadata/"
    f"knowledge)를 정확히 지키고, 입력의 모든 window_id를 정확히 한 "
    f"후보에 귀속시켜(누락·중복 금지, 병합은 인접 2개까지만) "
    f"JSON만 다시 출력하라."}
ph={"role":"assistant","content":"(직전 시도 출력 — 검증 실패로 폐기됨)"}
pay1=fus._serialize(windows)
base=build_fusion_messages(VID,pay1)
m2=base+[ph,fb]; m3=m2+[ph,fb]
print("\n== 재시도 피드백 user 메시지 원문(%d자):\n%s" % (len(fb["content"]),fb["content"]))

from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
def ntok(msgs): return len(tok(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False))["input_ids"])
print("\n== 입력 토큰(재구성, chat template 포함):")
print("Pass1 attempt1:",ntok(base),"| attempt2:",ntok(m2),"| attempt3:",ntok(m3))
print("Pass2 attempt1:",ntok(msg2), "(system",len(tok(msg2[0]['content'])['input_ids']),"user",len(tok(msg2[1]['content'])['input_ids']),"| user",len(msg2[1]['content']),"자 / system",len(msg2[0]['content']),"자)")

# ── (6) _parse_draft 재현 실험: 코드펜스+금지 필드 출력 시 ──
raw_demo='''```json
{"candidates":[{"window_ids":["W03","W04"],
 "metadata":{"task":"BIOS 진입 및 정보 확인","keywords":["BIOS"],"scenario_title":"BIOS 확인","id":"tk_내맘대로_001","scenario_id":"멋대로"},
 "knowledge":{"situation":"예시 상황","tacit_insight":"예시 인사이트","reasoning":"예시 근거","reasoning_origin":"utterance","conflict":false,"conflict_detail":null,
  "diagnostic_steps":[{"order":1,"action":"LLM이 지어낸 스텝","evidence":"utterance","timestamp":"00:99:99"}],
  "timestamp":"00:12:34"}}]}
```'''
d=fus._parse_draft(raw_demo)
print("\n== _parse_draft 재현 실험(금지 필드 포함 원시 문자열 투입):")
print("draft.candidates[0].knowledge:",d.candidates[0].knowledge.model_dump())
print("metadata:",d.candidates[0].metadata.model_dump())

# ── (7) LLM 창작 vs 코드 필드 글자 수(최종 후보 003 기준) ──
T=json.load(open(f"{BASE}/STEP5_LLM으로_암묵지_후보생성/output/tacit_json_motion_obs/{VID}.tacit.json"))
c=T["candidates"][2]
llm_fields={"window_ids":c["window_ids"],"task":c["metadata"]["task"],"keywords":c["metadata"]["keywords"],
    "scenario_title":c["metadata"]["scenario_title"],"situation":c["knowledge"]["situation"],
    "tacit_insight":c["knowledge"]["tacit_insight"],"reasoning":c["knowledge"]["reasoning"],
    "reasoning_origin":c["knowledge"]["reasoning_origin"],"conflict":c["knowledge"]["conflict"],
    "conflict_detail":c["knowledge"]["conflict_detail"]}
llm_len=len(json.dumps(llm_fields,ensure_ascii=False))
tot_len=len(json.dumps(c,ensure_ascii=False))
print(f"\n== 후보 003 글자수: 전체 {tot_len}자 중 LLM 창작 계열 {llm_len}자 ({llm_len/tot_len:.1%}), 코드 결정 {tot_len-llm_len}자")
la=lt=0
for c in T["candidates"]:
    lf={"window_ids":c["window_ids"],"task":c["metadata"]["task"],"keywords":c["metadata"]["keywords"],
        "scenario_title":c["metadata"]["scenario_title"],"situation":c["knowledge"]["situation"],
        "tacit_insight":c["knowledge"]["tacit_insight"],"reasoning":c["knowledge"]["reasoning"],
        "reasoning_origin":c["knowledge"]["reasoning_origin"],"conflict":c["knowledge"]["conflict"],
        "conflict_detail":c["knowledge"]["conflict_detail"]}
    la+=len(json.dumps(lf,ensure_ascii=False)); lt+=len(json.dumps(c,ensure_ascii=False))
print(f"== 8건 전체: {lt}자 중 LLM 창작 계열 {la}자 ({la/lt:.1%})")
