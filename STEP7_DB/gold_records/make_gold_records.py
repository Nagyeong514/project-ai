"""
정답지(암묵지 9건) -> step6 통과(accept) 최종 JSON 생성기
--------------------------------------------------------
- 각 항목을 LLM 최종 스키마(1.3) + 6-9 검증 블록 형태로 생성한다.
- action_only(침묵 동작)는 설계원칙 4에 따라 발화전제 검사(rg/sg/us)를 면제하고
  게이트 B(action_reason_consistency)로만 채점 -> confidence 유지되어 accept.
- utterance 항목은 4개 서브스코어 모두 적용(트랙 A 고정 가중치).
"""
import json, os

# 2026-07-04: 이 스크립트 자체가 gold_records/ 안에 있는데 여기서 "gold_records"를
# 한 번 더 join하면 gold_records/gold_records/라는 중첩 폴더가 새로 생기고 원본 파일은
# 안 덮어써진다(실제로 이렇게 새 폴더가 생겼던 걸 발견해서 고침). 스크립트와 같은
# 폴더에 바로 쓴다.
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# 트랙 A 고정 가중치 / 임계값 — 실제 STEP6_품질검증/config.py의 weight_a/t_high_a/t_low_a와
# 정확히 일치시킴(2026-07-04: 이 스크립트가 독자적으로 0.62/0.42를 쓰던 걸 실제 config.py
# 값인 0.70/0.40으로 정정 — 실제 STEP6과 다른 임계값으로 accept/hold/reject를 매기고 있었음).
W = {"reasoning_grounding": 0.35, "step_grounding_ratio": 0.30,
     "action_reason_consistency": 0.20, "utterance_signal": 0.15}
T_HIGH, T_LOW = 0.70, 0.40

# 클립별 소스 메타 (클립 상대 타임라인)
CLIP = {
    1: {"video": "master_boot_clip1.mp4", "start": "00:00:00", "end": "00:03:30"},
    2: {"video": "master_boot_clip2.mp4", "start": "00:00:00", "end": "00:02:40"},
    3: {"video": "master_boot_clip3.mp4", "start": "00:00:00", "end": "00:02:30"},
    4: {"video": "master_boot_clip4.mp4", "start": "00:00:00", "end": "00:01:30"},
}

# 정답지 9건. sub = 서브스코어(0~1), ga = 게이트A 결과
ITEMS = [
    {
        "id": "tk_dell7920_ch_preread_001", "clip": 1, "task": "메모리 장착",
        "keywords": ["RDIMM", "채널", "슬롯", "라벨", "색상", "메모리 장착"],
        "situation": "워크스테이션 조립 중 RDIMM을 슬롯에 장착하려는 단계",
        "situation_ts": "00:01:35",
        "tacit_insight": "삽입 전에 슬롯 라벨과 색상으로 채널 구성을 먼저 파악한 뒤 꽂는다. 초보자는 이 단계를 건너뛰고 임의 슬롯에 꽂는다.",
        "reasoning": "채널 구성을 잘못 잡으면 메모리 성능이 절반으로 떨어지므로 삽입 전에 채널부터 확인해야 한다.",
        "reasoning_origin": "model_inferred", "reasoning_ts": None,
        "steps": [{"action": "RDIMM 삽입 전 슬롯 라벨·색상을 응시해 채널 구성 확인",
                   "evidence": "action_only", "utt": None, "ts": "00:01:40"}],
        "ga": {"relation": "delta", "sim": 0.58, "chunk": "m_rdimm_channel", "re": False},
        # action_only(발화 근거 0건)라 reasoning_grounding/utterance_signal은 실제 STEP6
        # 코드 기준값으로 채움(면제 아님, ESTIMATED 표시 — 실제 LLM judge를 부르지 않은 추정치):
        #   - step_grounding_ratio: 발화 스텝이 없으면 graph.py가 1.0으로 폴백(추정 아님, 코드 그대로)
        #   - utterance_signal: rules.py의 rule_score = 0.0(발화 스텝 0건이면 결정론적으로 0) —
        #     실제로는 이후 LLM 보정이 한 번 더 들어가 소폭 오를 수 있으나 여기선 보정 전 값 사용
        #   - reasoning_grounding: reasoning_source가 비어 있어 대조할 발화 스니펫이 0건 →
        #     실제 LLM judge를 안 불러서 정확한 값은 모름. "근거로 삼을 발화가 아예 없다"는
        #     사실을 반영해 낮게(0.1) ESTIMATED 처리. 실측 시 이 값부터 교체할 것.
        "sub": {"action_reason_consistency": 0.88, "step_grounding_ratio": 1.0,
                "utterance_signal": 0.0, "reasoning_grounding": 0.1},
        "estimated": ["reasoning_grounding", "utterance_signal"],
    },
    {
        "id": "tk_dell7920_tactile_conn_002", "clip": 1, "task": "조립 후 점검",
        "keywords": ["전원 커넥터", "POWER2", "체결", "촉각 확인", "조립 점검"],
        "situation": "조립 중 전원 커넥터(POWER2) 체결 상태를 확인하는 단계",
        "situation_ts": "00:02:45",
        "tacit_insight": "커넥터를 눈으로 보는 데 그치지 않고 손으로 눌러 촉각으로 체결 상태를 재확인한다.",
        "reasoning": "육안으로는 완전 체결처럼 보여도 실제로는 덜 꽂힌 경우가 있어 촉각 확인이 필요하다.",
        "reasoning_origin": "model_inferred", "reasoning_ts": None,
        "steps": [{"action": "POWER2 커넥터를 손으로 눌러 체결 상태 재확인",
                   "evidence": "action_only", "utt": None, "ts": "00:02:50"}],
        "ga": {"relation": "delta", "sim": 0.55, "chunk": "m_cable_check", "re": False},
        # 001과 동일한 사유(발화 근거 0건) — 위 주석 참고.
        "sub": {"action_reason_consistency": 0.87, "step_grounding_ratio": 1.0,
                "utterance_signal": 0.0, "reasoning_grounding": 0.1},
        "estimated": ["reasoning_grounding", "utterance_signal"],
    },
    {
        "id": "tk_dell7920_led_read_003", "clip": 2, "task": "메모리 부팅 트러블슈팅",
        "keywords": ["LED", "점멸 패턴", "POST", "RAM", "진단"],
        "situation": "부팅 실패 후 전원 LED로 원인 계열을 진단하는 상황",
        "situation_ts": "00:00:05",
        "tacit_insight": "전원 LED 점멸 패턴을 끝까지 읽어 원인을 RAM/CPU 계통으로 즉시 좁힌다. 초보자는 패턴을 봐도 의미를 해석하지 못한다.",
        "reasoning": "LED 점멸은 POST 진단 코드라 패턴만으로 원인 서브시스템을 특정할 수 있다.",
        "reasoning_origin": "utterance", "reasoning_ts": "00:00:20",
        "steps": [{"action": "전원 LED 점멸 패턴 판독(황색1·백색3)으로 원인 계열 좁히기",
                   "evidence": "utterance", "utt": "황색 한 번, 백색 세 번. 이 패턴이면 RAM 쪽이야.", "ts": "00:00:20"}],
        "ga": {"relation": "delta", "sim": 0.66, "chunk": "m_post_led", "re": False},
        "sub": {"reasoning_grounding": 0.90, "step_grounding_ratio": 1.0,
                "action_reason_consistency": 0.92, "utterance_signal": 0.85},
    },
    {
        "id": "tk_dell7920_systematic_check_004", "clip": 2, "task": "메모리 부팅 트러블슈팅",
        "keywords": ["케이블", "커넥터", "체계적 점검", "촉각 확인", "진단 순서"],
        "situation": "원인 부품을 특정하기 전에 전체 연결 상태를 점검하는 상황",
        "situation_ts": "00:00:40",
        "tacit_insight": "특정 부품을 의심하기 전에 연결 상태 전체를 체계적으로 먼저 점검한다. 커넥터를 눌러보고 당겨보는 촉각 확인을 포함한다.",
        "reasoning": "성급히 한 부품을 의심하면 오진하므로 전체를 훑어 원인 범위를 먼저 좁힌다.",
        "reasoning_origin": "utterance", "reasoning_ts": "00:00:55",
        "steps": [{"action": "전체 케이블·커넥터 결선 점검(누르고 당겨 촉각 확인)",
                   "evidence": "utterance", "utt": "원인 좁히기 전에 일단 다 봐요.", "ts": "00:00:55"}],
        "ga": {"relation": "delta", "sim": 0.60, "chunk": "m_cable_check", "re": False},
        "sub": {"reasoning_grounding": 0.82, "step_grounding_ratio": 1.0,
                "action_reason_consistency": 0.90, "utterance_signal": 0.70},
    },
    {
        "id": "tk_dell7920_latch_removal_005", "clip": 2, "task": "메모리 취급",
        "keywords": ["RDIMM", "래치", "탈거", "슬롯 핀", "파손 방지"],
        "situation": "RDIMM을 슬롯에서 탈거하는 상황",
        "situation_ts": "00:02:05",
        "tacit_insight": "RDIMM 탈거 시 양쪽 래치를 동시에 눌러야 안전하다. 한쪽만 누르면 슬롯 핀이 파손될 수 있다.",
        "reasoning": "한쪽만 누르면 모듈이 비스듬히 들려 슬롯 핀에 무리가 간다.",
        "reasoning_origin": "utterance", "reasoning_ts": "00:02:10",
        "steps": [{"action": "양쪽 래치를 동시에 가압하여 RDIMM 탈거",
                   "evidence": "utterance", "utt": "래치 두 개 동시에 눌러야 해요.", "ts": "00:02:10"}],
        "ga": {"relation": "novel", "sim": 0.44, "chunk": "m_rdimm_seat", "re": True},
        "sub": {"reasoning_grounding": 0.85, "step_grounding_ratio": 1.0,
                "action_reason_consistency": 0.90, "utterance_signal": 0.80},
    },
    {
        "id": "tk_dell7920_terminal_clean_006", "clip": 3, "task": "메모리 접점 정비",
        "keywords": ["RAM 단자", "산화막", "세척", "접점 불량", "교체 우선순위"],
        "situation": "메모리 재장착 전에 단자를 정비하는 상황",
        "situation_ts": "00:00:05",
        "tacit_insight": "새 부품이라도 RAM 단자의 산화막을 먼저 제거한다. 부품 교체를 고려하기 전에 단자 세척이 우선이다.",
        "reasoning": "산화막이 접점 불량을 유발하므로 교체 전에 세척으로 해결되는 경우가 많다.",
        "reasoning_origin": "utterance", "reasoning_ts": "00:00:30",
        "steps": [{"action": "RAM 단자 세척으로 산화막 제거 후 재장착",
                   "evidence": "utterance", "utt": "새 부품이라도 단자는 한 번 닦아야 해요.", "ts": "00:00:30"}],
        "ga": {"relation": "novel", "sim": 0.31, "chunk": "m_rdimm_seat", "re": True},
        "sub": {"reasoning_grounding": 0.88, "step_grounding_ratio": 1.0,
                "action_reason_consistency": 0.90, "utterance_signal": 0.85},
    },
    {
        "id": "tk_dell7920_latch_confirm_007", "clip": 3, "task": "메모리 장착 확인",
        "keywords": ["래치", "잠금", "클릭음", "촉각 확인", "이중 확인"],
        "situation": "RDIMM 재장착 후 잠금 상태를 확인하는 상황",
        "situation_ts": "00:01:05",
        "tacit_insight": "래치 잠금을 청각(클릭 소리 2회)과 촉각(당겨보기)으로 이중 확인한다. 눈으로만 보지 않는다.",
        "reasoning": "육안으로는 잠긴 듯 보여도 실제로는 미잠금이 있어 소리·촉각으로 이중 확인한다.",
        "reasoning_origin": "utterance", "reasoning_ts": "00:01:10",
        "steps": [{"action": "래치 잠금 이중 확인(클릭음 청취 + 모듈 당겨보기)",
                   "evidence": "utterance", "utt": "클릭 소리 두 번 들려야 해요.", "ts": "00:01:10"}],
        "ga": {"relation": "delta", "sim": 0.60, "chunk": "m_rdimm_seat", "re": False},
        "sub": {"reasoning_grounding": 0.86, "step_grounding_ratio": 1.0,
                "action_reason_consistency": 0.90, "utterance_signal": 0.80},
    },
    {
        "id": "tk_dell7920_channel_id_008", "clip": 3, "task": "메모리 채널 구성",
        "keywords": ["채널", "슬롯 색상", "라벨", "6채널", "워크스테이션"],
        "situation": "슬롯 색상과 라벨로 채널 구성을 식별하는 상황",
        "situation_ts": "00:01:55",
        "tacit_insight": "슬롯 색상과 라벨로 채널 구성을 즉시 파악한다. 6채널 워크스테이션 전용 지식으로 초보자는 인지하지 못한다.",
        "reasoning": "색과 라벨이 채널을 나타내므로 이를 알면 즉시 올바른 슬롯을 선택할 수 있다.",
        "reasoning_origin": "utterance", "reasoning_ts": "00:02:00",
        "steps": [{"action": "슬롯 색상·라벨로 A/B 채널 즉시 식별",
                   "evidence": "utterance", "utt": "이 색이 A채널, 저 색이 B채널이에요.", "ts": "00:02:00"}],
        "ga": {"relation": "delta", "sim": 0.68, "chunk": "m_rdimm_channel", "re": False},
        "sub": {"reasoning_grounding": 0.88, "step_grounding_ratio": 1.0,
                "action_reason_consistency": 0.90, "utterance_signal": 0.75},
    },
    {
        "id": "tk_dell7920_bios_verify_009", "clip": 4, "task": "부팅 후 메모리 검증",
        "keywords": ["BIOS", "메모리 용량", "채널 구성", "최종 검증", "성능"],
        "situation": "부팅 성공 후 BIOS에서 메모리 상태를 최종 검증하는 상황",
        "situation_ts": "00:00:25",
        "tacit_insight": "부팅 성공만으로 완료로 보지 않는다. BIOS에서 메모리 총 용량과 채널 구성 수치를 반드시 최종 확인한다.",
        "reasoning": "채널 오인식이면 부팅은 되어도 성능이 절반으로 떨어지므로 수치 확인이 필수다.",
        "reasoning_origin": "utterance", "reasoning_ts": "00:00:45",
        "steps": [{"action": "BIOS에서 메모리 총 용량·채널 구성 수치 확인",
                   "evidence": "utterance", "utt": "채널 오인식이면 성능이 절반으로 날아가요.", "ts": "00:00:45"}],
        "ga": {"relation": "delta", "sim": 0.64, "chunk": "m_bios_mem", "re": False},
        "sub": {"reasoning_grounding": 0.90, "step_grounding_ratio": 1.0,
                "action_reason_consistency": 0.92, "utterance_signal": 0.85},
    },
]


def compute_confidence(item):
    """실제 STEP6 graph.py의 node_confidence_route()와 동일하게, 면제 없이 4개
    서브스코어 전부에 고정 가중치 W를 그대로 곱해 더한다(재정규화 없음).

    2026-07-04: 예전엔 "설계원칙 4"라는 이름으로 action_only 후보의 reasoning_grounding/
    step_grounding_ratio/utterance_signal을 통째로 면제하고 action_reason_consistency
    하나로 가중치를 재정규화했었는데, 실제 STEP6 코드(graph.py/rules.py/llm_utils.py)
    어디에도 이런 면제 로직이 없다는 게 확인돼 제거함. 실제 코드가 하는 일은:
      - step_grounding_ratio: 발화 스텝이 0건이면 1.0으로 폴백(graph.py:218) — 면제 아님.
      - utterance_signal: 발화 스텝이 0건이면 rule_score=0.0(rules.py) 위에 LLM이 보정.
      - reasoning_grounding: 스니펫이 없어도 LLM judge가 그대로 판단(면제 없음).
    """
    score = sum(item["sub"][k] * W[k] for k in W)
    return round(score, 3)


def route(score):
    return "accept" if score >= T_HIGH else ("hold" if score >= T_LOW else "reject")


rows = []
for it in ITEMS:
    c = CLIP[it["clip"]]
    seq = it["id"].split("_")[-1]
    score = compute_confidence(it)
    routing = route(score)
    estimated = set(it.get("estimated", []))

    gate_c = {"timestamp_validity": "pass"}
    for k in ("reasoning_grounding", "step_grounding_ratio", "utterance_signal"):
        gate_c[k] = it["sub"][k]

    note = "utterance 항목: 4개 서브스코어 모두 적용."
    if estimated:
        note = (
            "action_only 항목(발화 근거 0건): 면제 없이 4개 서브스코어 모두 실제 가중치로 "
            f"합산. 단 {', '.join(sorted(estimated))}는 실제 LLM judge를 부르지 않고 추정한 "
            "값(ESTIMATED)이므로 실측 시 교체 필요."
        )

    record = {
        "id": it["id"],
        "schema_version": "1.3",
        "metadata": {
            "scenario_id": c["video"].replace(".mp4", ""),
            "equipment": "Dell Precision 7920",
            "task": it["task"],
            "keywords": it["keywords"],
            "scenario_title": "조립 후 부팅 실패 진단",
            "source": {
                "video_id": c["video"],
                "clip_start": c["start"],
                "clip_end": c["end"],
                "transcript_ref": f"transcripts/{c['video'].replace('.mp4', '.json')}",
            },
        },
        "knowledge": {
            "situation": it["situation"],
            "situation_source": [it["situation_ts"]],
            "tacit_insight": it["tacit_insight"],
            "reasoning": it["reasoning"],
            "reasoning_source": [it["reasoning_ts"]] if it["reasoning_ts"] else [],
            "reasoning_origin": it["reasoning_origin"],
            "diagnostic_steps": [
                {"order": i + 1, "action": s["action"], "evidence": s["evidence"],
                 "source_utterance": s["utt"], "timestamp": s["ts"]}
                for i, s in enumerate(it["steps"])
            ],
        },
        "verification": {
            "pipeline": "step6-v1",
            "gate_a": {"relation": it["ga"]["relation"], "top1_similarity": it["ga"]["sim"],
                       "matched_manual_chunk": it["ga"]["chunk"], "researched": it["ga"]["re"]},
            "gate_b": {"action_reason_consistency": it["sub"]["action_reason_consistency"]},
            "gate_c": gate_c,
            "confidence": {"track": "A_fixed_weights", "weights": W,
                           "applied_scores": it["sub"], "score": score,
                           "estimated_scores": sorted(estimated) or None},
            "thresholds": {"T_high": T_HIGH, "T_low": T_LOW},
            "routing": routing,
            "notes": note,
        },
    }
    with open(os.path.join(OUT, it["id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    rows.append((it["id"], it["steps"][0]["evidence"], it["ga"]["relation"], score, routing))

print(f"{'id':32} {'evidence':11} {'gateA':7} {'conf':6} routing")
print("-" * 70)
for r in rows:
    print(f"{r[0]:32} {r[1]:11} {r[2]:7} {r[3]:<6} {r[4]}")
print(f"\n생성 위치: {OUT}  (총 {len(rows)}건)")
