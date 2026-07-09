# A안 준비 — Pass 2 비인접 병합 조건부 허용 (구현 초안, 미적용)

작성: 2026-07-09. **실행 보류** — STEP6 run5 결과 보고 후 적용 여부 결정(사용자 지시).
⚠️ 이 문서는 초안일 뿐 라이브 코드에 반영하지 않았다(full4가 현행 게이트로 돌아야 하므로).
적용 시 아래 old/new 블록을 Edit로 반영 → 미니 하네스 → CLIP4 스모크 재검증 순서.

## 배경
Qwen3-14B가 독립 시행 6/6에서 {W09(용량), W10(채널), W12(재확인)}를 하나의 지식으로
판단하나, 사이에 W11(Windows 로딩, 다른 목적)이 끼어 비인접 반려 게이트에 걸리고,
반려되면 인접 부분집합으로 타협하지 않고 병합을 전면 포기한다(잡 2336 재시도 3/3,
스모크 2348 재현). A안 = "낀 윈도우가 **단독 그룹으로 온전히 보존**되면 갭 1개까지
병합 허용" — 정보 손실 없이 모델의 일관된 의미 판단을 수용.

## 수정 1 — `step5_components/llm_fusion.py` `_check_grouping` 비인접 검사

### old (현행)
```python
        for g in draft.groups:
            if len(g.member_ids) < 2:
                continue
            idxs = sorted({int(w[1:]) for cid in g.member_ids for w in cand_wids[cid]})
            gaps = [(a, b) for a, b in zip(idxs, idxs[1:]) if b - a != 1]
            if gaps:
                raise ValueError(
                    f"비인접 병합 위반: 그룹 {g.member_ids}의 윈도우 "
                    f"{[f'W{i:02d}' for i in idxs]} 가 연속이 아니다(사이에 다른 후보의 "
                    f"구간이 낀다) — 시간상 이어진 후보들만 병합 가능")
```

### new (갭 1개 조건부 허용)
```python
        # 갭 허용(2026-07-09 A안): 병합 그룹 사이에 낀 윈도우가 '단독(1건) 그룹'으로
        # 온전히 보존되면 갭 총 1개까지 허용. 낀 후보가 다른 병합에 흡수돼 있으면 불허
        # (정보 보존 조건). Qwen3의 일관 판단 {W09,W10,W12}(사이 W11 단독) 수용용.
        singleton_wins = {int(w[1:]) for g2 in draft.groups if len(g2.member_ids) == 1
                          for w in cand_wids[g2.member_ids[0]]}
        for g in draft.groups:
            if len(g.member_ids) < 2:
                continue
            idxs = sorted({int(w[1:]) for cid in g.member_ids for w in cand_wids[cid]})
            gap_wins = [i for a, b in zip(idxs, idxs[1:]) for i in range(a + 1, b)]
            if gap_wins:
                uncovered = [i for i in gap_wins if i not in singleton_wins]
                if uncovered or len(gap_wins) > 1:
                    raise ValueError(
                        f"비인접 병합 위반: 그룹 {g.member_ids}의 윈도우 "
                        f"{[f'W{i:02d}' for i in idxs]} — 사이에 낀 윈도우는 1개까지, "
                        f"그것도 단독 그룹으로 보존될 때만 허용"
                        f"(낀 윈도우 {[f'W{i:02d}' for i in gap_wins]}, "
                        f"미보존 {[f'W{i:02d}' for i in uncovered] or '없음'})")
```

## 수정 2 — `step5_prompts/llm_grouping_prompt.py` 병합 판정 ① 예외 한 줄

### old
```
① 시간 연속: 멤버들의 구간이 이어지거나 가깝고, 그 사이에 다른 목적의
   작업 후보가 끼어 있지 않다.
```

### new
```
① 시간 연속: 멤버들의 구간이 이어지거나 가깝고, 그 사이에 다른 목적의
   작업 후보가 끼어 있지 않다. 예외: 화면 전환·로딩 대기처럼 잠깐 낀
   후보 1개가 자기 그룹(1건짜리)으로 온전히 남는 경우에는, 그 후보를
   사이에 두고도 같은 지식의 앞뒤 후보를 병합할 수 있다.
```

## 수정 3 — 스모크 판정기 복원(선택)
적용 후 검증 시 check_pass2_smoke.py의 (a·정보)를 다시 판정 항목으로 승격해
"{W09,W10,W12} 또는 {W09,W10} 병합 존재"로 확인하면 된다.

## 검증 계획(적용 시)
1. 오프라인: mini_grouping_harness.py Part A에 갭 케이스 추가 검토
   (현 A5 테스트 '비인접 반려'가 새 규칙에선 '낀 윈도우 미보존' 케이스로 남는지 확인 필요 —
   A5는 c01+c04 병합에 W02가 낀 케이스인데 c02/c03이 단독이면 이제 **허용**될 수 있음.
   테스트 기대값 수정 필요: 갭 2개 케이스나 낀 후보가 병합에 흡수된 케이스로 교체)
2. GPU: 미니 하네스 → CLIP4 스모크(진단 diag_grouping_clip4.py --use-retry-loop로
   {c09,c10,c12} 수렴 확인이 더 저렴) → 4클립.
