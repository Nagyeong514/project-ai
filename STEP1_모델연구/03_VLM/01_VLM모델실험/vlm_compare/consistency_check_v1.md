# 검증 A: confidence-uncertain_points 상호 규칙 위반 (사후 검증, 재추론 없음)

규칙: `uncertain_points`가 비어있지 않은데 `visual_confidence`가 `"high"`이면 위반.

| 모델 | 총 프레임 수 | 파싱실패 제외 | 검증 대상 | 위반 건수 | 위반율 | uncertain_points 사용률 |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 326 | 0 | 326 | 0 | 0.0% | 53/326 (16.3%) |
| InternVL3.5-8B | 326 | 0 | 326 | 0 | 0.0% | 42/326 (12.9%) |
| MiniCPM-V 4.5 | 326 | 4 | 322 | 1 | 0.3% | 3/322 (0.9%) |

**코멘트**: 세 모델의 위반율이 서로 비슷한 수준입니다.

검증 B(tools_or_parts vs YOLO 탐지결과)는 이번 실행에서 스킵. 매핑 사전은 추후 Hallucination Rate 메트릭 본작업에서 별도로 구축 예정.

## 부록: 위반 프레임 상세 목록

### Qwen3-VL-8B 위반 프레임 목록
(위반 없음)

### InternVL3.5-8B 위반 프레임 목록
(위반 없음)

### MiniCPM-V 4.5 위반 프레임 목록 (1건)
- CLIP4_reboot_bios frame_idx=81
