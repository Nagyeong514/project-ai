"""
Behavioral Highlights - 명장 행동 특징 추출기
----------------------------------------------
파이프라인 위치: Feature Extraction -> [Behavioral Highlights] -> Knowledge Synthesis

AI 없이 features.json에서 규칙 기반으로 주목할 구간을 추출한다.
결과가 일관되고 빠르며, 암묵지 판정(SUCCESS/MISMATCH)과 독립적으로 동작한다.

[판정 규칙 - 우선순위 순]
  measurement_speech  : 발화에 수치+단위 포함 (숫자/영단어 모두 인식)
  procedural_speech   : 발화에 절차/주의 키워드 포함
  tool_active_speech  : 도메인 관련 객체 탐지 중 발화 존재
  active_hands        : MediaPipe 손 동작 감지 (비음성 구간)

[우선순위]
  하나의 음성 구간은 가장 높은 우선순위 규칙 하나로만 분류된다.
  measurement > procedural > tool_active
"""

import re
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BehavioralHighlight:
    timestamp_sec:       float
    highlight_type:      str
    trigger_reason:      str
    feature_description: str           = ""
    speech_quote:        Optional[str]  = None
    detected_tools:      List[str]      = field(default_factory=list)
    hand_active:         bool           = False


class BehavioralHighlighter:
    # 수치 단위 패턴: 아라비아 숫자 또는 영어 숫자 단어 + 단위
    _NUM = (
        r'(?:\b\d+\.?\d*'
        r'|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|'
        r'quarter|half|hundred))'
    )
    _UNIT = (
        r'(?:ft[-\s]?lb|foot[-\s]?pound|n[-\s]?m|newton[-\s]?meter|'
        r'mm|cm|inch(?:es)?|in\b|rpm|degree|°[cCfF]|bar|psi|'
        r'kg|kilogram|pound(?:s)?|lb(?:s)?\b|torque|kpa|mpa|cc|liter)'
    )
    MEASUREMENT_RE = re.compile(_NUM + r'\s*' + _UNIT, re.IGNORECASE)

    # 절차/주의 키워드 (영어 + 한국어)
    PROCEDURAL_KEYWORDS = [
        'first', 'then', 'next', 'after that', 'careful', 'always', 'never',
        'make sure', 'important', 'trick', 'key', 'remember', 'critical',
        'must', 'ensure', 'verify', 'note that', 'be sure', 'check',
        '먼저', '다음', '그다음', '주의', '항상', '절대', '중요', '반드시',
        '확인', '포인트', '비결', '핵심', '조심', '꼭', '우선',
    ]

    # 자동차 정비 도메인에서 의미 있는 COCO 클래스만 허용
    DOMAIN_RELEVANT = {
        'person', 'car', 'truck', 'motorcycle', 'bicycle', 'bus',
        'scissors', 'knife', 'bottle', 'cell phone', 'laptop', 'clock',
    }

    def run(self, features: dict) -> List[BehavioralHighlight]:
        """features dict에서 BehavioralHighlight 목록을 추출한다."""
        highlights: List[BehavioralHighlight] = []
        audio_segs  = features.get('audio_transcript', [])
        frame_feats = features.get('frame_features', [])

        for seg in audio_segs:
            text  = seg.get('text', '').strip()
            start = float(seg.get('start_sec', 0))
            end   = float(seg.get('end_sec', 0))
            mid   = round((start + end) / 2, 3)

            if not text:
                continue

            overlap = [f for f in frame_feats if start <= f.get('timestamp_sec', -1) <= end]

            # 도메인 관련 레이블만 남김 (노이즈 제거)
            raw_tools = list({t['label'] for f in overlap for t in f.get('detected_tools', [])})
            domain_tools = [t for t in raw_tools if t in self.DOMAIN_RELEVANT]

            hand_active = any(
                f.get('hand_movement_vector', {}).get('left_hand') or
                f.get('hand_movement_vector', {}).get('right_hand')
                for f in overlap
            )

            # 우선순위 1: 수치+단위 발화
            measure_matches = self.MEASUREMENT_RE.findall(text)
            if measure_matches:
                desc = self._describe_measurement(mid, measure_matches, hand_active)
                highlights.append(BehavioralHighlight(
                    timestamp_sec       = mid,
                    highlight_type      = 'measurement_speech',
                    trigger_reason      = f'수치 단위 발화: {measure_matches}',
                    feature_description = desc,
                    speech_quote        = text,
                    detected_tools      = domain_tools,
                    hand_active         = hand_active,
                ))
                continue  # 상위 규칙 매칭 시 하위 규칙 스킵

            # 우선순위 2: 절차/주의 키워드 발화
            hit_kw = [kw for kw in self.PROCEDURAL_KEYWORDS if kw.lower() in text.lower()]
            if hit_kw:
                desc = self._describe_procedural(mid, hit_kw, hand_active)
                highlights.append(BehavioralHighlight(
                    timestamp_sec       = mid,
                    highlight_type      = 'procedural_speech',
                    trigger_reason      = f'절차 키워드 발화: {hit_kw}',
                    feature_description = desc,
                    speech_quote        = text,
                    detected_tools      = domain_tools,
                    hand_active         = hand_active,
                ))
                continue

            # 우선순위 3: 도메인 관련 객체 탐지 중 발화
            if domain_tools:
                desc = self._describe_tool_active(mid, domain_tools, hand_active)
                highlights.append(BehavioralHighlight(
                    timestamp_sec       = mid,
                    highlight_type      = 'tool_active_speech',
                    trigger_reason      = f'작업 중 발화: {domain_tools}',
                    feature_description = desc,
                    speech_quote        = text,
                    detected_tools      = domain_tools,
                    hand_active         = hand_active,
                ))

        # 규칙 4: 손 동작 감지 (비음성 구간)
        speech_ranges = [(seg.get('start_sec', 0), seg.get('end_sec', 0)) for seg in audio_segs]
        for frame in frame_feats:
            ts = frame.get('timestamp_sec', 0)
            hm = frame.get('hand_movement_vector', {})
            if not (hm.get('left_hand') or hm.get('right_hand')):
                continue
            if any(s <= ts <= e for s, e in speech_ranges):
                continue
            domain_tools = [
                t['label'] for t in frame.get('detected_tools', [])
                if t['label'] in self.DOMAIN_RELEVANT
            ]
            highlights.append(BehavioralHighlight(
                timestamp_sec       = ts,
                highlight_type      = 'active_hands',
                trigger_reason      = '손 동작 감지 (비음성 구간)',
                feature_description = f'[{ts:.1f}초] 발화 없는 손 작업 집중 구간',
                detected_tools      = domain_tools,
                hand_active         = True,
            ))

        logger.info(f"[Highlighter] {len(highlights)}개 하이라이트 추출 완료")
        return highlights

    # ------------------------------------------------------------------
    # 상황 기술문 생성 메서드
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_measurement(ts: float, matches: list, hand_active: bool) -> str:
        vals = ' / '.join(str(m) for m in matches[:3])
        hand = ', 손 동작 활성' if hand_active else ''
        return f"[{ts:.1f}초] 정비 수치 발화 감지 — {vals} 언급{hand}"

    @staticmethod
    def _describe_procedural(ts: float, keywords: list, hand_active: bool) -> str:
        kw = ', '.join(keywords[:3])
        hand = ', 손 동작 활성' if hand_active else ''
        return f"[{ts:.1f}초] 작업 절차/주의사항 발화 — 키워드: {kw}{hand}"

    @staticmethod
    def _describe_tool_active(ts: float, tools: list, hand_active: bool) -> str:
        tool_str = ', '.join(tools[:3])
        hand = ', 손 동작 활성' if hand_active else ''
        return f"[{ts:.1f}초] 작업 중 발화 감지 — 감지 객체: {tool_str}{hand}"


def run_from_file(features_path: str, output_path: str) -> List[BehavioralHighlight]:
    """features JSON 파일을 읽어 하이라이트를 추출하고 저장한다."""
    with open(features_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
    highlights = BehavioralHighlighter().run(features)
    save_highlights(highlights, output_path)
    return highlights


def save_highlights(highlights: List[BehavioralHighlight], path: str) -> None:
    """BehavioralHighlight 목록을 JSON 파일로 저장한다."""
    data = [
        {
            'timestamp_sec':       h.timestamp_sec,
            'highlight_type':      h.highlight_type,
            'trigger_reason':      h.trigger_reason,
            'feature_description': h.feature_description,
            'speech_quote':        h.speech_quote,
            'detected_tools':      h.detected_tools,
            'hand_active':         h.hand_active,
        }
        for h in highlights
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[Highlighter] 하이라이트 저장: {path}")
