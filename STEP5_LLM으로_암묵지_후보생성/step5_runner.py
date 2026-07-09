"""
STEP5 실행 본체 — 타임스탬프 정렬(Aligner) + LLM 융합.

영상 갈래(STEP4)와 음성 갈래(STEP3)가 합류하는 지점 — 파이프라인 전체에서 transcript와
detections/observations를 동시에 필요로 하는 곳은 여기 하나뿐이다(STEP4는 끝까지
영상만 다룬다). Aligner는 모델을 안 쓰는 순수 로직이라 GPU 순차 로딩 순서에 안 끼어든다.

입력: transcripts/<video_id>.json(STEP3), output/_frames/<video_id>/frames_meta.json(STEP3,
     duration 조회용), output/detections/<video_id>.json(STEP4),
     output/vlm_observations/<video_id>.observations.json(STEP4)
산출물: output/aligned_windows/<video_id>.windows.json, output/tacit_json/<video_id>.tacit.json
"""

from __future__ import annotations

import json
from pathlib import Path

from tacit_common import artifacts
from tacit_common.config import PipelineConfig
from tacit_common.schema.intermediate import FrameMeta, seconds_to_hhmmss
from step5_schema.tacit_schema import TacitKnowledgeDocument
from step5_registry import build_aligner, build_llm


class Step5Runner:
    def __init__(self, cfg: PipelineConfig, aligner=None, llm=None):
        self.cfg = cfg
        self.aligner = aligner or build_aligner(cfg.aligner)
        self.llm = llm or build_llm(cfg.llm)
        self.frames_dir = cfg.paths.resolve(cfg.paths.step3_frames_dir)
        self.transcript_dir = cfg.paths.resolve(cfg.paths.step3_transcript_dir)
        self.detections_dir = cfg.paths.resolve(cfg.paths.step4_detections_dir)
        self.observations_dir = cfg.paths.resolve(cfg.paths.step4_observations_dir)
        self.windows_dir = cfg.paths.resolve(cfg.paths.step5_aligned_windows_dir)
        self.tacit_json_dir = cfg.paths.resolve(cfg.paths.step5_tacit_json_dir)

    def run(self, video_id: str) -> TacitKnowledgeDocument:
        transcript = artifacts.load_transcript(self.transcript_dir, video_id)
        flat_dets = artifacts.load_detections(self.detections_dir, video_id)
        actions = artifacts.load_observations(self.observations_dir, video_id)
        fmeta = artifacts.load_frames_meta(self.frames_dir, video_id)
        dur = fmeta["duration"]

        print(f"[STEP5][1/2] 타임스탬프 정렬: {video_id}")
        windows = self.aligner.align(actions, transcript, flat_dets)
        artifacts.save_aligned_windows(self.windows_dir, video_id, windows)

        print(f"[STEP5][2/2] LLM 융합: {video_id}")
        meta = FrameMeta(video_id=video_id, path=video_id, fps=fmeta["fps"],
                          n_frames=len(fmeta["frame_paths"]))
        doc = self.llm.fuse(windows, meta)
        if hasattr(self.llm, "unload"):
            self.llm.unload()

        # 우리가 이미 아는 결정적 메타데이터는 코드가 채운다(LLM 추측 금지).
        self._finalize_metadata(doc, dur)

        # 교차검증: VLM 관찰문/STT 발화문을 넘겨 '관찰 둔갑'·'발화 지어냄'까지 잡는다.
        obs_texts = [a.action for a in actions]
        utt_texts = [u.raw_text for u in transcript.utterances] + \
                    [u.normalized_text for u in transcript.utterances]
        for c in doc.candidates:
            for w in c.cross_check(observation_texts=obs_texts, utterance_texts=utt_texts):
                print(f"[CROSS-CHECK][{c.id}] {w}")

        self._save(doc)
        # Pass 2(응집) 검증용: 응집 전 Pass 1 스냅샷도 함께 남긴다 — 1건 그룹의
        # 서술이 바이트 동일한지 대조 + 응집 전/후 후보 수 추적(보고 항목).
        snap = getattr(self.llm, "last_pass1_snapshot", None)
        if snap is not None:
            p = Path(self.tacit_json_dir) / f"{doc.video_id}.pass1.json"
            with p.open("w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            print(f"[OK] Pass 1 스냅샷(응집 전 {len(snap.get('candidates', []))}건) 저장 → {p}")
        return doc

    def _finalize_metadata(self, doc: TacitKnowledgeDocument, dur: float) -> None:
        """id / scenario_id / equipment / source.video_id / transcript_ref / clip_start·clip_end
        는 전부 코드가 아는 값 — LLM 출력을 신뢰하지 않고 여기서 덮어쓴다."""
        video_id = doc.video_id
        stem = video_id.rsplit(".", 1)[0].replace("-", "_")
        transcript_ref = f"{self.cfg.transcript_dir}/{video_id}.json"
        clip_end = seconds_to_hhmmss(dur) if dur else None

        for i, c in enumerate(doc.candidates, start=1):
            c.id = f"tk_{stem}_{i:03d}"
            c.metadata.scenario_id = stem
            c.metadata.equipment = self.cfg.equipment
            src = c.metadata.source
            src.video_id = video_id
            src.transcript_ref = transcript_ref
            src.clip_start = "00:00:00"
            src.clip_end = clip_end

    def _save(self, doc: TacitKnowledgeDocument) -> str:
        out_dir = Path(self.tacit_json_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{doc.video_id}.tacit.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 암묵지 후보 {len(doc.candidates)}건 저장 → {path}")
        return str(path)
