"""
faster-whisper 내장 vad_filter=True 검증 — STEP3_전처리(프로덕션) 반영 가능성 타진.

06_VAD재검증_v2에서 지금까지 쓴 VAD 방식(Silero로 발화 구간만 잘라 이어붙임, `collect_chunks`)은
결과 오디오가 원본보다 훨씬 짧아져 Whisper가 내놓는 seg.start/end가 원본 영상 타임라인과
어긋난다 — STEP3_전처리/tacit_pipeline/components/aligner.py가 발화 타임스탬프를 VLM 액션
타임스탬프와 ±윈도우로 직접 매칭하기 때문에 이 어긋남은 그대로 정렬 오류가 된다
(`_내일_서버전환_가이드.md`에 이미 "VAD는 병합 부작용 → OFF 유지"로 기록돼 있음).

faster-whisper에 내장된 vad_filter=True는 내부적으로 Silero VAD를 쓰지만 오디오를 잘라
붙이지 않고 무음 구간만 스킵하면서 세그먼트 타임스탬프를 원본 오디오 기준으로 리포트한다
(문서상 동작 — 이 스크립트로 실측 확인). 원본(트리밍 안 한) CLIP1~4를 그대로 넣어 두 가지를
같이 확인한다:
  1) CER이 06의 (병합)VAD 방식만큼 개선되는가.
  2) 세그먼트 타임스탬프가 실제로 원본 오디오 전체 구간에 분포하는가(병합 방식처럼
     트리밍 후 길이 안으로 쏠려있지 않은가) — aligner.py 호환성 실측.

사용법 (GPU 노드, 01_STT선정/.venv 파이썬 + LD_LIBRARY_PATH 필수):
  LD_LIBRARY_PATH=... .venv/bin/python3 scripts/run_vad_filter_builtin.py
"""
import csv
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STT_PROJECT = HERE.parent / "01_STT선정"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STT_PROJECT))

CLIPS = ["CLIP1", "CLIP2", "CLIP3", "CLIP4"]
REPEATS = 3
MODEL_ID = "large-v3-turbo"

GT_DIR = HERE / "data" / "ground_truth"
NO_VAD_DIR = STT_PROJECT / "data" / "raw_v2"  # 원본(트리밍 안 함) 오디오

OUT_DIR = HERE / "results" / "vad_filter_builtin"
TRANSCRIPTS_DIR = OUT_DIR / "transcripts"

RESULT_COLS = [
    "clip", "cer_mean", "cer_std", "wer_mean", "wer_std", "rtf_mean", "rtf_std", "n_repeats",
    "audio_duration_s", "seg_first_start_s", "seg_last_end_s", "n_segments",
    "timestamp_sanity",
]


def _save_texts(clip, raw_text, filtered_text):
    raw_dir = TRANSCRIPTS_DIR / "raw"
    filtered_dir = TRANSCRIPTS_DIR / "filtered"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{clip}.txt").write_text(raw_text, encoding="utf-8")
    (filtered_dir / f"{clip}.txt").write_text(filtered_text, encoding="utf-8")


def main():
    from pipeline.stt.faster_whisper_runner import FasterWhisperRunner
    from evaluation.hallucination_filter import filter_hallucinations
    from evaluation.metrics import evaluate

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runner = FasterWhisperRunner(
        model_id=MODEL_ID,
        decoding_params={
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},  # 06에서 확정한 Silero 기본값과 동일하게
        },
    )

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()

        t0 = time.perf_counter()
        for clip in CLIPS:
            wav_path = str(NO_VAD_DIR / f"{clip}.wav")
            gt_raw = (GT_DIR / f"{clip}.txt").read_text(encoding="utf-8").strip()

            cers, wers, rtfs = [], [], []
            last_raw = last_filtered = last_m = None
            last_result = None
            for _ in range(REPEATS):
                result = runner.transcribe(wav_path)
                raw_text = result.full_text()
                filtered_text = filter_hallucinations(raw_text)
                m = evaluate(filtered_text, gt_raw, segments=result.segments)
                cers.append(m.cer)
                wers.append(m.wer)
                rtfs.append(result.rtf)
                last_raw, last_filtered, last_m, last_result = raw_text, filtered_text, m, result

            cer_mean = statistics.mean(cers)
            cer_std = statistics.pstdev(cers) if len(cers) > 1 else 0.0
            wer_mean = statistics.mean(wers)
            wer_std = statistics.pstdev(wers) if len(wers) > 1 else 0.0
            rtf_mean = statistics.mean(rtfs)
            rtf_std = statistics.pstdev(rtfs) if len(rtfs) > 1 else 0.0

            segs = last_result.segments
            first_start = min(s.start for s in segs) if segs else -1.0
            last_end = max(s.end for s in segs) if segs else -1.0
            duration = last_result.audio_duration_s
            # 병합(트리밍) 방식이었다면 last_end가 원본 duration의 극히 일부(트리밍후 길이)에
            # 그친다 — 원본 전체 구간에 걸쳐 분포하면 타임스탬프가 살아있다는 뜻.
            sanity = "원본기준유지" if last_end >= duration * 0.5 else "의심(트리밍후처럼 짧게 쏠림)"

            _save_texts(clip, last_raw, last_filtered)

            writer.writerow({
                "clip": clip,
                "cer_mean": round(cer_mean, 4), "cer_std": round(cer_std, 4),
                "wer_mean": round(wer_mean, 4), "wer_std": round(wer_std, 4),
                "rtf_mean": round(rtf_mean, 4), "rtf_std": round(rtf_std, 4),
                "n_repeats": REPEATS,
                "audio_duration_s": round(duration, 2),
                "seg_first_start_s": round(first_start, 2),
                "seg_last_end_s": round(last_end, 2),
                "n_segments": len(segs),
                "timestamp_sanity": sanity,
            })
            print(f"  {clip}: CER={cer_mean:.4f}(±{cer_std:.4f}) RTF={rtf_mean:.4f} "
                  f"duration={duration:.1f}s seg_range=[{first_start:.1f},{last_end:.1f}] "
                  f"n_seg={len(segs)} sanity={sanity} repeats={cers}")
        print(f"\n총 소요: {time.perf_counter() - t0:.1f}s")

    print(f"\n결과 저장: {csv_path}")
    print(f"전사 텍스트: {TRANSCRIPTS_DIR}/raw|filtered/")


if __name__ == "__main__":
    main()
