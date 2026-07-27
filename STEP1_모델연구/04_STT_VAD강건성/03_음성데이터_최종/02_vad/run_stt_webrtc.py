# ======================================================================
# [인수인계 헤더] 용도: VAD 비교 실험 조건 C(WebRTC 트리밍본) STT 실행 — faster-whisper large-v3-turbo + 도메인 프롬프트, GPU 필요
# 입력: dataset/audio_webrtc/clip{1..4}.wav
# 출력: results_vad_compare/stt_outputs/C_webrtc/clip{1..4}.txt, results_vad_compare/metrics/stt_timing_C_webrtc.csv
# 실행 예시: srun -p RTX6000 -w n5 --gres=gpu:1 .venv/bin/python scripts/run_stt_webrtc.py
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
VAD 방식 비교 실험(조건 C: WebRTC VAD)용 STT 실행 스크립트.
faster_whisper_prompt 파라미터를 A/B 조건과 동일하게 고정하고,
dataset/audio_webrtc/clip{n}.wav를 입력으로 사용한다.
"""

import csv
import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel

PROJ = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJ / "dataset" / "audio_webrtc"
OUT_DIR = PROJ / "results_vad_compare" / "stt_outputs" / "C_webrtc"
TIMING_LOG = PROJ / "results_vad_compare" / "metrics" / "stt_timing_C_webrtc.csv"

CLIPS = ["clip1", "clip2", "clip3", "clip4"]
DOMAIN_PROMPT = "Dell Precision 7920 조립, Motherboard, RAM, BIOS, GPU, 슬롯"


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)

    print("[C_webrtc] 모델 로딩 중 (large-v3-turbo, GPU float16)...")
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    rows = []
    for clip in CLIPS:
        wav_path = AUDIO_DIR / f"{clip}.wav"
        audio_sec = wav_duration_sec(wav_path)

        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            str(wav_path),
            language="ko",
            initial_prompt=DOMAIN_PROMPT,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        infer_sec = time.perf_counter() - t0
        rtf = infer_sec / audio_sec

        (OUT_DIR / f"{clip}.txt").write_text(text.strip() + "\n", encoding="utf-8")
        rows.append({"clip_id": clip, "infer_sec": f"{infer_sec:.3f}",
                     "trimmed_audio_sec": f"{audio_sec:.3f}", "rtf_vs_trimmed": f"{rtf:.4f}"})
        print(f"[C_webrtc] {clip}: infer={infer_sec:.2f}s (trimmed {audio_sec:.1f}s), RTF_vs_trimmed={rtf:.4f}")

    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "infer_sec", "trimmed_audio_sec", "rtf_vs_trimmed"])
        w.writeheader()
        w.writerows(rows)
    print("완료:", TIMING_LOG)


if __name__ == "__main__":
    main()
