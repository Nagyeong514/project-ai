# ======================================================================
# [인수인계 헤더] 용도: 노이즈 강건성 실험 STT 실행 — D1~D4 노이즈 합성본에 확정 설정(large-v3-turbo, ko, 도메인 프롬프트, float16) 적용, GPU 필요
# 입력: dataset/audio_noise/{D1..D4}/clip{1..4}.wav
# 출력: results_noise_robustness/stt_outputs/{D1..D4}/clip{n}.txt, results_noise_robustness/timing_log.csv
# 실행 예시: sbatch scripts/run_stt_noise.sbatch  (직접 실행 시: srun -p RTX6000 -w n5 --gres=gpu:1 + LD_LIBRARY_PATH 설정 필요)
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
노이즈 강건성 실험 STT 실행 스크립트.

dataset/audio_noise/{D1..D4}/clip{1..4}.wav 를 입력으로,
기존 파이프라인과 완전히 동일한 설정(faster-whisper large-v3-turbo, GPU float16,
language="ko", 도메인 initial_prompt — run_stt.py의 faster_whisper_prompt 조건)으로
STT를 실행한다.

D0(노이즈 없음)는 기존 results/stt_outputs/faster_whisper_prompt/ 결과를 재사용하므로
여기서는 실행하지 않는다.

반드시 GPU 노드에서 실행할 것 (이전 실험과 동일: SLURM srun, n5, RTX6000):
  srun -p RTX6000 -w n5 --gres=gpu:1 .venv/bin/python scripts/run_stt_noise.py

산출물:
  results_noise_robustness/stt_outputs/{D1..D4}/clip{1..4}.txt
  results_noise_robustness/timing_log.csv (model=faster_whisper_prompt, condition, clip_id, rtf)
"""

import csv
import time
import wave
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
NOISE_DIR = PROJ / "dataset" / "audio_noise"
OUT_DIR = PROJ / "results_noise_robustness" / "stt_outputs"
TIMING_LOG = PROJ / "results_noise_robustness" / "timing_log.csv"

CONDS = ["D1", "D2", "D3", "D4"]
CLIPS = ["clip1", "clip2", "clip3", "clip4"]

# run_stt.py 와 동일한 도메인 프롬프트
DOMAIN_PROMPT = "Dell Precision 7920 조립, Motherboard, RAM, BIOS, GPU, 슬롯"


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def append_timing(row: dict) -> None:
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not TIMING_LOG.exists()
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "clip_id", "infer_sec", "audio_sec", "rtf"])
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    import torch
    assert torch.cuda.is_available() or _ctranslate2_cuda_ok(), "GPU 필요: srun으로 n5에서 실행할 것"

    from faster_whisper import WhisperModel

    print("[faster_whisper_prompt] 모델 로딩 중 (large-v3-turbo, GPU float16)...")
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    for cond in CONDS:
        for clip in CLIPS:
            out_path = OUT_DIR / cond / f"{clip}.txt"
            if out_path.exists():
                print(f"[{cond}] {clip}: 이미 완료, 건너뜀")
                continue
            wav_path = NOISE_DIR / cond / f"{clip}.wav"
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
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text.strip() + "\n", encoding="utf-8")
            append_timing({
                "model": "faster_whisper_prompt", "condition": cond, "clip_id": clip,
                "infer_sec": f"{infer_sec:.3f}", "audio_sec": f"{audio_sec:.3f}", "rtf": f"{rtf:.4f}",
            })
            print(f"[{cond}] {clip}: RTF={rtf:.4f}  ({infer_sec:.1f}s / {audio_sec:.1f}s)")


def _ctranslate2_cuda_ok() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


if __name__ == "__main__":
    main()
