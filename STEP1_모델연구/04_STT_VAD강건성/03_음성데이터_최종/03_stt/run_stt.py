# ======================================================================
# [인수인계 헤더] 용도: STT 모델 비교 실행 — faster-whisper large-v3-turbo(프롬프트 유/무), speechbrain KsponSpeech, OWSM v4 + RTF 기록, GPU 필요
# 입력: dataset/audio_vad/clip{1..4}.wav (VAD 전처리본; raw vs vad 실험 때는 AUDIO_DIR를 dataset/audio로 바꿔 실행했음)
# 출력: results/stt_outputs/{모델태그}/clip{n}.txt, results/metrics/timing_log.csv
# 실행 예시: srun -p RTX6000 -w n5 --gres=gpu:1 .venv/bin/python scripts/run_stt.py --model all
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
STT 모델 비교 실행 스크립트.
3개 모델(faster-whisper large-v3-turbo[+prompt 유/무], speechbrain KsponSpeech conformer,
espnet OWSM v4 medium 1B)을 동일한 audio/clip{1..4}.wav 입력으로 순차 실행하고,
결과 텍스트와 RTF(추론 시간/음성 길이, 모델 로딩 시간 제외)를 기록한다.

사용법:
    python run_stt.py --model faster_whisper
    python run_stt.py --model speechbrain
    python run_stt.py --model owsm
    python run_stt.py --model all
"""

import argparse
import csv
import json
import time
import wave
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJ / "dataset" / "audio_vad"
OUT_DIR = PROJ / "results" / "stt_outputs"
TIMING_LOG = PROJ / "results" / "metrics" / "timing_log.csv"

CLIPS = ["clip1", "clip2", "clip3", "clip4"]

DOMAIN_PROMPT = "Dell Precision 7920 조립, Motherboard, RAM, BIOS, GPU, 슬롯"


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def append_timing(row: dict) -> None:
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not TIMING_LOG.exists()
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clip_id", "infer_sec", "audio_sec", "rtf"])
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def write_transcript(out_path: Path, text: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.strip() + "\n", encoding="utf-8")


def is_done(tag: str, clip: str) -> bool:
    return (OUT_DIR / tag / f"{clip}.txt").exists()


def run_faster_whisper() -> None:
    variants = [
        ("faster_whisper_noprompt", None),
        ("faster_whisper_prompt", DOMAIN_PROMPT),
    ]

    if all(is_done(tag, clip) for tag, _ in variants for clip in CLIPS):
        print("[faster_whisper] 모든 clip 이미 완료, 건너뜀")
        return

    from faster_whisper import WhisperModel

    print("[faster_whisper] 모델 로딩 중 (large-v3-turbo, GPU float16)...")
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    for tag, prompt in variants:
        for clip in CLIPS:
            if is_done(tag, clip):
                print(f"[{tag}] {clip}: 이미 완료, 건너뜀")
                continue
            wav_path = AUDIO_DIR / f"{clip}.wav"
            audio_sec = wav_duration_sec(wav_path)

            t0 = time.perf_counter()
            segments, _info = model.transcribe(
                str(wav_path),
                language="ko",
                initial_prompt=prompt,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            infer_sec = time.perf_counter() - t0

            rtf = infer_sec / audio_sec
            write_transcript(OUT_DIR / tag / f"{clip}.txt", text)
            append_timing({
                "model": tag, "clip_id": clip,
                "infer_sec": f"{infer_sec:.3f}", "audio_sec": f"{audio_sec:.3f}", "rtf": f"{rtf:.4f}",
            })
            print(f"[{tag}] {clip}: RTF={rtf:.4f}  ({infer_sec:.1f}s / {audio_sec:.1f}s)")


SPEECHBRAIN_CHUNK_SEC = 20.0


def run_speechbrain() -> None:
    if all(is_done("speechbrain", clip) for clip in CLIPS):
        print("[speechbrain] 모든 clip 이미 완료, 건너뜀")
        return

    import torch
    from speechbrain.inference.ASR import EncoderDecoderASR

    print("[speechbrain] 모델 로딩 중 (asr-conformer-transformerlm-ksponspeech)...")
    asr = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-conformer-transformerlm-ksponspeech",
        savedir="pretrained_models/asr-conformer-transformerlm-ksponspeech",
        run_opts={"device": "cuda"},
    )

    for clip in CLIPS:
        if is_done("speechbrain", clip):
            print(f"[speechbrain] {clip}: 이미 완료, 건너뜀")
            continue
        wav_path = AUDIO_DIR / f"{clip}.wav"
        audio_sec = wav_duration_sec(wav_path)
        full_wav = asr.load_audio(str(wav_path))

        # 이 conformer 모델은 긴 오디오를 통째로 넣으면 self-attention 메모리/시간이
        # O(n^2)로 커져 3분 클립 하나에 CPU에서 수십 분+수십GB RAM이 걸린다(실측 확인).
        # OWSM과 동일한 이유로, 짧은 창으로 나눠 순차 전사 후 이어붙인다.
        sr = 16000
        chunk_len = int(SPEECHBRAIN_CHUNK_SEC * sr)
        texts = []

        t0 = time.perf_counter()
        for start in range(0, full_wav.shape[0], chunk_len):
            chunk = full_wav[start:start + chunk_len]
            wavs = chunk.unsqueeze(0)
            wav_lens = torch.ones(1)
            chunk_text, _tokens = asr.transcribe_batch(wavs, wav_lens)
            if chunk_text and chunk_text[0].strip():
                texts.append(chunk_text[0].strip())
        text = " ".join(texts)
        infer_sec = time.perf_counter() - t0

        rtf = infer_sec / audio_sec
        write_transcript(OUT_DIR / "speechbrain" / f"{clip}.txt", text)
        append_timing({
            "model": "speechbrain", "clip_id": clip,
            "infer_sec": f"{infer_sec:.3f}", "audio_sec": f"{audio_sec:.3f}", "rtf": f"{rtf:.4f}",
        })
        print(f"[speechbrain] {clip}: RTF={rtf:.4f}  ({infer_sec:.1f}s / {audio_sec:.1f}s)")


def run_owsm() -> None:
    if all(is_done("owsm", clip) for clip in CLIPS):
        print("[owsm] 모든 clip 이미 완료, 건너뜀")
        return

    from espnet2.bin.s2t_inference import Speech2Text
    import soundfile as sf

    print("[owsm] 모델 로딩 중 (espnet/owsm_v4_medium_1B)...")
    speech2text = Speech2Text.from_pretrained(
        "espnet/owsm_v4_medium_1B",
        device="cuda",
        lang_sym="<kor>",
        task_sym="<asr>",
    )

    for clip in CLIPS:
        if is_done("owsm", clip):
            print(f"[owsm] {clip}: 이미 완료, 건너뜀")
            continue
        wav_path = AUDIO_DIR / f"{clip}.wav"
        audio_sec = wav_duration_sec(wav_path)
        speech, _sr = sf.read(str(wav_path))

        # OWSM은 __call__ 한번에 학습 시 고정 길이(약 30초)로 오디오를 자르므로,
        # 3분 내외의 원본 클립을 그대로 넣으면 앞부분만 처리된다. decode_long()이
        # 내부적으로 30초 단위 청킹 + 타임스탬프 기반 오프셋 이동을 해준다.
        t0 = time.perf_counter()
        utterances = speech2text.decode_long(speech)
        text = " ".join(u[2].strip() for u in utterances if u[2].strip())
        infer_sec = time.perf_counter() - t0

        rtf = infer_sec / audio_sec
        write_transcript(OUT_DIR / "owsm" / f"{clip}.txt", text)
        append_timing({
            "model": "owsm", "clip_id": clip,
            "infer_sec": f"{infer_sec:.3f}", "audio_sec": f"{audio_sec:.3f}", "rtf": f"{rtf:.4f}",
        })
        print(f"[owsm] {clip}: RTF={rtf:.4f}  ({infer_sec:.1f}s / {audio_sec:.1f}s)")


RUNNERS = {
    "faster_whisper": run_faster_whisper,
    "speechbrain": run_speechbrain,
    "owsm": run_owsm,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(RUNNERS) + ["all"], required=True)
    args = parser.parse_args()

    targets = list(RUNNERS) if args.model == "all" else [args.model]
    for name in targets:
        RUNNERS[name]()


if __name__ == "__main__":
    main()
