# ======================================================================
# [인수인계 헤더] 용도: 노이즈 강건성 실험용 노이즈 합성 — D1~D3 화이트 노이즈(SNR 20/10/0dB), D4 ESC-50 기계음 혼합(SNR 10dB), seed=42 (CPU 가능)
# 입력: dataset/audio_vad/clip{1..4}.wav, dataset/ESC-50/ (audio + meta/esc50.csv)
# 출력: dataset/audio_noise/{D1..D4}/clip{1..4}.wav, results_noise_robustness/d4_noise_log.txt
# 실행 예시: (프로젝트 루트에서) .venv/bin/python scripts/add_noise.py
# ※ 이 파일은 handover용 사본입니다. 경로가 스크립트 위치 기준으로 계산되므로
#    실제 실행은 원본 위치(프로젝트 루트의 scripts/ 또는 dataset/)의 파일로 하세요.
# ======================================================================
"""
노이즈 강건성 실험용 노이즈 합성 스크립트.

dataset/audio_vad/clip{1..4}.wav (VAD 적용된 16kHz mono, 현재 파이프라인의 실제 STT 입력)에
아래 4가지 조건의 노이즈를 SNR 기준으로 정확히 스케일링하여 섞고,
dataset/audio_noise/{D1,D2,D3,D4}/clip{1..4}.wav 로 저장한다.

  D1: 화이트 노이즈, SNR 20dB
  D2: 화이트 노이즈, SNR 10dB
  D3: 화이트 노이즈, SNR 0dB
  D4: ESC-50 실제 기계음 혼합, SNR 10dB
      - 배경층(연속음): engine/vacuum_cleaner 중 랜덤 1개 파일을 클립 길이에 맞게 loop
      - 이벤트층(간헐음): hand_saw 중 랜덤 1개 파일을 클립 중간중간 2~3회 삽입
        (계획서의 "drilling 또는 hand_saw" 중 drilling은 ESC-50의 50개 카테고리에
         존재하지 않아 hand_saw만 사용 — d4_noise_log.txt 및 보고서에 명시)
      - 두 층을 합친 최종 노이즈와 음성의 SNR이 정확히 10dB가 되도록 스케일
      - ESC-50은 44.1kHz이므로 16kHz로 리샘플링

random seed 고정(SEED=42)으로 재현 가능. D4에서 클립별로 사용한 ESC-50 파일은
results_noise_robustness/d4_noise_log.txt 에 기록한다.

SNR 정의: SNR(dB) = 10 * log10( P_speech / P_noise ),  P = mean(x^2)
"""

import csv
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

PROJ = Path(__file__).resolve().parent.parent
VAD_DIR = PROJ / "dataset" / "audio_vad"
ESC_DIR = PROJ / "dataset" / "ESC-50"
OUT_DIR = PROJ / "dataset" / "audio_noise"
LOG_DIR = PROJ / "results_noise_robustness"

CLIPS = ["clip1", "clip2", "clip3", "clip4"]
SR = 16000
SEED = 42

WHITE_CONDS = {"D1": 20.0, "D2": 10.0, "D3": 0.0}
D4_SNR_DB = 10.0
BG_CATEGORIES = ["engine", "vacuum_cleaner"]
EVENT_CATEGORIES = ["hand_saw"]  # drilling은 ESC-50에 없음


def power(x: np.ndarray) -> float:
    return float(np.mean(x ** 2))


def scale_noise_to_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """speech 대비 snr_db가 되도록 noise를 스케일링해서 반환."""
    p_s = power(speech)
    p_n = power(noise)
    target_p_n = p_s / (10 ** (snr_db / 10.0))
    return noise * np.sqrt(target_p_n / p_n)


def actual_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    return 10 * np.log10(power(speech) / power(noise))


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 클리핑 방지: 피크가 1.0을 넘으면 전체를 같은 비율로 줄임 (SNR 불변)
    peak = np.max(np.abs(x))
    if peak > 0.999:
        x = x * (0.999 / peak)
    sf.write(str(path), x, SR, subtype="PCM_16")


def load_esc_meta() -> dict:
    """category -> [filename, ...] (파일명 정렬 후 반환: seed 재현성 보장)"""
    by_cat = {}
    with open(ESC_DIR / "meta" / "esc50.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_cat.setdefault(row["category"], []).append(row["filename"])
    for cat in by_cat:
        by_cat[cat].sort()
    return by_cat


def load_esc_audio(filename: str) -> np.ndarray:
    """ESC-50 wav(44.1kHz)를 16kHz mono로 로드."""
    y, _sr = librosa.load(str(ESC_DIR / "audio" / filename), sr=SR, mono=True)
    return y.astype(np.float64)


def build_d4_noise(rng: np.random.Generator, speech_len: int, by_cat: dict) -> tuple:
    """배경층(loop) + 이벤트층(2~3회 삽입) 노이즈와 선택 로그를 반환."""
    # 배경층: engine/vacuum_cleaner 풀에서 랜덤 1개 → 클립 길이에 맞게 loop
    bg_pool = [(cat, fn) for cat in BG_CATEGORIES for fn in by_cat[cat]]
    bg_cat, bg_file = bg_pool[rng.integers(len(bg_pool))]
    bg_audio = load_esc_audio(bg_file)
    reps = int(np.ceil(speech_len / len(bg_audio)))
    bg_layer = np.tile(bg_audio, reps)[:speech_len]

    # 이벤트층: hand_saw 풀에서 랜덤 1개 → 클립 중간중간 2~3회 삽입
    ev_pool = [(cat, fn) for cat in EVENT_CATEGORIES for fn in by_cat[cat]]
    ev_cat, ev_file = ev_pool[rng.integers(len(ev_pool))]
    ev_audio = load_esc_audio(ev_file)
    n_events = int(rng.integers(2, 4))  # 2 또는 3회

    ev_layer = np.zeros(speech_len)
    # 클립을 n_events개 구간으로 나눠 각 구간 안의 랜덤 위치에 삽입(중간중간 분산 배치)
    seg_len = speech_len // n_events
    positions = []
    for i in range(n_events):
        seg_start = i * seg_len
        max_start = seg_start + max(1, seg_len - len(ev_audio))
        start = int(rng.integers(seg_start, max_start))
        end = min(start + len(ev_audio), speech_len)
        ev_layer[start:end] += ev_audio[: end - start]
        positions.append(start / SR)

    # 두 층은 원본 진폭 그대로 합산 → 최종 SNR 스케일링은 합산 노이즈에 일괄 적용
    noise = bg_layer + ev_layer
    log = {
        "bg_category": bg_cat, "bg_file": bg_file,
        "event_category": ev_cat, "event_file": ev_file,
        "n_events": n_events, "event_positions_sec": [round(p, 2) for p in positions],
    }
    return noise, log


def main():
    rng = np.random.default_rng(SEED)
    by_cat = load_esc_meta()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    d4_log_lines = [
        "# D4 노이즈 합성 로그 (seed=42)",
        "# 배경층: engine/vacuum_cleaner 중 랜덤 1개를 클립 길이에 맞게 loop",
        "# 이벤트층: hand_saw 중 랜덤 1개를 클립 중간중간 2~3회 삽입",
        "#   * 계획서의 'drilling 또는 hand_saw' 중 drilling 카테고리는 ESC-50에",
        "#     존재하지 않아(50개 카테고리 미포함) hand_saw만 사용함",
        "# 두 층 합산 후 음성 대비 SNR 10dB로 일괄 스케일링",
        "",
    ]

    for clip in CLIPS:
        speech, _sr = librosa.load(str(VAD_DIR / f"{clip}.wav"), sr=SR, mono=True)
        speech = speech.astype(np.float64)
        n = len(speech)

        # D1~D3: 화이트 노이즈
        for cond, snr_db in WHITE_CONDS.items():
            white = rng.standard_normal(n)
            noise = scale_noise_to_snr(speech, white, snr_db)
            mix = speech + noise
            save_wav(OUT_DIR / cond / f"{clip}.wav", mix)
            print(f"[{cond}] {clip}: 목표 SNR={snr_db}dB, 실측 SNR={actual_snr(speech, noise):.2f}dB")

        # D4: ESC-50 기계음 혼합
        raw_noise, log = build_d4_noise(rng, n, by_cat)
        noise = scale_noise_to_snr(speech, raw_noise, D4_SNR_DB)
        mix = speech + noise
        save_wav(OUT_DIR / "D4" / f"{clip}.wav", mix)
        snr_meas = actual_snr(speech, noise)
        print(f"[D4] {clip}: 목표 SNR={D4_SNR_DB}dB, 실측 SNR={snr_meas:.2f}dB, "
              f"bg={log['bg_file']}({log['bg_category']}), ev={log['event_file']}x{log['n_events']}")

        d4_log_lines.append(
            f"{clip}: 배경층={log['bg_file']} (category={log['bg_category']}, loop), "
            f"이벤트층={log['event_file']} (category={log['event_category']}, "
            f"{log['n_events']}회 삽입, 시작위치(초)={log['event_positions_sec']}), "
            f"최종 SNR={snr_meas:.2f}dB"
        )

    (LOG_DIR / "d4_noise_log.txt").write_text("\n".join(d4_log_lines) + "\n", encoding="utf-8")
    print("완료:", OUT_DIR, "/", LOG_DIR / "d4_noise_log.txt")


if __name__ == "__main__":
    main()
