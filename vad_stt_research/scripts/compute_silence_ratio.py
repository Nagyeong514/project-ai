"""
실험 착수 전 데이터셋 메타데이터 생성:
각 오디오 파일의 무음 비율을 사전 계산해 metadata.csv 에 저장.
고정 기준 VAD(Silero 기본값)로 산출.
"""
import argparse
import csv
import os
from pathlib import Path

import soundfile as sf

from pipeline.vad.silero_vad import SileroVAD
from pipeline.merge.chunk_extractor import compute_silence_ratio


def main(audio_dir: str, output_csv: str) -> None:
    vad = SileroVAD(threshold=0.5, speech_pad_ms=0)   # padding 없이 순수 비율 측정
    audio_files = sorted(Path(audio_dir).glob("**/*.wav"))

    rows = []
    for path in audio_files:
        info = sf.info(str(path))
        duration_min = info.duration / 60.0
        if duration_min < 60:
            print(f"[건너뜀] {path.name} ({duration_min:.1f}분 < 60분)")
            continue

        print(f"처리 중: {path.name} ({duration_min:.1f}분)")
        try:
            segments = vad.detect(str(path))
            ratio = compute_silence_ratio(str(path), segments)
        except Exception as e:
            print(f"  [오류] {e}")
            ratio = None

        group = (
            "low_silence" if ratio is not None and ratio <= 0.20
            else "high_silence" if ratio is not None and ratio >= 0.50
            else "mid_silence"
        )
        rows.append({
            "file_id": path.stem,
            "file_path": str(path),
            "duration_min": round(duration_min, 2),
            "silence_ratio": round(ratio, 4) if ratio is not None else "",
            "group": group,
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "file_path", "duration_min", "silence_ratio", "group"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n메타데이터 저장 완료: {output_csv} ({len(rows)}개 파일)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="무음 비율 사전 계산")
    parser.add_argument("audio_dir", help="오디오 파일 디렉토리")
    parser.add_argument("--output", default="data/metadata.csv", help="출력 CSV 경로")
    args = parser.parse_args()
    main(args.audio_dir, args.output)
