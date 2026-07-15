"""
STEP3 오디오 추출 단독 실행 엔트리포인트.

master_videos(또는 --video-dir)의 영상에서 오디오 트랙만 뽑아
output/audio/<video_id>.wav (16kHz mono — STT 표준 입력)로 저장한다.

사용:
    python3 run_audio_extract.py                       # master_videos 4개 전부
    python3 run_audio_extract.py --video-dir <폴더>    # 다른 폴더
    python3 run_audio_extract.py --no-skip-existing    # 이미 있어도 재추출

GPU 불필요 — 로그인 노드에서 바로 실행 가능(ffmpeg subprocess만 사용).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project-ai 루트 → tacit_common
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_VIDEO_DIR = "/home/ai_user/team_a2/members/안나경/master/master_videos"
OUT_DIR = Path(__file__).resolve().parent / "output" / "audio"


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP3 오디오 추출(영상→16kHz mono WAV)")
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="입력 영상 폴더")
    ap.add_argument("--config", default="../파이프라인_통합실행/config.yaml",
                    help="ffmpeg_bin을 읽어올 config.yaml")
    ap.add_argument("--no-skip-existing", action="store_true", help="기존 wav가 있어도 재추출")
    args = ap.parse_args()

    # 프리플라이트(5초 안에 죽는 검증만): 입력 폴더/영상 존재, ffmpeg 경로, 출력 폴더 쓰기
    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob("*.mp4"))
    assert video_dir.is_dir(), f"입력 폴더 없음: {video_dir}"
    assert videos, f"입력 폴더에 mp4가 없음: {video_dir}"

    ffmpeg_bin = None
    cfg_path = Path(__file__).resolve().parent / args.config
    if cfg_path.exists():
        from tacit_common.config import PipelineConfig
        cfg = PipelineConfig.load(str(cfg_path))
        ffmpeg_bin = cfg.frame_extraction.ffmpeg_bin

    from step3_components.audio_extract import extract_audio
    from step3_components.frame_extract import _ffmpeg_bin, probe_duration
    assert Path(_ffmpeg_bin(ffmpeg_bin)).exists(), f"ffmpeg 없음: {_ffmpeg_bin(ffmpeg_bin)}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done, skipped = 0, 0
    for v in videos:
        out = OUT_DIR / f"{v.name}.wav"
        if out.exists() and out.stat().st_size > 0 and not args.no_skip_existing:
            print(f"[SKIP] {out.name} (이미 있음)")
            skipped += 1
            continue
        extract_audio(str(v), str(out), ffmpeg_bin=ffmpeg_bin)
        dur = probe_duration(str(out), ffmpeg_bin)
        vdur = probe_duration(str(v), ffmpeg_bin)
        # 스모크 검증: 결과가 비어있지 않고, 길이가 원본과 대체로 일치해야 통과
        assert dur > 0 and abs(dur - vdur) < 2.0, \
            f"오디오 길이 이상: {out.name} audio={dur:.1f}s video={vdur:.1f}s"
        print(f"[OK] {v.name} → {out.name} ({dur:.1f}s, {out.stat().st_size // 1024}KB)")
        done += 1

    print(f"완료: 추출 {done}건, 스킵 {skipped}건 → {OUT_DIR}")


if __name__ == "__main__":
    main()
