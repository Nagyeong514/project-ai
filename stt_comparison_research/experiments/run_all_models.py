"""
단일 오디오 파일에 대해 6-arm 전체 모델을 실행하고 결과 dict 반환.
RTF 측정: warmup 1회 + measurement 3회, 평균±SD.
"""
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from evaluation.metrics import evaluate
from pipeline.audio.preprocessor import get_audio_duration
from pipeline.stt import BaseSTT, STTResult, get_stt


def _run_with_rtf(
    runner: BaseSTT,
    audio_path: str,
    warmup_runs: int,
    measurement_runs: int,
) -> tuple[STTResult, float, float]:
    """워밍업 후 반복 측정. (last_result, mean_rtf, std_rtf) 반환."""
    for _ in range(warmup_runs):
        runner.transcribe(audio_path)

    rtf_values = []
    result = None
    for _ in range(measurement_runs):
        result = runner.transcribe(audio_path)
        rtf_values.append(result.rtf)

    mean_rtf = statistics.mean(rtf_values)
    std_rtf = statistics.stdev(rtf_values) if len(rtf_values) > 1 else 0.0
    return result, mean_rtf, std_rtf


def run_all_models(
    audio_path: str,
    ground_truth: str,
    model_configs: Dict[str, Any],
    rtf_cfg: Dict[str, int],
    is_api: bool = False,
) -> List[Dict[str, Any]]:
    """
    audio_path: 16kHz mono WAV
    ground_truth: 해당 파일의 정규화 전 텍스트 (normalizer가 내부에서 처리)
    model_configs: experiment_config.yaml 의 models 블록
    rtf_cfg: {warmup_runs, measurement_runs}
    is_api: True면 RTF 측정 1회만 수행 (네트워크 지연 포함, 별도 표기)
    """
    results = []
    warmup = rtf_cfg.get("warmup_runs", 1)
    repeats = rtf_cfg.get("measurement_runs", 3)

    for model_key, cfg in model_configs.items():
        runner = get_stt(cfg)
        api_flag = cfg["engine"] in ("clova_api", "kakao_api")

        if api_flag:
            # API는 1회만 측정, RTF에 '(net)' 태그
            result = runner.transcribe(audio_path)
            mean_rtf, std_rtf = result.rtf, 0.0
        else:
            result, mean_rtf, std_rtf = _run_with_rtf(runner, audio_path, warmup, repeats)

        scores = evaluate(result.full_text(), ground_truth)
        results.append(
            {
                "model_key": model_key,
                "model_label": cfg.get("label", model_key),
                "cer": round(scores["cer"], 4),
                "wer": round(scores["wer"], 4),
                "rtf_mean": round(mean_rtf, 4),
                "rtf_std": round(std_rtf, 4),
                "rtf_note": "net" if api_flag else "",
                "deployable": cfg.get("deployable", False),
                "hypothesis": scores["hypothesis"],
            }
        )

    return results
