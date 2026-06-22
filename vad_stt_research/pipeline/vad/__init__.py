from .silero_vad import SileroVAD
from .pyannote_vad import PyannoteVAD
from .webrtc_vad import WebRTCVAD
from .librosa_vad import LibrosaVAD

VAD_REGISTRY = {
    "silero": SileroVAD,
    "pyannote": PyannoteVAD,
    "webrtc": WebRTCVAD,
    "librosa": LibrosaVAD,
}

def get_vad(engine: str, **kwargs):
    if engine not in VAD_REGISTRY:
        raise ValueError(f"Unknown VAD engine: {engine}. Choose from {list(VAD_REGISTRY)}")
    return VAD_REGISTRY[engine](**kwargs)
