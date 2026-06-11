from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 22050


@dataclass
class AudioInfo:
    sample_rate: int
    duration: float
    original_sample_rate: int


def load_audio(path: Path) -> tuple[np.ndarray, AudioInfo]:
    """Load audio file, convert to mono 22050Hz. Returns (samples, info)."""
    audio, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    original_sr = sf.info(str(path)).samplerate
    duration = len(audio) / TARGET_SR
    return audio, AudioInfo(
        sample_rate=TARGET_SR,
        duration=duration,
        original_sample_rate=original_sr,
    )
