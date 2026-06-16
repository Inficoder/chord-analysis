import soundfile as sf
import librosa
import numpy as np
from pathlib import Path


def load_audio(file_path: Path, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at target_sr. Returns (samples, sr)."""
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def get_duration(file_path: Path) -> float:
    """Get duration in seconds."""
    info = sf.info(str(file_path))
    return info.duration


def write_audio(file_path: Path, audio: np.ndarray, sr: int):
    """Write mono float32 audio to file."""
    sf.write(str(file_path), audio, sr)


def validate_audio(
    file_path: Path, allowed_exts: set[str], max_size: int, max_duration: float,
) -> str | None:
    """Return error message if invalid, None if valid."""
    suffix = file_path.suffix.lower()
    if suffix not in allowed_exts:
        return f"Unsupported format: {suffix}. Allowed: {allowed_exts}"
    size = file_path.stat().st_size
    if size > max_size:
        return f"File too large: {size} bytes (max {max_size})"
    try:
        dur = get_duration(file_path)
        if dur > max_duration:
            return f"File too long: {dur:.0f}s (max {max_duration}s)"
    except Exception:
        pass
    return None
