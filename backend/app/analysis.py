import numpy as np
import librosa

CQT_HOP_LENGTH = 1024


def detect_key(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Detect musical key using madmom CNN if available, falling back to
    librosa Krumhansl-Schmuckler. Returns (key_name, confidence).
    """
    try:
        from madmom.features.key import CNNKeyRecognitionProcessor
        proc = CNNKeyRecognitionProcessor()
        key_str = proc(audio)
        return key_str, 0.9
    except (ImportError, OSError):
        return _detect_key_librosa(audio, sr)


def _detect_key_librosa(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """Fallback key detection using librosa K-S algorithm."""
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=CQT_HOP_LENGTH
    )
    chroma_mean = chroma.mean(axis=1)

    # Krumhansl-Schmuckler major/minor profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    major_corr = [np.corrcoef(np.roll(chroma_mean, i), major_profile)[0, 1]
                  for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(chroma_mean, i), minor_profile)[0, 1]
                  for i in range(12)]

    best_major = int(np.argmax(major_corr))
    best_minor = int(np.argmax(minor_corr))

    if max(major_corr) >= max(minor_corr):
        confidence = _normalize_corr(max(major_corr))
        return f"{pitch_names[best_major]} major", round(confidence, 3)
    else:
        confidence = _normalize_corr(max(minor_corr))
        return f"{pitch_names[best_minor]} minor", round(confidence, 3)


def _normalize_corr(corr: float) -> float:
    """Map raw correlation to 0-1 range."""
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))
