import numpy as np
from app.analysis import detect_key


def test_detect_key_simple_major():
    """Generate a C major chord arpeggio and verify key detection."""
    sr = 22050
    t = np.linspace(0, 2, sr * 2)
    # C major triad notes: C4=261.6, E4=329.6, G4=392.0
    c = np.sin(2 * np.pi * 261.6 * t)
    e = np.sin(2 * np.pi * 329.6 * t)
    g = np.sin(2 * np.pi * 392.0 * t)
    audio = (c + e + g) / 3
    audio = audio.astype(np.float32)

    key, confidence = detect_key(audio, sr)

    assert key in ["C major", "C"], f"Unexpected key: {key}"
    assert confidence > 0.3


def test_detect_key_returns_confidence_range():
    sr = 22050
    audio = np.random.randn(sr * 2).astype(np.float32)  # noise, still should run
    key, confidence = detect_key(audio, sr)
    assert 0.0 <= confidence <= 1.0
    assert isinstance(key, str)
