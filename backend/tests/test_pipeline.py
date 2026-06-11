import numpy as np
import soundfile as sf
from app.pipeline import run_pipeline
from app.models import AnalysisResult


def test_run_pipeline_with_sine_wav(tmp_path):
    """Create a simple sine wave WAV (C major triad) and run full pipeline."""
    path = tmp_path / "test.wav"
    sr = 22050
    t = np.linspace(0, 3, sr * 3)
    c = np.sin(2 * np.pi * 261.6 * t)
    e = np.sin(2 * np.pi * 329.6 * t)
    g = np.sin(2 * np.pi * 392.0 * t)
    audio = ((c + e + g) / 3).astype(np.float32)
    sf.write(str(path), audio, sr)

    result = run_pipeline(path)

    assert isinstance(result, AnalysisResult)
    assert result.key is not None
    assert len(result.key) > 0
    assert result.key_confidence >= 0.0
    assert len(result.chords) > 0
    assert all(c.function for c in result.chords)
