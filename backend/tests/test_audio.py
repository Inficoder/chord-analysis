import numpy as np
import soundfile as sf
from pathlib import Path
from app.audio import load_audio, AudioInfo


def test_load_audio_wav(tmp_path: Path):
    # Generate a simple WAV file: 1 second, 44100Hz stereo
    path = tmp_path / "test.wav"
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
    stereo = np.column_stack([samples, samples * 0.5])
    sf.write(str(path), stereo, 44100)

    audio, info = load_audio(path)

    assert info.sample_rate == 22050
    assert audio.ndim == 1  # mono
    assert len(audio) == 22050  # 1s at 22050Hz
    assert isinstance(audio, np.ndarray)


def test_load_audio_info(tmp_path: Path):
    path = tmp_path / "test2.wav"
    samples = np.zeros(44100 * 3, dtype=np.float32)
    sf.write(str(path), samples, 44100)

    _, info = load_audio(path)
    assert abs(info.duration - 3.0) < 0.1
    assert info.original_sample_rate == 44100
