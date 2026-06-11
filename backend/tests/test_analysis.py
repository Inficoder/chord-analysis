import numpy as np
from app.analysis import detect_key, recognize_chords, ChordSegment, analyze_functions


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


def test_recognize_chords_returns_events():
    """Even with noise, should return a list of chord events."""
    sr = 22050
    audio = np.random.randn(sr * 3).astype(np.float32)

    chords = recognize_chords(audio, sr)

    assert isinstance(chords, list)
    assert len(chords) > 0
    for ev in chords:
        assert hasattr(ev, "start")
        assert hasattr(ev, "end")
        assert hasattr(ev, "chord")
        assert ev.start < ev.end


def test_analyze_functions_in_c_major():
    """I, IV, V in C major should be T, S, D."""
    key = "C major"
    chords = [
        ChordSegment(start=0.0, end=1.0, chord="C"),
        ChordSegment(start=1.0, end=2.0, chord="F"),
        ChordSegment(start=2.0, end=3.0, chord="G"),
    ]

    result = analyze_functions(chords, key)

    assert result[0]["function"] == "I (T)"
    assert result[1]["function"] == "IV (S)"
    assert result[2]["function"] == "V (D)"


def test_analyze_functions_handles_unknown_chord():
    key = "C major"
    chords = [ChordSegment(start=0.0, end=1.0, chord="X")]

    result = analyze_functions(chords, key)

    assert len(result) == 1
    assert result[0]["chord"] == "X"
    assert result[0]["function"] != ""  # should produce something, not crash
