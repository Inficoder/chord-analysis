from pathlib import Path

from app.audio import load_audio
from app.analysis import detect_key, recognize_chords, analyze_functions
from app.models import AnalysisResult, ChordEvent


def run_pipeline(audio_path: Path) -> AnalysisResult:
    """Run the full analysis pipeline on an audio file."""
    audio, info = load_audio(audio_path)
    sr = info.sample_rate

    # Stage 1: Key detection
    key, confidence = detect_key(audio, sr)

    # Stage 2: Chord recognition
    chord_segments = recognize_chords(audio, sr)

    # Stage 3: Harmonic function analysis
    annotated = analyze_functions(chord_segments, key)

    chords = [
        ChordEvent(
            start=item["start"],
            end=item["end"],
            chord=item["chord"],
            function=item["function"],
        )
        for item in annotated
    ]

    return AnalysisResult(
        key=key,
        key_confidence=confidence,
        chords=chords,
    )
