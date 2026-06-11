import numpy as np
import librosa

CQT_HOP_LENGTH = 1024

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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

    major_corr = [np.corrcoef(np.roll(chroma_mean, i), major_profile)[0, 1]
                  for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(chroma_mean, i), minor_profile)[0, 1]
                  for i in range(12)]

    best_major = int(np.argmax(major_corr))
    best_minor = int(np.argmax(minor_corr))

    if max(major_corr) >= max(minor_corr):
        confidence = _normalize_corr(max(major_corr))
        return f"{PITCH_NAMES[best_major]} major", round(confidence, 3)
    else:
        confidence = _normalize_corr(max(minor_corr))
        return f"{PITCH_NAMES[best_minor]} minor", round(confidence, 3)


def _normalize_corr(corr: float) -> float:
    """Map raw correlation to 0-1 range."""
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))


CHORD_TYPES = ["", "m", "7", "m7", "maj7", "dim", "aug", "sus4", "sus2"]

CHORD_TEMPLATES: dict[str, np.ndarray] = {}


def _chord_template(chord_type: str) -> np.ndarray:
    """Build a 12-bin chroma template for a chord type, rooted at C."""
    if chord_type in CHORD_TEMPLATES:
        return CHORD_TEMPLATES[chord_type]

    template = np.zeros(12)
    intervals = {
        "": [0, 4, 7],           # major
        "m": [0, 3, 7],          # minor
        "7": [0, 4, 7, 10],      # dominant 7
        "m7": [0, 3, 7, 10],     # minor 7
        "maj7": [0, 4, 7, 11],   # major 7
        "dim": [0, 3, 6],        # diminished
        "aug": [0, 4, 8],        # augmented
        "sus4": [0, 5, 7],       # sus4
        "sus2": [0, 2, 7],       # sus2
    }
    for interval in intervals.get(chord_type, [0, 4, 7]):
        template[interval % 12] = 1.0
    CHORD_TEMPLATES[chord_type] = template
    return template


class ChordSegment:
    def __init__(self, start: float, end: float, chord: str):
        self.start = start
        self.end = end
        self.chord = chord


def recognize_chords(audio: np.ndarray, sr: int) -> list[ChordSegment]:
    """
    Recognize chords using madmom Chordino if available, falling back to
    chroma template matching. Returns list of ChordSegment.
    """
    try:
        from madmom.features.chords import CNNChordFeatureProcessor
        proc = CNNChordFeatureProcessor(fps=2)
        chord_probs = proc(audio)
        return _process_chordino_output(chord_probs)
    except (ImportError, OSError):
        return _recognize_chords_librosa(audio, sr)


def _process_chordino_output(chord_probs: np.ndarray) -> list[ChordSegment]:
    """Convert Chordino probability output to chord segments."""
    segments = []
    hop = 0.5  # FPS=2
    for i in range(len(chord_probs)):
        frame = chord_probs[i]
        chord_idx = int(np.argmax(frame))
        chord_name = _chordino_idx_to_name(chord_idx)
        segments.append(ChordSegment(
            start=round(i * hop, 2),
            end=round((i + 1) * hop, 2),
            chord=chord_name,
        ))
    return _merge_consecutive_same_chords(segments)


def _chordino_idx_to_name(idx: int) -> str:
    """Map Chordino output index (0-170) to chord name."""
    root = PITCH_NAMES[idx % 12]
    quality_idx = idx // 12
    qualities = [
        "maj", "min", "maj7", "min7", "7", "dim", "aug",
        "maj6", "min6", "sus4", "sus2", "dim7", "hdim7", "maj9", "min9",
    ]
    if quality_idx < len(qualities):
        q = qualities[quality_idx]
        if q == "maj":
            return root
        elif q == "min":
            return f"{root}m"
        else:
            return f"{root}{q}"
    return PITCH_NAMES[idx % 12]


def _recognize_chords_librosa(audio: np.ndarray, sr: int) -> list[ChordSegment]:
    """Fallback chord recognition using chroma template matching."""
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=CQT_HOP_LENGTH
    )
    fps = sr / CQT_HOP_LENGTH
    hop_sec = 1.0 / fps

    segments = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        best_chord, _ = _match_frame_to_template(frame)
        t = round(i * hop_sec, 2)
        segments.append(ChordSegment(
            start=t,
            end=round(t + hop_sec, 2),
            chord=best_chord,
        ))
    return _merge_consecutive_same_chords(segments)


def _match_frame_to_template(frame: np.ndarray) -> tuple[str, float]:
    """Find best matching chord for a chroma frame. Returns (chord_name, score)."""
    best_score = -1
    best_chord = "N"
    for root_idx in range(12):
        for chord_type in CHORD_TYPES:
            template = np.roll(_chord_template(chord_type), root_idx)
            score = np.dot(frame, template) / (np.linalg.norm(frame) * np.linalg.norm(template) + 1e-10)
            if score > best_score:
                best_score = score
                quality_str = f"{chord_type}" if chord_type else ""
                best_chord = f"{PITCH_NAMES[root_idx]}{quality_str}"
    return best_chord, best_score


def _merge_consecutive_same_chords(segments: list[ChordSegment]) -> list[ChordSegment]:
    """Merge adjacent segments with the same chord label."""
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.chord == merged[-1].chord:
            merged[-1].end = seg.end
        else:
            merged.append(seg)
    return merged
