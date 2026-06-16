from app.schemas import ChordSegment, KeyResult, KeySegment

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _normalize_root(root: str) -> str:
    """Convert flat note names to sharp equivalents."""
    return FLAT_TO_SHARP.get(root, root)

MAJOR_SCALE = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
MINOR_SCALE = {0: "i", 2: "ii°", 3: "bIII", 5: "iv", 7: "v", 8: "bVI", 10: "bVII"}

BORROWED_MAJOR = {(3, "maj"): "bIII", (5, "min"): "iv", (8, "maj"): "bVI", (10, "maj"): "bVII"}
BORROWED_MINOR = {(5, "maj"): "IV", (7, "maj"): "V", (11, "dim"): "#vii°"}

QUALITY_SUFFIX = {
    "7": "7", "maj7": "maj7", "min7": "min7", "m7b5": "m7b5", "dim7": "dim7",
    "minMaj7": "minMaj7", "9": "9", "maj9": "maj9", "min9": "min9",
    "add9": "add9", "11": "11", "13": "13", "alt": "alt",
    "6": "6", "min6": "min6",
}

TONIC_MAP = {
    "C major": 0, "C# major": 1, "D major": 2, "D# major": 3,
    "E major": 4, "F major": 5, "F# major": 6, "G major": 7,
    "G# major": 8, "A major": 9, "A# major": 10, "B major": 11,
    "C minor": 0, "C# minor": 1, "D minor": 2, "D# minor": 3,
    "E minor": 4, "F minor": 5, "F# minor": 6, "G minor": 7,
    "G# minor": 8, "A minor": 9, "A# minor": 10, "B minor": 11,
}

FUNCTION_MAP = {
    "I": "tonic", "i": "tonic", "IV": "predominant", "iv": "predominant",
    "ii": "predominant", "ii°": "predominant", "V": "dominant",
    "V7": "dominant", "vii°": "dominant", "vi": "tonic",
    "bVII": "borrowed", "bIII": "borrowed", "bVI": "borrowed",
}


def _parse_chord(label: str) -> tuple[str, str]:
    """'C:maj' -> ('C', 'maj'), 'N' -> ('', 'N')."""
    if label == "N" or ":" not in label:
        return ("", "N")
    root_str, quality = label.split(":")
    return (root_str, quality)


def _get_local_key(chord_start: float, key_segments: list[KeySegment], fallback: str) -> str:
    """Find the active local key for a given time position."""
    for seg in key_segments:
        if seg.start <= chord_start < seg.end:
            return seg.key
    return fallback


def _is_minor(key: str) -> bool:
    return "minor" in key


def _interval_from_tonic(root: str, key: str) -> int:
    tonic = TONIC_MAP.get(key, 0)
    root_normalized = _normalize_root(root)
    root_idx = PITCH_CLASSES.index(root_normalized) if root_normalized else 0
    return (root_idx - tonic) % 12


def _find_secondary_dominant(root: str, quality: str, current_roman: str,
                              next_chords: list[ChordSegment], local_key: str) -> str:
    """Detect V/X or vii°/X based on dominant function + resolution."""
    if quality not in ("7", "maj", "dim", "dim7"):
        return ""
    if not next_chords:
        return ""
    this_root_idx = PITCH_CLASSES.index(_normalize_root(root)) if root else -1
    next_root = next_chords[0].root
    if not next_root:
        return ""
    next_root_idx = PITCH_CLASSES.index(_normalize_root(next_root))
    if (this_root_idx - next_root_idx) % 12 == 7:
        target_roman = next_chords[0].roman or _basic_diatonic(next_root, local_key)
        return f"V/{target_roman}" if quality == "7" else f"V/{target_roman}"
    return ""


def _basic_diatonic(root: str, key: str) -> str:
    interval = _interval_from_tonic(root, key)
    scale = MINOR_SCALE if _is_minor(key) else MAJOR_SCALE
    return scale.get(interval, "")


def analyze_chord(label: str, local_key: str, next_chords: list[ChordSegment] | None = None) -> str:
    """
    Convert a chord label to Roman numeral given the local key.
    Handles: diatonic, borrowed, secondary dominant, chromatic fallback.
    """
    if label == "N":
        return "N"

    root, quality = _parse_chord(label)
    if not root:
        return "?"

    interval = _interval_from_tonic(root, local_key)
    is_min = _is_minor(local_key)

    # 1. Diatonic
    diatonic = _basic_diatonic(root, local_key)
    # 2. Borrowed
    borrowed = None
    if not diatonic:
        borrowed = BORROWED_MAJOR.get((interval, quality)) if not is_min else None
        if not borrowed and is_min:
            borrowed = BORROWED_MINOR.get((interval, quality))
    # 3. Secondary dominant
    secondary = ""
    if next_chords:
        secondary = _find_secondary_dominant(root, quality, diatonic or borrowed or "", next_chords, local_key)
    # 4. Fallback: chromatic
    base = diatonic or borrowed or secondary
    if not base:
        tonic = TONIC_MAP.get(local_key, 0)
        root_idx_abs = PITCH_CLASSES.index(_normalize_root(root))
        rel = (root_idx_abs - tonic) % 12
        root_name = PITCH_CLASSES[rel]
        base = root_name

    # Append quality suffix for non-triad chords
    suffix = QUALITY_SUFFIX.get(quality, "")
    clean = base.rstrip("°")
    return f"{clean}{suffix}" if suffix else base


def run_harmony_analysis(chords: list[ChordSegment], key_segments: list[KeySegment],
                          global_key: str) -> list[ChordSegment]:
    """Annotate chord segments with Roman numerals using local key context."""
    result = []
    for i, ch in enumerate(chords):
        local_key = _get_local_key(ch.start, key_segments, global_key)
        next_chords = chords[i + 1:i + 3] if i + 1 < len(chords) else []
        ch.roman = analyze_chord(ch.label, local_key, next_chords)
        ch.local_key = local_key
        ch.function = FUNCTION_MAP.get(ch.roman.strip("7majin°øb#"), "")
        result.append(ch)
    return result
