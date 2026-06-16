import numpy as np
import torch
from app.schemas import KeyResult, KeySegment, KeyAlternative, ChordSegment

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KEY_LABELS = [
    "C major", "C# major", "D major", "D# major", "E major", "F major",
    "F# major", "G major", "G# major", "A major", "A# major", "B major",
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _normalize_root(root: str) -> str:
    return FLAT_TO_SHARP.get(root, root)


def _correlate(dist: np.ndarray, profile: np.ndarray) -> float:
    d_mean = dist.mean()
    p_mean = profile.mean()
    num = ((dist - d_mean) * (profile - p_mean)).sum()
    den = np.sqrt(((dist - d_mean) ** 2).sum() * ((profile - p_mean) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def _chord_root_distribution(chords: list[ChordSegment]) -> np.ndarray:
    dist = np.zeros(12, dtype=np.float64)
    for ch in chords:
        if ch.root and ch.quality != "N":
            try:
                root_idx = PITCH_CLASSES.index(_normalize_root(ch.root))
                duration = max(ch.end - ch.start, 0.1)
                dist[root_idx] += duration
            except ValueError:
                pass
    total = dist.sum()
    return dist / total if total > 0 else dist


def ks_detect_key_from_dist(pitch_dist: np.ndarray) -> KeyResult:
    best_key = "C major"
    best_corr = -1.0
    alternatives: list[KeyAlternative] = []
    for tonic in range(12):
        rotated_major = np.roll(MAJOR_PROFILE, tonic)
        rotated_minor = np.roll(MINOR_PROFILE, tonic)
        corr_major = _correlate(pitch_dist, rotated_major)
        corr_minor = _correlate(pitch_dist, rotated_minor)
        key_major = f"{PITCH_CLASSES[tonic]} major"
        key_minor = f"{PITCH_CLASSES[tonic]} minor"
        alternatives.append(KeyAlternative(key=key_major, confidence=round(max(0, corr_major), 4)))
        alternatives.append(KeyAlternative(key=key_minor, confidence=round(max(0, corr_minor), 4)))
        if corr_major > best_corr:
            best_corr, best_key = corr_major, key_major
        if corr_minor > best_corr:
            best_corr, best_key = corr_minor, key_minor
    alternatives.sort(key=lambda x: x.confidence, reverse=True)
    return KeyResult(
        key=best_key,
        confidence=round(max(0.0, min(1.0, (best_corr + 1.0) / 2.0)), 4),
        method="ks",
        alternatives=alternatives[:5],
    )


def ks_detect_key(chords: list[ChordSegment]) -> KeyResult:
    if not chords:
        return KeyResult(key="C major", confidence=0.0, method="ks")
    dist = _chord_root_distribution(chords)
    return ks_detect_key_from_dist(dist)


def ssl_detect_key(key_logits: torch.Tensor) -> KeyResult | None:
    if key_logits is None:
        return None
    probs = torch.softmax(key_logits, dim=-1).squeeze().cpu().numpy()
    if probs.ndim == 0:
        return None
    top_indices = np.argsort(probs)[::-1]
    best_idx = top_indices[0]
    alternatives = [
        KeyAlternative(key=KEY_LABELS[int(idx)], confidence=round(float(probs[int(idx)]), 4))
        for idx in top_indices[:5]
    ]
    return KeyResult(
        key=KEY_LABELS[int(best_idx)],
        confidence=round(float(probs[int(best_idx)]), 4),
        method="ssl",
        alternatives=alternatives,
    )


def fuse_key_results(ssl_key: KeyResult | None, ks_key: KeyResult) -> KeyResult:
    if ssl_key is None:
        return ks_key
    if ssl_key.key == ks_key.key:
        conf = round(max(ssl_key.confidence, ks_key.confidence) * 1.05, 4)
        return KeyResult(key=ssl_key.key, confidence=min(1.0, conf), method="fused",
                         alternatives=ssl_key.alternatives[:3])
    else:
        return KeyResult(
            key=ssl_key.key,
            confidence=round(ssl_key.confidence * 0.7, 4),
            method="fused",
            alternatives=[
                ssl_key.alternatives[0],
                KeyAlternative(key=ks_key.key, confidence=ks_key.confidence),
            ],
        )


def detect_key_segments(chords: list[ChordSegment], window_beats: int = 64) -> list[KeySegment]:
    """Sliding window key detection over chord root distribution."""
    if len(chords) < 8:
        return []
    segments = []
    stride = max(window_beats // 2, 1)
    for i in range(0, len(chords), stride):
        window = chords[i:i + window_beats]
        if len(window) < 4:
            continue
        ks = ks_detect_key(window)
        if ks.confidence > 0.4:
            segments.append(KeySegment(
                start=window[0].start,
                end=window[-1].end,
                key=ks.key,
                confidence=ks.confidence,
            ))
    return _merge_adjacent_key_segments(segments)


def _merge_adjacent_key_segments(segments: list[KeySegment]) -> list[KeySegment]:
    if len(segments) < 2:
        return segments
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.key == last.key and seg.start - last.end < 8.0:
            last.end = seg.end
            last.confidence = max(last.confidence, seg.confidence)
        else:
            merged.append(seg)
    return merged
