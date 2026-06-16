import numpy as np
import torch
from app.schemas import ChordSegment, ChordAlternative, BeatPoint
from app.utils.viterbi import (
    state_to_label, build_uniform_transition, _log_softmax,
    viterbi_decode, merge_adjacent, top_k_alternatives,
    NUM_STATES, CHORD_QUALITIES,
)


def detect_chords(
    backbone,
    audio: torch.Tensor,
    sr: int,
    beats: list[BeatPoint],
    frame_rate: float,
    min_chord_duration: float = 0.5,
) -> list[ChordSegment]:
    """
    Detect chords using beat-synchronous pooling + Viterbi decoding.

    Args:
        backbone: AnalysisBackbone instance with loaded model
        audio: (1, samples) float32 tensor at target_sr
        sr: sample rate
        beats: beat points from beat_track
        frame_rate: backbone frame rate
        min_chord_duration: minimum chord segment in seconds

    Returns:
        list[ChordSegment] with root/quality/bass, confidence, alternatives
    """
    beat_times = [b.time for b in beats]

    if len(beat_times) < 2:
        return _fallback_chords(backbone, audio, sr, frame_rate)

    # Beat-synchronous pooling
    features = backbone.extract_features(audio, sr)
    pooled = backbone._pool_by_beats(features, beat_times, frame_rate)

    # Run chord head on pooled features
    chord_out = backbone.chord_head(pooled)
    root_logits = chord_out["root_logits"].squeeze(0).cpu().numpy()   # (T_beat, 13)
    quality_logits = chord_out["quality_logits"].squeeze(0).cpu().numpy()  # (T_beat, 22)
    bass_logits = chord_out["bass_logits"].squeeze(0).cpu().numpy()     # (T_beat, 12)

    # Combine root+quality to joint (264) logits
    combined_logits = _combine_logits(root_logits, quality_logits)

    # Viterbi decoding
    tran = build_uniform_transition(prior_stay=0.85)
    tran_log = np.log(tran + 1e-10)
    obs_log = _log_softmax(combined_logits)
    min_frames = max(1, int(min_chord_duration * len(beat_times) / (beat_times[-1] - beat_times[0] + 0.01)))
    path = viterbi_decode(obs_log, tran_log, min_duration_frames=min_frames)

    # Merge adjacent identical states
    segments = merge_adjacent(path)

    # N detection: override low-energy or silence regions
    chord_segments = _build_segments(
        segments, combined_logits, root_logits, quality_logits, bass_logits,
        beats, beat_times, backbone.device,
    )

    return chord_segments


def _combine_logits(root_logits: np.ndarray, quality_logits: np.ndarray) -> np.ndarray:
    """Combine split root/quality logits into joint (264) logits via outer product."""
    root_t = torch.from_numpy(root_logits)
    qual_t = torch.from_numpy(quality_logits)
    root_probs = torch.softmax(root_t, dim=-1)
    qual_probs = torch.softmax(qual_t, dim=-1)
    combined = torch.einsum("ti,tj->tij", root_probs[:, :12], qual_probs)
    return (combined + 1e-10).log().reshape(combined.shape[0], -1).numpy()


def _build_segments(
    raw_segments: list[tuple[int, int, int]],
    combined_logits: np.ndarray,
    root_logits: np.ndarray,
    quality_logits: np.ndarray,
    bass_logits: np.ndarray,
    beats: list[BeatPoint],
    beat_times: list[float],
    device: str,
) -> list[ChordSegment]:
    """Convert Viterbi segments to ChordSegment list with alternatives."""
    T_beat = len(beat_times)
    chord_segments = []

    for seg_idx, (start_beat, end_beat, state) in enumerate(raw_segments):
        if start_beat >= T_beat:
            break
        end_beat = min(end_beat, T_beat - 1)

        label = state_to_label(state)

        # Parse root/quality from label
        if label == "N":
            root, quality, bass = "", "N", None
        else:
            root, quality = label.split(":")
            bass_idx = int(np.argmax(bass_logits[start_beat:end_beat + 1].mean(axis=0)))
            from app.utils.viterbi import PITCH_CLASSES
            bass = PITCH_CLASSES[bass_idx]

        # Confidence from segment-averaged logits
        seg_logits_mean = combined_logits[start_beat:end_beat + 1].mean(axis=0)
        probs = np.exp(_log_softmax(seg_logits_mean))
        confidence = round(float(probs[state]), 4)

        # Top-3 alternatives
        alternatives = top_k_alternatives(combined_logits, (start_beat, end_beat), k=3)

        start_time = beat_times[start_beat]
        end_time = beat_times[end_beat] + 0.05  # small padding

        chord_segments.append(ChordSegment(
            index=seg_idx,
            start=round(start_time, 2),
            end=round(end_time, 2),
            label=label,
            root=root,
            quality=quality,
            bass=bass,
            beat_start=beats[start_beat].beat_index if start_beat < len(beats) else 0,
            beat_end=beats[end_beat].beat_index if end_beat < len(beats) else 0,
            bar=beats[start_beat].bar_index if start_beat < len(beats) else 0,
            confidence=confidence,
            alternatives=[
                ChordAlternative(label=a["label"], confidence=a["confidence"])
                for a in alternatives
            ],
        ))

    return chord_segments


def _fallback_chords(backbone, audio: torch.Tensor, sr: int,
                     frame_rate: float) -> list[ChordSegment]:
    """Fallback: frame-level chord detection when beats unavailable."""
    features = backbone.extract_features(audio, sr)
    chord_out = backbone.chord_head(features)
    root_logits = chord_out["root_logits"].squeeze(0).cpu().numpy()
    quality_logits = chord_out["quality_logits"].squeeze(0).cpu().numpy()
    bass_logits = chord_out["bass_logits"].squeeze(0).cpu().numpy()

    combined = _combine_logits(root_logits, quality_logits)
    probs = np.exp(_log_softmax(combined))
    best_states = np.argmax(probs, axis=-1)

    raw = merge_adjacent(best_states.tolist())
    segments = _build_segments_fallback(
        raw, combined, root_logits, quality_logits, bass_logits, frame_rate,
    )
    return segments


def _build_segments_fallback(
    raw_segments: list[tuple[int, int, int]],
    combined_logits: np.ndarray,
    root_logits: np.ndarray,
    quality_logits: np.ndarray,
    bass_logits: np.ndarray,
    frame_rate: float,
) -> list[ChordSegment]:
    """Build ChordSegments from frame-level fallback without beat info."""
    from app.utils.viterbi import PITCH_CLASSES
    chord_segments = []

    for seg_idx, (start_f, end_f, state) in enumerate(raw_segments):
        label = state_to_label(state)
        if label == "N":
            root, quality, bass = "", "N", None
        else:
            root, quality = label.split(":")
            bass_idx = int(np.argmax(bass_logits[start_f:end_f + 1].mean(axis=0)))
            bass = PITCH_CLASSES[bass_idx]

        seg_logits_mean = combined_logits[start_f:end_f + 1].mean(axis=0)
        probs = np.exp(_log_softmax(seg_logits_mean))
        confidence = round(float(probs[state]), 4)

        alternatives = top_k_alternatives(combined_logits, (start_f, end_f), k=3)

        chord_segments.append(ChordSegment(
            index=seg_idx,
            start=round(start_f / frame_rate, 2),
            end=round((end_f + 1) / frame_rate, 2),
            label=label,
            root=root,
            quality=quality,
            bass=bass,
            beat_start=0,
            beat_end=0,
            bar=0,
            confidence=confidence,
            alternatives=[
                ChordAlternative(label=a["label"], confidence=a["confidence"])
                for a in alternatives
            ],
        ))

    return chord_segments
