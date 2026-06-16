import numpy as np
import torch
from scipy.signal import find_peaks
from collections import Counter
from app.schemas import BeatPoint, Tempo, TimeSignature


def detect_beats(
    beat_logits: torch.Tensor,
    downbeat_logits: torch.Tensor,
    frame_rate: float,
) -> tuple[list[BeatPoint], Tempo, TimeSignature]:
    """
    Detect beats and downbeats from backbone output logits.

    Args:
        beat_logits: (1, T, 1) frame-level beat activations
        downbeat_logits: (1, T, 1) frame-level downbeat activations
        frame_rate: frames per second (sr / hop_length)

    Returns:
        (beats, tempo, time_signature)
    """
    beat_act = torch.sigmoid(beat_logits).squeeze().cpu().numpy()
    downbeat_act = torch.sigmoid(downbeat_logits).squeeze().cpu().numpy()

    peaks, props = find_peaks(beat_act, height=0.3, distance=max(1, int(frame_rate * 0.2)))
    if len(peaks) < 2:
        return [], Tempo(bpm=120, confidence=0.0), TimeSignature(value="4/4", confidence=0.0)

    heights = props["peak_heights"]
    beat_times = [round(float(p) / frame_rate, 3) for p in peaks]
    beat_confidences = [round(float(min(h, 1.0)), 4) for h in heights]

    # Tempo from median inter-beat interval
    ibis = np.diff(beat_times)
    median_ibi = float(np.median(ibis))
    bpm = round(60.0 / median_ibi) if median_ibi > 0 else 120
    bpm = max(40, min(240, bpm))
    tempo_conf = round(min(1.0, 1.0 - (np.std(ibis) / (median_ibi + 0.001))), 4)

    # Downbeat detection: find downbeat peaks near beat positions
    downbeat_peaks, db_props = find_peaks(
        downbeat_act, height=0.35, distance=max(1, int(frame_rate * 0.4))
    )
    downbeat_frames = set(downbeat_peaks)

    # Time signature: find dominant bar length in beats
    db_indices = []
    for i, p in enumerate(peaks):
        # Check if this beat has a downbeat within ±1 frame
        nearby = [db for db in downbeat_frames if abs(db - p) <= 2]
        if nearby:
            db_indices.append(i)

    if len(db_indices) >= 2:
        bar_lengths = np.diff(db_indices)
        # Round to common time signatures: 2,3,4,5,6,7
        common = [2, 3, 4, 5, 6]
        rounded = []
        for bl in bar_lengths:
            best = min(common, key=lambda c: abs(c - bl))
            if abs(best - bl) <= 1:
                rounded.append(best)
        if rounded:
            beats_per_bar = Counter(rounded).most_common(1)[0][0]
        else:
            beats_per_bar = 4
        ts_conf = round(len(rounded) / len(bar_lengths), 4) if bar_lengths else 0.0
    else:
        beats_per_bar = 4
        ts_conf = 0.0

    ts = TimeSignature(value=f"{beats_per_bar}/4", confidence=ts_conf)

    # Build BeatPoint list with bar assignments
    beats = []
    bar_idx = 0
    last_db_idx = -beats_per_bar
    current_db_indices = set(db_indices)

    for i, (t, conf) in enumerate(zip(beat_times, beat_confidences)):
        is_db = i in current_db_indices

        if is_db:
            # Check if this starts a new bar (not an internal downbeat accent)
            if i - last_db_idx >= beats_per_bar - 1:
                bar_idx += 1
                last_db_idx = i

        beat_in_bar = (i - last_db_idx) % beats_per_bar
        if not is_db and i - last_db_idx >= beats_per_bar:
            bar_idx += 1
            last_db_idx = i
            beat_in_bar = 0

        beats.append(BeatPoint(
            time=t,
            beat_index=i,
            bar_index=bar_idx,
            beat_in_bar=beat_in_bar + 1,
            is_downbeat=is_db or beat_in_bar == 0,
            confidence=conf,
        ))

    return beats, Tempo(bpm=bpm, confidence=tempo_conf), ts
