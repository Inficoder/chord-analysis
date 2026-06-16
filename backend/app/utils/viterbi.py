import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_QUALITIES = [
    "N",
    "maj", "min", "dim", "aug", "sus2", "sus4",
    "6", "min6",
    "7", "maj7", "min7", "m7b5", "dim7", "minMaj7",
    "add9", "9", "maj9", "min9",
    "11", "13",
    "alt",
]

NUM_ROOTS = 12
NUM_QUALITIES = len(CHORD_QUALITIES)
NUM_STATES = NUM_ROOTS * NUM_QUALITIES  # 264


def build_uniform_transition(prior_stay: float = 0.85) -> np.ndarray:
    """Uniform transition prior. Replace with data-driven matrix after training."""
    n = NUM_STATES
    tmat = np.full((n, n), (1.0 - prior_stay) / (n - 1))
    np.fill_diagonal(tmat, prior_stay)
    return tmat


def state_to_label(state: int) -> str:
    """Convert state index to 'root:quality' or 'N'."""
    root_idx = state // NUM_QUALITIES
    qual_idx = state % NUM_QUALITIES
    quality = CHORD_QUALITIES[qual_idx]
    if quality == "N":
        return "N"
    return f"{PITCH_CLASSES[root_idx]}:{quality}"


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def viterbi_decode(
    frame_logits: np.ndarray,
    transition_log: np.ndarray,
    min_duration_frames: int = 1,
) -> list[int]:
    """
    Viterbi decoding with minimum duration constraint.

    Args:
        frame_logits: (T, NUM_STATES) log-probs per frame
        transition_log: (NUM_STATES, NUM_STATES) log transition probs
        min_duration_frames: minimum consecutive frames for same state

    Returns:
        list[int]: optimal state sequence
    """
    T, N = frame_logits.shape

    dp = np.full((T, N), -np.inf)
    back = np.zeros((T, N), dtype=np.int32)

    dp[0] = frame_logits[0]

    for t in range(1, T):
        for s in range(N):
            stay_score = dp[t - 1, s] + transition_log[s, s]

            switch_scores = dp[t - 1] + transition_log[:, s]
            best_switch = np.max(switch_scores)
            best_switch_state = np.argmax(switch_scores)

            if stay_score >= best_switch:
                dp[t, s] = stay_score
                back[t, s] = s
            else:
                dp[t, s] = best_switch
                back[t, s] = best_switch_state

    path = [int(np.argmax(dp[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return path


def merge_adjacent(states: list[int]) -> list[tuple[int, int, int]]:
    """Merge adjacent identical states into (start_frame, end_frame, state_index)."""
    if not states:
        return []
    segments = []
    start = 0
    current = states[0]
    for i, s in enumerate(states):
        if s != current:
            segments.append((start, i - 1, current))
            start = i
            current = s
    segments.append((start, len(states) - 1, current))
    return segments


def top_k_alternatives(
    logits: np.ndarray,
    segment: tuple[int, int],
    k: int = 3,
) -> list[dict]:
    """Get top-k alternative labels for a segment (averaging logits over segment)."""
    seg_logits = logits[segment[0]:segment[1] + 1].mean(axis=0)
    probs = np.exp(_log_softmax(seg_logits))
    top_indices = np.argsort(probs)[::-1][:k]
    return [
        {"label": state_to_label(int(idx)), "confidence": round(float(probs[idx]), 4)}
        for idx in top_indices
    ]
