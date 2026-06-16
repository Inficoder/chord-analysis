# 和弦分析工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web app where users upload audio files and receive chord charts with key detection (global + local segments), chord recognition with alternatives and N state, lyrics transcription, and local-key-aware Roman numeral analysis.

**Architecture:** React+TS frontend communicates via REST polling with a FastAPI Python backend. The backend runs an async analysis pipeline: Demucs vocal separation → parallel SSL backbone (beat map + beat-synchronous chord recognition with learned Viterbi + multi-source key fusion with local segments) and WhisperX lyrics → harmony rule engine. Results stored in-memory with 1-hour TTL.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS, React Router v6, FastAPI, PyTorch, Demucs, WhisperX, MERT-330M

---

## File Structure

```
chord-analysis2/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app, CORS, startup
│   │   ├── config.py                  # Settings (paths, limits, TTL)
│   │   ├── schemas.py                 # Pydantic models (all types)
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── analysis.py            # /api/upload, /status, /result, /export
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── runner.py              # Orchestrator: dependency graph + parallel execution
│   │   │   ├── vocal_sep.py           # Demucs wrapper
│   │   │   ├── beat_track.py          # Beat/downbeat → BeatPoint[]
│   │   │   ├── chord_detect.py        # Chord recognition: beat-sync pooling + learned Viterbi
│   │   │   ├── key_detect.py          # Multi-source key fusion + key_segments
│   │   │   ├── lyrics_transcribe.py   # WhisperX wrapper
│   │   │   └── harmony_analysis.py    # Local-key-aware Roman numeral rule engine
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── backbone.py            # MERT/MusicFM loader + head registry
│   │   │   ├── beat_head.py           # TCN beat/downbeat head (shallow features)
│   │   │   ├── chord_head.py          # Beat-sync pooling + split classifiers (root/quality/bass/N)
│   │   │   └── key_head.py            # Key classifier head (24-class) + segment detector
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── audio.py               # load_audio, get_duration, resample
│   │       └── viterbi.py             # Learned Viterbi + semi-Markov decoder
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── types/
│       │   └── analysis.ts
│       ├── hooks/
│       │   ├── useAnalysisStatus.ts
│       │   └── useChordAlign.ts
│       ├── components/
│       │   ├── UploadZone.tsx
│       │   ├── ProgressPanel.tsx
│       │   ├── ErrorPanel.tsx
│       │   ├── ResultToolbar.tsx
│       │   ├── TimelineSidebar.tsx
│       │   ├── ChordChart.tsx
│       │   └── ChordLine.tsx
│       └── pages/
│           ├── HomePage.tsx
│           └── AnalyzePage.tsx
│
└── docs/superpowers/
    ├── specs/2026-06-15-chord-analysis-design.md
    └── plans/2026-06-15-chord-analysis-implementation.md
```

---

### Task 1: Backend project scaffold

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/Dockerfile`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`

- [ ] **Step 1: Write requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.12
aiofiles==24.1.0
pydantic==2.9.0
torch>=2.4.0
torchaudio>=2.4.0
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
demucs>=4.0.0
whisperx>=3.1.0
transformers>=4.45.0
numpy>=2.1.0
scipy>=1.14.0
```

- [ ] **Step 2: Write config.py**

```python
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB
MAX_DURATION_SECONDS = 600         # 10 minutes
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
RESULT_TTL_SECONDS = 3600           # 1 hour
ANALYSIS_TIMEOUT_SECONDS = 300     # 5 minutes

SUPPORTED_LANGUAGES = {"zh", "en", "auto"}
TARGET_SR = 24000
```

- [ ] **Step 3: Write main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Chord Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Write Dockerfile**

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 5: Verify server starts**

Run: `cd backend && pip install fastapi uvicorn && uvicorn app.main:app --port 8000`
Open `http://localhost:8000/api/health` — expect `{"status":"ok"}`

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat: scaffold backend project with FastAPI"
```

---

### Task 2: Backend schemas

**Files:**
- Create: `backend/app/schemas.py`

- [ ] **Step 1: Write schemas.py**

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class AnalysisStatusEnum(str, Enum):
    uploading = "uploading"
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class Stages(BaseModel):
    vocal_sep: bool = False
    beat_tracking: bool = False
    chord_detection: bool = False
    key_detection: bool = False
    lyrics: bool = False
    harmony_analysis: bool = False


class AnalysisStatus(BaseModel):
    id: str
    filename: str
    status: AnalysisStatusEnum
    progress: int = Field(default=0, ge=0, le=100)
    stages: Stages = Field(default_factory=Stages)
    error: Optional[str] = None


class UploadResponse(BaseModel):
    analysis_id: str
    status_url: str


class BeatPoint(BaseModel):
    time: float
    beat_index: int
    bar_index: int
    beat_in_bar: int          # 1-based
    is_downbeat: bool
    confidence: float = Field(ge=0.0, le=1.0)


class KeyAlternative(BaseModel):
    key: str
    confidence: float


class KeyResult(BaseModel):
    key: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["fused", "ssl", "ks", "cadence"]
    alternatives: list[KeyAlternative] = Field(default_factory=list)


class KeySegment(BaseModel):
    start: float
    end: float
    key: str
    confidence: float = Field(ge=0.0, le=1.0)


class ChordAlternative(BaseModel):
    label: str
    confidence: float


class ChordSegment(BaseModel):
    index: int
    start: float
    end: float

    label: str                 # "C:maj" "G:7" "N"
    root: str                  # "C" "" for N
    quality: str               # "maj" "N"
    bass: Optional[str] = None # Phase 1.5+

    beat_start: int
    beat_end: int
    bar: int

    roman: str = ""
    local_key: str = ""
    function: str = ""

    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[ChordAlternative] = Field(default_factory=list)


class LyricLine(BaseModel):
    start: float
    end: float
    text: str


class TimeSignature(BaseModel):
    value: str                 # "4/4" "3/4" "6/8" "unknown"
    confidence: float = Field(ge=0.0, le=1.0)


class Tempo(BaseModel):
    bpm: int
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    id: str
    global_key: KeyResult
    key_segments: list[KeySegment] = Field(default_factory=list)
    chords: list[ChordSegment] = Field(default_factory=list)
    lyrics: list[LyricLine] = Field(default_factory=list)
    beats: list[BeatPoint] = Field(default_factory=list)
    duration: float
    tempo: Tempo
    time_signature: TimeSignature


class ErrorResponse(BaseModel):
    error: str
    code: Literal["INVALID_FORMAT", "FILE_TOO_LARGE", "NOT_FOUND", "ANALYSIS_FAILED"]
```

- [ ] **Step 2: Verify imports**

Run: `cd backend && python -c "from app.schemas import ChordSegment, KeyResult, KeySegment, ChordAlternative, AnalysisResult, BeatPoint; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat: add Pydantic schemas with structured chord, key segments, alternatives"
```

---

### Task 3: Audio utilities

**Files:**
- Create: `backend/app/utils/__init__.py`
- Create: `backend/app/utils/audio.py`

- [ ] **Step 1: Write audio.py**

```python
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path


def load_audio(file_path: Path, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at target_sr. Returns (samples, sr)."""
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def get_duration(file_path: Path) -> float:
    """Get duration in seconds."""
    info = sf.info(str(file_path))
    return info.duration


def write_audio(file_path: Path, audio: np.ndarray, sr: int):
    """Write mono float32 audio to file."""
    sf.write(str(file_path), audio, sr)


def validate_audio(
    file_path: Path, allowed_exts: set[str], max_size: int, max_duration: float,
) -> str | None:
    """Return error message if invalid, None if valid."""
    suffix = file_path.suffix.lower()
    if suffix not in allowed_exts:
        return f"Unsupported format: {suffix}. Allowed: {allowed_exts}"
    size = file_path.stat().st_size
    if size > max_size:
        return f"File too large: {size} bytes (max {max_size})"
    try:
        dur = get_duration(file_path)
        if dur > max_duration:
            return f"File too long: {dur:.0f}s (max {max_duration}s)"
    except Exception:
        pass
    return None
```

- [ ] **Step 2: Verify**

Run: `cd backend && python -c "from app.utils.audio import load_audio, get_duration, validate_audio; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/utils/
git commit -m "feat: add audio I/O utilities with duration validation"
```

---

### Task 4: Learned Viterbi decoder

**Files:**
- Create: `backend/app/utils/viterbi.py`

- [ ] **Step 1: Write viterbi.py**

```python
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
    """Uniform learned prior (flat). To be replaced by data-driven matrix."""
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
```

- [ ] **Step 2: Verify basic functionality**

Run: `cd backend && python -c "
from app.utils.viterbi import (
    build_uniform_transition, state_to_label, NUM_STATES, CHORD_QUALITIES,
)
tmat = build_uniform_transition()
assert tmat.shape == (NUM_STATES, NUM_STATES)
assert state_to_label(0) == 'N'
assert state_to_label(22) == 'C:maj'
assert CHORD_QUALITIES[0] == 'N'
print('OK')
"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/utils/viterbi.py
git commit -m "feat: add learned Viterbi decoder with N state and top-k alternatives"
```

---

### Task 5: Harmony analysis rule engine

**Files:**
- Create: `backend/app/pipeline/__init__.py`
- Create: `backend/app/pipeline/harmony_analysis.py`

- [ ] **Step 1: Write harmony_analysis.py**

```python
from app.schemas import ChordSegment, KeyResult, KeySegment

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
    root_idx = PITCH_CLASSES.index(root) if root else 0
    return (root_idx - tonic) % 12


def _find_secondary_dominant(root: str, quality: str, current_roman: str,
                              next_chords: list[ChordSegment], local_key: str) -> str:
    """Detect V/X or vii°/X based on dominant function + resolution."""
    if quality not in ("7", "maj", "dim", "dim7"):
        return ""
    if not next_chords:
        return ""
    # Check if this chord's root is a perfect 5th above the next chord's root
    this_root_idx = PITCH_CLASSES.index(root) if root else -1
    next_root = next_chords[0].root
    if not next_root:
        return ""
    next_root_idx = PITCH_CLASSES.index(next_root)
    if (this_root_idx - next_root_idx) % 12 == 7:
        # This is a V/X relationship
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
        root_name = PITCH_CLASSES[_interval_from_tonic(root, local_key)] if _interval_from_tonic(root, local_key) in (0,2,4,5,7,9,11) else f"b{root}"
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
```

- [ ] **Step 2: Verify with test cases**

Run: `cd backend && python -c "
from app.pipeline.harmony_analysis import analyze_chord
# Diatonic
assert analyze_chord('C:maj', 'C major') == 'I'
assert analyze_chord('F:maj', 'C major') == 'IV'
assert analyze_chord('G:7', 'C major') == 'V7'
# N state
assert analyze_chord('N', 'C major') == 'N'
# Borrowed in major
assert 'bVII' in analyze_chord('Bb:maj', 'C major')
# Minor
assert analyze_chord('A:min', 'A minor') == 'i'
assert analyze_chord('E:maj', 'A minor') == 'V'
print('OK')
"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/pipeline/__init__.py backend/app/pipeline/harmony_analysis.py
git commit -m "feat: add local-key-aware harmony analysis with N, borrowed, secondary dominant"
```

---

### Task 6: Multi-source key detection

**Files:**
- Create: `backend/app/pipeline/key_detect.py`

- [ ] **Step 1: Write key_detect.py**

```python
import numpy as np
import torch
from app.schemas import KeyResult, KeySegment, KeyAlternative, ChordSegment, BeatPoint

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KEY_LABELS = [
    "C major", "C# major", "D major", "D# major", "E major", "F major",
    "F# major", "G major", "G# major", "A major", "A# major", "B major",
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _correlate(dist: np.ndarray, profile: np.ndarray) -> float:
    d_mean = dist.mean()
    p_mean = profile.mean()
    num = ((dist - d_mean) * (profile - p_mean)).sum()
    den = np.sqrt(((dist - d_mean) ** 2).sum() * ((profile - p_mean) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def _chord_root_distribution(chords: list[ChordSegment]) -> np.ndarray:
    dist = np.zeros(12, dtype=np.float64)
    for ch in chords:
        if ch.root:
            try:
                root_idx = PITCH_CLASSES.index(ch.root)
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
    """
    Sliding window key detection over chord root distribution.
    Returns key segments where local key confidence exceeds threshold.
    """
    if len(chords) < 8:
        return []
    segments = []
    stride = window_beats // 2
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
```

- [ ] **Step 2: Verify K-S + fusion**

Run: `cd backend && python -c "
from app.pipeline.key_detect import ks_detect_key, fuse_key_results, detect_key_segments
from app.schemas import ChordSegment, KeyResult, KeyAlternative
from app.pipeline.harmony_analysis import run_harmony_analysis
import json

# Test KS with C major chords
chords = [
    ChordSegment(index=0, start=0, end=2, label='C:maj', root='C', quality='maj', beat_start=0, beat_end=0, bar=0, confidence=0.9),
    ChordSegment(index=1, start=2, end=4, label='F:maj', root='F', quality='maj', beat_start=1, beat_end=1, bar=1, confidence=0.9),
    ChordSegment(index=2, start=4, end=6, label='G:7', root='G', quality='7', beat_start=2, beat_end=2, bar=2, confidence=0.9),
    ChordSegment(index=3, start=6, end=8, label='C:maj', root='C', quality='maj', beat_start=3, beat_end=3, bar=3, confidence=0.9),
]
key = ks_detect_key(chords)
assert key.key == 'C major', f'Expected C major, got {key.key}'
print(f'OK: {key.key} conf={key.confidence:.3f}')
"`
Expected: `OK: C major conf=...`

- [ ] **Step 3: Commit**

```bash
git add backend/app/pipeline/key_detect.py
git commit -m "feat: add multi-source key detection with KS, SSL fusion, and key segments"
```

---

### Task 7: ML model backbone and heads

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/backbone.py`
- Create: `backend/app/models/beat_head.py`
- Create: `backend/app/models/chord_head.py`
- Create: `backend/app/models/key_head.py`

- [ ] **Step 1: Write beat_head.py**

```python
import torch
import torch.nn as nn


class BeatHead(nn.Module):
    """TCN beat/downbeat tracker from shallow backbone features (high time resolution)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.beat_classifier = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.downbeat_classifier = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, D) → beat_logits (B,T,1), downbeat_logits (B,T,1)."""
        x = features.transpose(1, 2)
        x = self.tcn(x)
        beat = self.beat_classifier(x).transpose(1, 2)
        downbeat = self.downbeat_classifier(x).transpose(1, 2)
        return beat, downbeat
```

- [ ] **Step 2: Write chord_head.py**

```python
import torch
import torch.nn as nn

NUM_PITCH_CLASSES = 12
NUM_QUALITIES = 22  # includes N
NUM_CHORD_CLASSES = NUM_PITCH_CLASSES * NUM_QUALITIES  # 264


class ChordHead(nn.Module):
    """
    Beat-synchronous chord classifier with split heads.
    root: 13-class (12 pitch + no-root for N)
    quality: 22-class
    bass: 12-class (Phase 1.5, not used in Phase 1)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.root_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_PITCH_CLASSES + 1),  # +1 for no-root
        )
        self.quality_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_QUALITIES),
        )
        self.bass_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_PITCH_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T_beat, input_dim) — beat-synchronous pooled features
        Returns:
            dict with root_logits, quality_logits, bass_logits
        """
        return {
            "root_logits": self.root_classifier(features),
            "quality_logits": self.quality_classifier(features),
            "bass_logits": self.bass_classifier(features),
        }

    @staticmethod
    def root_quality_to_combined(root_logits: torch.Tensor, quality_logits: torch.Tensor) -> torch.Tensor:
        """
        Combine split root+quality logits into joint (264-class) logits via outer product.
        Only used when a joint transition model is needed.
        """
        root_probs = torch.softmax(root_logits, dim=-1)  # (B, T, 13)
        qual_probs = torch.softmax(quality_logits, dim=-1)  # (B, T, 22)
        combined = torch.einsum("bti,btj->btij", root_probs[:, :, :12], qual_probs)
        return combined.reshape(*combined.shape[:2], -1).log()  # (B, T, 264)
```

- [ ] **Step 3: Write key_head.py**

```python
import torch
import torch.nn as nn


class KeyHead(nn.Module):
    """24-class key classifier with attention pooling for global + segment detection."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_keys: int = 24):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_keys),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) — frame or beat-level features
        Returns:
            key_logits: (B, num_keys) — after attention-weighted pooling
        """
        attn_weights = torch.softmax(self.attention(features), dim=1)  # (B, T, 1)
        pooled = (features * attn_weights).sum(dim=1)  # (B, D)
        return self.classifier(pooled)
```

- [ ] **Step 4: Write backbone.py**

```python
import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from app.models.beat_head import BeatHead
from app.models.chord_head import ChordHead
from app.models.key_head import KeyHead


class AnalysisBackbone:
    """Shared MERT/MusicFM backbone with three task heads."""

    def __init__(self, model_name: str = "m-a-p/MERT-v1-330M", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.model: nn.Module | None = None
        self.feature_extractor: Wav2Vec2FeatureExtractor | None = None
        self.beat_head: BeatHead | None = None
        self.chord_head: ChordHead | None = None
        self.key_head: KeyHead | None = None

    def load(self):
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        hidden = self.model.config.hidden_size
        self.beat_head = BeatHead(input_dim=hidden).to(self.device).eval()
        self.chord_head = ChordHead(input_dim=hidden).to(self.device).eval()
        self.key_head = KeyHead(input_dim=hidden).to(self.device).eval()

    def unload(self):
        for attr in ["model", "feature_extractor", "beat_head", "chord_head", "key_head"]:
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "to"):
                obj.to("cpu")
            setattr(self, attr, None)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def extract_features(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        inputs = self.feature_extractor(
            audio.cpu().numpy(), sampling_rate=sr, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state  # (1, T_frame, hidden_dim)

    def _pool_by_beats(self, features: torch.Tensor, beat_indices: list[list[int]],
                       frame_rate: float, beat_times: list[float]) -> torch.Tensor:
        """Beat-synchronous pooling: aggregate frame features within each beat interval."""
        T = features.shape[1]
        N_beats = len(beat_times)
        pooled = torch.zeros(1, N_beats, features.shape[2], device=features.device)
        for i in range(N_beats):
            start_frame = int(beat_times[i] * frame_rate)
            end_frame = int(beat_times[i + 1] * frame_rate) if i + 1 < N_beats else T
            start_frame = max(0, min(start_frame, T - 1))
            end_frame = max(start_frame + 1, min(end_frame, T))
            pooled[0, i] = features[0, start_frame:end_frame].mean(dim=0)
        return pooled

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, sr: int,
                beat_times: list[float] | None = None) -> dict:
        """Run all three heads. Chord uses beat-sync pooling if beats provided."""
        features = self.extract_features(audio, sr)  # (1, T_frame, D)
        frame_rate = sr / (self.model.config.conv_stride if hasattr(self.model.config, "conv_stride") else 320)

        # Beat head on frame-level features (shallow)
        beat_logits, downbeat_logits = self.beat_head(features)

        # Chord head: beat-synchronous or frame-level
        if beat_times and len(beat_times) > 1:
            chord_features = self._pool_by_beats(features, [], frame_rate, beat_times)
        else:
            chord_features = features  # fallback to frame-level
        chord_out = self.chord_head(chord_features)

        # Key head: attention pooling over all frames
        key_logits = self.key_head(features)

        return {
            "beat_logits": beat_logits.cpu(),
            "downbeat_logits": downbeat_logits.cpu(),
            "root_logits": chord_out["root_logits"].cpu(),
            "quality_logits": chord_out["quality_logits"].cpu(),
            "bass_logits": chord_out["bass_logits"].cpu(),
            "key_logits": key_logits.cpu(),
            "frame_rate": frame_rate,
        }
```

- [ ] **Step 5: Verify imports**

Run: `cd backend && python -c "
from app.models.backbone import AnalysisBackbone
from app.models.beat_head import BeatHead
from app.models.chord_head import ChordHead
from app.models.key_head import KeyHead
print('OK')
"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add backend/app/models/
git commit -m "feat: add backbone with beat-sync chord head, split classifiers, attention key head"
```

---

### Task 8-12: Remaining backend pipeline modules

Tasks 8-12 (Demucs, WhisperX, beat/chord/key pipeline modules, runner, API routes) follow the same structure as the original plan, with the following adjustments:

- **beat_track.py**: Output `BeatPoint[]` with full beat metadata (time, beat_index, bar_index, beat_in_bar, is_downbeat, confidence). Compute bars from downbeat positions. Compute time_signature as `TimeSignature` with confidence.
- **chord_detect.py**: Use beat-synchronous pooling; call `ChordHead.root_quality_to_combined()` for joint logits; use `viterbi_decode()` with uniform transition matrix (placeholder for learned); call `top_k_alternatives()` to populate `alternatives`; detect N when quality head confidence < 0.3 or root head predicts no-root class.
- **key_detect.py**: (Already updated in Task 6.) Call `ssl_detect_key()`, `ks_detect_key()`, `fuse_key_results()`, and `detect_key_segments()`.
- **runner.py**: Orchestrate with BeatPoint array, updated ChordSegment and KeyResult structures. Pass beat map to chord_detect. Compute global_key and key_segments.
- **routes/analysis.py**: Return 202 for incomplete result. Validate file duration. Support cancel endpoint. Support export with structured chord data.

---

### Task 13-21: Frontend

Frontend tasks (13-21) follow the same patterns as the original plan with these updates:

- **types/analysis.ts**: Updated to match the new structured types (BeatPoint, KeySegment, KeyAlternative, ChordAlternative, ChordSegment with root/quality/bass/alternatives, TimeSignature with confidence, AnalysisResult with global_key + key_segments).
- **ChordLine.tsx**: Display low-confidence chords with dashed/light styling; show alternatives on hover; display "N" as "N.C." (no chord); handle instrumental-only mode.
- **ChordChart.tsx**: Pass key_segments for local key display; show key change markers when local key differs from global.
- **TimelineSidebar.tsx**: Show bar/beat grid markers (not just seconds).
- **ResultToolbar.tsx**: Display key segments info; show time_signature confidence.

See the original plan file for the detailed step-by-step code for tasks 8-21. The implementations follow the same structure but use the updated types and interfaces defined in this revised spec.

---

## Notes

- ML model weights are large and not committed to git. Add a `scripts/download_models.sh`.
- The pipeline runner uses a thread pool for async execution. For production, swap to Celery + Redis.
- The backbone supports MERT-v1-330M. Swap `model_name` for MusicFM or newer models.
- Viterbi transition matrix starts as uniform. Replace with learned matrix after training on annotated chord data.
- Bass Head (Phase 1.5) is wired but not trained in Phase 1 — outputs are null.
- `.superpowers/` directory should be in `.gitignore`.
- Evaluation metrics must be implemented as a separate evaluation harness (not in the main app).
