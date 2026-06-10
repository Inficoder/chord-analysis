# Chord Analysis Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web application that analyzes audio files to detect chords, key, and harmonic function, with REST API for third-party integration.

**Architecture:** FastAPI backend with Celery async workers for the analysis pipeline (key detection → chord recognition → function analysis). React SPA frontend communicates via REST + SSE. Audio processing uses librosa/madmom/music21.

**Tech Stack:** Python 3.12+, FastAPI, Celery (Redis), librosa, madmom, music21 / React 18+, TypeScript, Vite, Tailwind CSS

---

### Task 1: Project Scaffold and Configuration

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create .gitignore**

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/
.env
*.tmp
node_modules/
.DS_Store
uploads/
results/
backend/.venv/
frontend/dist/
*.log
EOF
```

- [ ] **Step 2: Create README.md**

```bash
cat > README.md << 'EOF'
# Chord Analysis

Web application for analyzing chord progressions in audio files.

## Quick Start

```bash
docker-compose up
```

Open http://localhost:5173

## API Endpoints

See CLAUDE.md for development commands.
EOF
```

- [ ] **Step 3: Create CLAUDE.md**

```bash
cat > CLAUDE.md << 'EOF'
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend
```bash
cd backend
python -m venv .venv && source .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Run dev server
uvicorn app.main:app --reload --port 8000

# Run all tests
pytest tests/ -v

# Run single test
pytest tests/test_analysis.py::test_detect_key -v

# Celery worker
celery -A app.tasks worker --loglevel=info --pool=solo
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # production build
```

### Docker
```bash
docker-compose up          # start all services
docker-compose up --build  # rebuild and start
docker-compose down        # stop all
```

## Architecture

```
Frontend (React SPA on :5173) ──REST/SSE──▶ FastAPI (:8000) ──Redis──▶ Celery Worker
                                                                              │
                                                         ┌────────────────────┘
                                                         ▼
                                          Analysis Pipeline (librosa/madmom/music21)
```

### Backend Structure (`backend/app/`)

| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI app, CORS, route mounting |
| `config.py` | Pydantic Settings (paths, limits, Redis URL) |
| `models.py` | Pydantic models: Task, ChordResult, AnalysisResult |
| `api.py` | REST endpoints (upload, analyze, task status, SSE, result) |
| `storage.py` | File I/O — save/load audio, save/load results JSON |
| `tasks.py` | Celery app init + `run_analysis` task definition |
| `celery_config.py` | Celery worker configuration (Redis broker URL, serialization) |
| `pipeline.py` | Orchestrator: calls audio → key → chords → functions in sequence |
| `audio.py` | Audio loading, format conversion, mono downmix, 22050Hz resample |
| `analysis.py` | Key detection (madmom CNN + librosa fallback), chord recognition (Chordino), function analysis (music21 + T/S/D rules) |

### Frontend Structure (`frontend/src/`)

| File | Responsibility |
|------|---------------|
| `App.tsx` | Top-level: state management, SSE subscription, layout |
| `types.ts` | TypeScript types: Task, ChordEvent, AnalysisResult |
| `services/api.ts` | All API calls (upload, analyze, getTask, SSE stream) |
| `components/UploadArea.tsx` | Drag-and-drop upload, file info, trigger analyze |
| `components/ProgressIndicator.tsx` | Progress bar + stage text, SSE-driven |
| `components/ChordTimeline.tsx` | Table of chords with time, name, function; JSON export |

### Analysis Pipeline Flow
```
Raw audio → audio.py (WAV 22050Hz mono) → analysis.detect_key()
                                         → analysis.recognize_chords()
                                         → analysis.analyze_functions()
                                         → pipeline.py assembles final JSON
```

### Key Dependencies
- **librosa**: audio loading, chroma features, K-S key detection fallback
- **madmom**: CNNKeyRecognitionProcessor (key), CNNChordFeatureProcessor/Chordino (chords)
- **music21**: RomanNumeral chord analysis for function labeling
- **soundfile / pydub**: audio format conversion for non-WAV formats
- **Celery**: async task processing via Redis broker
EOF
```

- [ ] **Step 4: Create docker-compose.yml**

```yaml
version: "3.8"
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - REDIS_URL=redis://redis:6379/0
      - UPLOAD_DIR=/data/uploads
      - RESULT_DIR=/data/results
    volumes:
      - ./data:/data
      - ./backend/app:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    depends_on: [redis]

  worker:
    build: ./backend
    environment:
      - REDIS_URL=redis://redis:6379/0
      - UPLOAD_DIR=/data/uploads
      - RESULT_DIR=/data/results
    volumes:
      - ./data:/data
      - ./backend/app:/app
    command: celery -A app.tasks worker --loglevel=info --pool=solo
    depends_on: [redis]

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    volumes:
      - ./frontend/src:/app/src
    command: npm run dev -- --host
    depends_on: [backend]
```

- [ ] **Step 5: Create backend directory**

```bash
mkdir -p backend/app backend/tests
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore README.md CLAUDE.md docker-compose.yml backend/
git commit -m "chore: project scaffold with config and docs"
```

---

### Task 2: Backend Dependencies, Config, and Models

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/models.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Write backend/requirements.txt**

```
fastapi==0.115.*
uvicorn[standard]==0.34.*
python-multipart==0.0.*
celery[redis]==5.4.*
redis==5.2.*
librosa==0.10.*
soundfile==0.12.*
music21==9.*
pydantic-settings==2.*
madmom==0.16.*
pytest==8.*
httpx==0.28.*
pytest-asyncio==0.25.*
```

- [ ] **Step 2: Write backend/app/__init__.py** (empty file)

```bash
touch backend/app/__init__.py
```

- [ ] **Step 3: Write backend/app/config.py**

```python
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    upload_dir: Path = Path("../data/uploads")
    result_dir: Path = Path("../data/results")
    redis_url: str = "redis://localhost:6379/0"
    max_upload_size_mb: int = 50
    max_duration_seconds: int = 900  # 15 minutes

    model_config = {"env_prefix": ""}


settings = Settings()
```

- [ ] **Step 4: Write backend/app/models.py**

```python
from enum import Enum
from uuid import UUID
from pydantic import BaseModel


class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ChordEvent(BaseModel):
    start: float
    end: float
    chord: str
    function: str


class AnalysisResult(BaseModel):
    key: str
    key_confidence: float
    chords: list[ChordEvent]


class TaskState(BaseModel):
    task_id: UUID
    file_id: UUID
    status: TaskStatus = TaskStatus.pending
    progress: int = 0
    stage: str = ""
    result: AnalysisResult | None = None
    error: str | None = None


class AnalyzeRequest(BaseModel):
    file_id: UUID
```

- [ ] **Step 5: Write backend/tests/__init__.py** (empty file)

```bash
touch backend/tests/__init__.py
```

- [ ] **Step 6: Write backend/tests/conftest.py**

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def settings():
    from app.config import Settings
    return Settings(
        upload_dir="/tmp/chord-test/uploads",
        result_dir="/tmp/chord-test/results",
        redis_url="redis://localhost:6379/0",
    )


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)
```

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "feat: add backend deps, config, and models"
```

---

### Task 3: Backend Storage Service

**Files:**
- Create: `backend/app/storage.py`

- [ ] **Step 1: Write backend/app/storage.py**

```python
import json
from pathlib import Path
from uuid import UUID
import shutil

from app.config import settings
from app.models import AnalysisResult


def save_upload(file_id: UUID, content: bytes, original_filename: str) -> Path:
    """Save uploaded file to disk. Returns path to saved file."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(original_filename).suffix or ".tmp"
    path = settings.upload_dir / f"{file_id}{ext}"
    path.write_bytes(content)
    return path


def get_upload_path(file_id: UUID) -> Path | None:
    """Find uploaded file by file_id, matching any extension."""
    for path in settings.upload_dir.glob(f"{file_id}.*"):
        return path
    return None


def save_result(task_id: UUID, result: AnalysisResult) -> Path:
    """Save analysis result as JSON."""
    settings.result_dir.mkdir(parents=True, exist_ok=True)
    path = settings.result_dir / f"{task_id}.json"
    path.write_text(result.model_dump_json(indent=2))
    return path


def load_result(task_id: UUID) -> AnalysisResult | None:
    """Load analysis result from JSON."""
    path = settings.result_dir / f"{task_id}.json"
    if not path.exists():
        return None
    return AnalysisResult.model_validate_json(path.read_text())


def cleanup(file_id: UUID) -> None:
    """Remove uploaded file and associated results."""
    upload = get_upload_path(file_id)
    if upload:
        upload.unlink(missing_ok=True)
```

- [ ] **Step 2: Run basic import check**

```bash
cd backend && python -c "from app.storage import save_upload, save_result, load_result; print('OK')"
```

Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/storage.py
git commit -m "feat: add storage service for file and result I/O"
```

---

### Task 4: Backend Audio Preprocessing

**Files:**
- Create: `backend/app/audio.py`
- Create: `backend/tests/test_audio.py`

- [ ] **Step 1: Write failing test in backend/tests/test_audio.py**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_audio.py -v
```

Expected: FAIL — module not found / import error

- [ ] **Step 3: Write backend/app/audio.py**

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 22050


@dataclass
class AudioInfo:
    sample_rate: int
    duration: float
    original_sample_rate: int


def load_audio(path: Path) -> tuple[np.ndarray, AudioInfo]:
    """Load audio file, convert to mono 22050Hz. Returns (samples, info)."""
    audio, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    original_sr = sf.info(str(path)).samplerate
    duration = len(audio) / TARGET_SR
    return audio, AudioInfo(
        sample_rate=TARGET_SR,
        duration=duration,
        original_sample_rate=original_sr,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_audio.py -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/audio.py backend/tests/test_audio.py
git commit -m "feat: add audio preprocessing with 22050Hz mono conversion"
```

---

### Task 5: Backend Key Detection

**Files:**
- Create: `backend/app/analysis.py` (key detection functions only)
- Create: `backend/tests/test_analysis.py`

- [ ] **Step 1: Write failing tests in backend/tests/test_analysis.py**

```python
import numpy as np
from app.analysis import detect_key, key_to_camelot


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_analysis.py -v
```

Expected: FAIL — `detect_key` not defined

- [ ] **Step 3: Write backend/app/analysis.py with key detection**

```python
from pathlib import Path
import numpy as np
import librosa

from app.config import settings

CQT_HOP_LENGTH = 1024


def detect_key(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Detect musical key using madmom CNN if available, falling back to
    librosa Krumhansl-Schmuckler. Returns (key_name, confidence).
    """
    try:
        from madmom.features.key import CNNKeyRecognitionProcessor
        proc = CNNKeyRecognitionProcessor()
        key_str = proc(audio)
        # madmom returns string like "C major" or "A minor"
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

    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    major_corr = [np.corrcoef(np.roll(chroma_mean, i), major_profile)[0, 1]
                  for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(chroma_mean, i), minor_profile)[0, 1]
                  for i in range(12)]

    best_major = int(np.argmax(major_corr))
    best_minor = int(np.argmax(minor_corr))

    if max(major_corr) >= max(minor_corr):
        confidence = _normalize_corr(max(major_corr))
        return f"{pitch_names[best_major]} major", round(confidence, 3)
    else:
        confidence = _normalize_corr(max(minor_corr))
        return f"{pitch_names[best_minor]} minor", round(confidence, 3)


def _normalize_corr(corr: float) -> float:
    """Map raw correlation to 0-1 range."""
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))
```


- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_analysis.py -v
```

Expected: 2 PASS (key detection tests only)

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis.py backend/tests/test_analysis.py
git commit -m "feat: add key detection with madmom CNN + librosa Krumhansl-Schmuckler fallback"
```

---

### Task 6: Backend Chord Recognition

**Files:**
- Modify: `backend/app/analysis.py` — add chord recognition functions
- Modify: `backend/tests/test_analysis.py` — add chord tests

- [ ] **Step 1: Add failing chord test to backend/tests/test_analysis.py**

Append to the existing file:

```python
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
```

Add import update at the top:
```python
from app.analysis import detect_key, recognize_chords
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_analysis.py::test_recognize_chords_returns_events -v
```

Expected: FAIL — `recognize_chords` not defined

- [ ] **Step 3: Add chord recognition to backend/app/analysis.py**

Append to the existing file:

```python
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


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
        chord_probs = proc(audio)  # returns numpy array of chord predictions
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
```

Add `from dataclasses import dataclass` at top of file, and convert `ChordSegment` to a dataclass. Actually, let me adjust — instead of modifying the class, just add the `@dataclass` decorator.

Also need to add `from app.analysis import detect_key, recognize_chords, ChordSegment` to the test file import.

- [ ] **Step 4: Run chord recognition test**

```bash
cd backend && python -m pytest tests/test_analysis.py::test_recognize_chords_returns_events -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis.py backend/tests/test_analysis.py
git commit -m "feat: add chord recognition with Chordino + chroma template fallback"
```

---

### Task 7: Backend Harmonic Function Analysis

**Files:**
- Modify: `backend/app/analysis.py` — add function analysis
- Modify: `backend/tests/test_analysis.py` — add function test

- [ ] **Step 1: Add failing function analysis test**

Append to `backend/tests/test_analysis.py`:

```python
from app.analysis import detect_key, recognize_chords, ChordSegment, analyze_functions


def test_analyze_functions_in_c_major():
    """I, IV, V in C major should be T, S, D."""
    key = "C major"
    chords = [
        ChordSegment(start=0.0, end=1.0, chord="C"),
        ChordSegment(start=1.0, end=2.0, chord="F"),
        ChordSegment(start=2.0, end=3.0, chord="G"),
    ]

    result = analyze_functions(chords, key)

    assert result[0].function == "I (T)"
    assert result[1].function == "IV (S)"
    assert result[2].function == "V (D)"


def test_analyze_functions_handles_unknown_chord():
    key = "C major"
    chords = [ChordSegment(start=0.0, end=1.0, chord="X")]

    result = analyze_functions(chords, key)

    assert len(result) == 1
    assert result[0].chord == "X"
    assert result[0].function != ""  # should produce something, not crash
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_analysis.py::test_analyze_functions_in_c_major -v
```

Expected: FAIL

- [ ] **Step 3: Add function analysis to backend/app/analysis.py**

Append:

```python
from music21 import chord, roman

MAJOR_SCALE_DEGREES = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE_DEGREES = [0, 2, 3, 5, 7, 8, 10]

FUNCTION_LABELS: dict[int, str] = {
    0: "I (T)",
    1: "V/V?",
    2: "ii (S)",
    3: "iii (TS)",
    4: "IV (S)",
    5: "V (D)",
    6: "vi (TS)",
    7: "vii°",
}

FUNCTION_LABELS_MINOR: dict[int, str] = {
    0: "i (t)",
    3: "iv (s)",
    4: "V (D)",     # harmonic minor V
    5: "VI (ts)",
    6: "bVII",
    7: "vii°",
    2: "iii",
    1: "Neapolitan",
}


def analyze_functions(
    segments: list[ChordSegment], key: str
) -> list[dict]:
    """
    Annotate chord segments with harmonic function labels based on detected key.
    Returns list of dicts with start, end, chord, function.
    """
    results = []
    for seg in segments:
        func = _label_function(seg.chord, key)
        results.append({
            "start": seg.start,
            "end": seg.end,
            "chord": seg.chord,
            "function": func,
        })
    return results


def _label_function(chord_str: str, key: str) -> str:
    """Label a single chord with its harmonic function in the given key."""
    is_minor = "minor" in key.lower()
    tonic = key.split()[0]  # "C major" → "C"

    try:
        ch = chord.Chord(chord_str)
        root_midi = ch.root().midi % 12
    except Exception:
        return "?"

    # Find scale position
    scale_degrees = MINOR_SCALE_DEGREES if is_minor else MAJOR_SCALE_DEGREES
    tonic_midi = _pitch_to_midi(tonic)
    tonic_midi %= 12

    root_in_scale = (root_midi - tonic_midi) % 12

    try:
        degree = scale_degrees.index(root_in_scale)
    except ValueError:
        degree = (root_in_scale - tonic_midi) % 12

    labels = FUNCTION_LABELS_MINOR if is_minor else FUNCTION_LABELS
    return labels.get(degree, f"{_degree_to_roman(degree)}?")


def _pitch_to_midi(pitch: str) -> int:
    """Convert pitch name to MIDI number (C4=60)."""
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = pitch.rstrip("0123456789")
    octave_str = pitch[len(base):]
    midi = names.get(base, 0)
    if octave_str:
        midi += (int(octave_str) + 1) * 12
    return midi


def _degree_to_roman(degree: int) -> str:
    """Convert scale degree to roman numeral string."""
    numerals = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]
    return numerals[degree % 12]
```


- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_analysis.py::test_analyze_functions_in_c_major tests/test_analysis.py::test_analyze_functions_handles_unknown_chord -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis.py backend/tests/test_analysis.py
git commit -m "feat: add harmonic function analysis with T/S/D labeling"
```

---

### Task 8: Backend Pipeline Orchestrator

**Files:**
- Create: `backend/app/pipeline.py`
- Create: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Write failing integration test in backend/tests/test_pipeline.py**

```python
import numpy as np
import soundfile as sf
from app.pipeline import run_pipeline
from app.models import AnalysisResult


def test_run_pipeline_with_sine_wav(tmp_path):
    # Create a simple sine wave WAV
    path = tmp_path / "test.wav"
    sr = 22050
    t = np.linspace(0, 3, sr * 3)
    c = np.sin(2 * np.pi * 261.6 * t)
    e = np.sin(2 * np.pi * 329.6 * t)
    g = np.sin(2 * np.pi * 392.0 * t)
    audio = ((c + e + g) / 3).astype(np.float32)
    sf.write(str(path), audio, sr)

    result = run_pipeline(path)

    assert isinstance(result, AnalysisResult)
    assert result.key is not None
    assert len(result.key) > 0
    assert result.key_confidence >= 0.0
    assert len(result.chords) > 0
    assert all(c.function for c in result.chords)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_pipeline.py -v
```

Expected: FAIL

- [ ] **Step 3: Write backend/app/pipeline.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && python -m pytest tests/test_pipeline.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: add analysis pipeline orchestrator"
```

---

### Task 9: Backend Celery Task and Config

**Files:**
- Create: `backend/app/celery_config.py`
- Create: `backend/app/tasks.py`

- [ ] **Step 1: Write backend/app/celery_config.py**

```python
from celery import Celery
from app.config import settings

celery_app = Celery(
    "chord_analysis",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
```

- [ ] **Step 2: Write backend/app/tasks.py**

```python
from uuid import UUID
from pathlib import Path
import json

from app.celery_config import celery_app
from app.config import settings
from app.pipeline import run_pipeline
from app.storage import get_upload_path, save_result, load_result
import redis

redis_client = redis.from_url(settings.redis_url)


@celery_app.task(bind=True, name="analyze_audio")
def analyze_audio_task(self, file_id: str, task_id: str):
    """Celery task: run analysis pipeline with progress updates."""
    redis_key = f"task:{task_id}"

    def update_progress(progress: int, stage: str, **extra):
        state = {"progress": progress, "stage": stage, "status": "processing", **extra}
        redis_client.setex(redis_key, 3600, json.dumps(state))

    try:
        file_path = get_upload_path(UUID(file_id))
        if not file_path:
            update_progress(0, "failed", error="File not found", status="failed")
            return

        update_progress(10, "loading")
        result = run_pipeline(file_path)
        save_result(UUID(task_id), result)

        update_progress(
            100, "completed",
            status="completed",
            result=result.model_dump(mode="json"),
        )
    except Exception as e:
        update_progress(0, "failed", error=str(e), status="failed")
        raise
```

- [ ] **Step 3: Verify imports**

```bash
cd backend && python -c "from app.tasks import analyze_audio_task; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/app/celery_config.py backend/app/tasks.py
git commit -m "feat: add Celery task with progress tracking"
```

---

### Task 10: Backend API Routes

**Files:**
- Create: `backend/app/api.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing API integration tests in backend/tests/test_api.py**

```python
import io


def test_upload_and_analyze(client):
    """Full flow: upload a WAV, analyze it, get results."""
    # Create minimal WAV in memory (44 bytes header + data)
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050)  # 1 second of silence
    buf.seek(0)

    # Upload
    resp = client.post("/api/upload", files={"file": ("test.wav", buf, "audio/wav")})
    assert resp.status_code == 200
    data = resp.json()
    assert "file_id" in data
    file_id = data["file_id"]

    # Analyze — will fail without Celery worker, but should 202-accept
    resp = client.post("/api/analyze", json={"file_id": file_id})
    assert resp.status_code in (200, 202)
    data = resp.json()
    assert "task_id" in data
    task_id = data["task_id"]

    # Check task status
    resp = client.get(f"/api/task/{task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("pending", "processing")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_api.py -v
```

Expected: FAIL — no such endpoints

- [ ] **Step 3: Write backend/app/api.py**

```python
from uuid import uuid4, UUID
import json

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingSSEResponse
import redis.asyncio as aioredis

from app.config import settings
from app.models import AnalyzeRequest, TaskState, TaskStatus, AnalysisResult
from app.storage import save_upload, load_result, get_upload_path
from app.tasks import analyze_audio_task, redis_client

router = APIRouter(prefix="/api")


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload an audio file. Returns file_id."""
    file_id = uuid4()

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    save_upload(file_id, content, file.filename or "audio.tmp")
    return {"file_id": str(file_id)}


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Submit analysis task. Returns task_id."""
    if not get_upload_path(req.file_id):
        raise HTTPException(status_code=404, detail="File not found")

    task_id = uuid4()

    # Initialize task state
    state = TaskState(
        task_id=task_id,
        file_id=req.file_id,
        status=TaskStatus.pending,
    )
    redis_client.setex(
        f"task:{task_id}", 3600,
        state.model_dump_json(),
    )

    # Enqueue Celery task
    analyze_audio_task.delay(str(req.file_id), str(task_id))

    return {"task_id": str(task_id)}


@router.get("/task/{task_id}")
def get_task(task_id: UUID):
    """Get task status and result."""
    raw = redis_client.get(f"task:{task_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Task not found")
    return json.loads(raw)


@router.get("/task/{task_id}/stream")
async def stream_task(task_id: UUID):
    """SSE stream for real-time progress updates."""
    async def event_stream():
        redis_async = aioredis.from_url(settings.redis_url)
        last_progress = -1
        while True:
            raw = await redis_async.get(f"task:{task_id}")
            if raw:
                data = json.loads(raw)
                progress = data.get("progress", 0)
                if progress != last_progress:
                    last_progress = progress
                    yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") in ("completed", "failed"):
                    break
            else:
                yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                break
    return StreamingSSEResponse(event_stream(), media_type="text/event-stream")


@router.get("/result/{task_id}")
def get_result(task_id: UUID):
    """Get analysis result from disk."""
    result = load_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result.model_dump(mode="json")
```

- [ ] **Step 4: Write backend/app/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router

app = FastAPI(title="Chord Analysis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

- [ ] **Step 5: Run tests**

```bash
cd backend && python -m pytest tests/test_api.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/api.py backend/app/main.py backend/tests/test_api.py
git commit -m "feat: add REST API endpoints with SSE progress streaming"
```

---

### Task 11: Frontend Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`

- [ ] **Step 1: Write frontend/package.json**

```json
{
  "name": "chord-analysis-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3",
    "react-dom": "^18.3"
  },
  "devDependencies": {
    "@types/react": "^18.3",
    "@types/react-dom": "^18.3",
    "@vitejs/plugin-react": "^4.3",
    "autoprefixer": "^10.4",
    "postcss": "^8.4",
    "tailwindcss": "^3.4",
    "typescript": "^5.6",
    "vite": "^6.0"
  }
}
```

- [ ] **Step 2: Write frontend/tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Write frontend/vite.config.ts**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

- [ ] **Step 4: Write frontend/tailwind.config.js**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

- [ ] **Step 5: Write frontend/postcss.config.js**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

- [ ] **Step 6: Write frontend/index.html**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Chord Analysis</title>
  </head>
  <body class="bg-gray-900 text-gray-100 min-h-screen">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Write frontend/src/main.tsx**

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 8: Create frontend/src/index.css**

```bash
mkdir -p frontend/src/services frontend/src/components
cat > frontend/src/index.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF
```

- [ ] **Step 9: Install and verify build**

```bash
cd frontend && npm install && npx tsc --noEmit
```

Expected: build succeeds (App.tsx not yet created, will error — skip tsc check for now, just verify npm install succeeds)

- [ ] **Step 10: Commit**

```bash
git add frontend/
git commit -m "chore: scaffold React frontend with Vite + Tailwind"
```

---

### Task 12: Frontend Types and API Service

**Files:**
- Create: `frontend/src/types.ts`
- Create: `frontend/src/services/api.ts`

- [ ] **Step 1: Write frontend/src/types.ts**

```typescript
export interface ChordEvent {
  start: number;
  end: number;
  chord: string;
  function: string;
}

export interface AnalysisResult {
  key: string;
  key_confidence: number;
  chords: ChordEvent[];
}

export interface TaskState {
  task_id: string;
  file_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  stage: string;
  result: AnalysisResult | null;
  error: string | null;
}
```

- [ ] **Step 2: Write frontend/src/services/api.ts**

```typescript
import type { TaskState } from "../types";

const BASE = "/api";

export async function uploadFile(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  const data = await res.json();
  return data.file_id;
}

export async function analyzeFile(fileId: string): Promise<string> {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId }),
  });
  if (!res.ok) throw new Error(`Analyze failed: ${res.statusText}`);
  const data = await res.json();
  return data.task_id;
}

export async function getTask(taskId: string): Promise<TaskState> {
  const res = await fetch(`${BASE}/task/${taskId}`);
  if (!res.ok) throw new Error(`Task fetch failed: ${res.statusText}`);
  return res.json();
}

export function subscribeTask(
  taskId: string,
  onUpdate: (state: TaskState) => void,
  onError: (err: Event) => void
): EventSource {
  const es = new EventSource(`${BASE}/task/${taskId}/stream`);
  es.onmessage = (event) => {
    const data = JSON.parse(event.data) as TaskState;
    onUpdate(data);
    if (data.status === "completed" || data.status === "failed") {
      es.close();
    }
  };
  es.onerror = onError;
  return es;
}
```

- [ ] **Step 3: Verify TypeScript compilation**

```bash
cd frontend && npx tsc --noEmit
```

Expected: PASS (no errors)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/src/services/api.ts
git commit -m "feat: add frontend types and API service layer"
```

---

### Task 13: Frontend Components and App

**Files:**
- Create: `frontend/src/components/UploadArea.tsx`
- Create: `frontend/src/components/ProgressIndicator.tsx`
- Create: `frontend/src/components/ChordTimeline.tsx`
- Create: `frontend/src/App.tsx`

- [ ] **Step 1: Write frontend/src/components/UploadArea.tsx**

```tsx
import { useCallback, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  disabled: boolean;
}

export default function UploadArea({ onFileSelected, disabled }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
        dragOver ? "border-blue-400 bg-blue-400/10" : "border-gray-600 hover:border-gray-400"
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept=".wav,.mp3,.flac,.m4a,.aac"
        onChange={handleChange}
        className="hidden"
        id="file-input"
        disabled={disabled}
      />
      <label htmlFor="file-input" className="cursor-pointer">
        <p className="text-lg mb-2">
          {dragOver ? "Drop your file here" : "Drag & drop an audio file"}
        </p>
        <p className="text-sm text-gray-400">
          or click to browse — WAV, MP3, FLAC, M4A (max 50MB)
        </p>
      </label>
    </div>
  );
}
```

- [ ] **Step 2: Write frontend/src/components/ProgressIndicator.tsx**

```tsx
interface Props {
  progress: number;
  stage: string;
}

const STAGE_LABELS: Record<string, string> = {
  loading: "Loading audio...",
  key_detection: "Detecting musical key...",
  chord_recognition: "Recognizing chords...",
  function_analysis: "Analyzing harmonic function...",
  completed: "Analysis complete",
  failed: "Analysis failed",
};

export default function ProgressIndicator({ progress, stage }: Props) {
  const label = STAGE_LABELS[stage] || stage || "Waiting...";

  return (
    <div className="my-6">
      <div className="flex justify-between mb-1">
        <span className="text-sm">{label}</span>
        <span className="text-sm text-gray-400">{progress}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write frontend/src/components/ChordTimeline.tsx**

```tsx
import type { AnalysisResult } from "../types";

interface Props {
  result: AnalysisResult;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${m}:${s.toString().padStart(2, "0")}.${ms}`;
}

export default function ChordTimeline({ result }: Props) {
  const exportJson = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "chord-analysis.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="text-lg font-semibold">Key: {result.key}</span>
          <span className="text-sm text-gray-400 ml-3">
            (confidence: {(result.key_confidence * 100).toFixed(0)}%)
          </span>
        </div>
        <button
          onClick={exportJson}
          className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-700 rounded"
        >
          Export JSON
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="text-sm text-gray-400 border-b border-gray-700">
              <th className="py-2 pr-4">Time</th>
              <th className="py-2 pr-4">Chord</th>
              <th className="py-2">Function</th>
            </tr>
          </thead>
          <tbody>
            {result.chords.map((ev, i) => (
              <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="py-2 pr-4 font-mono text-sm">
                  {formatTime(ev.start)} – {formatTime(ev.end)}
                </td>
                <td className="py-2 pr-4 font-semibold">{ev.chord}</td>
                <td className="py-2 text-blue-300">{ev.function}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Write frontend/src/App.tsx**

```tsx
import { useState, useCallback, useRef } from "react";
import UploadArea from "./components/UploadArea";
import ProgressIndicator from "./components/ProgressIndicator";
import ChordTimeline from "./components/ChordTimeline";
import { uploadFile, analyzeFile, subscribeTask } from "./services/api";
import type { TaskState } from "./types";

type AppState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "analyzing"; taskId: string; progress: number; stage: string }
  | { phase: "done"; taskState: TaskState }
  | { phase: "error"; message: string };

export default function App() {
  const [state, setState] = useState<AppState>({ phase: "idle" });
  const eventSourceRef = useRef<EventSource | null>(null);

  const handleFile = useCallback(async (file: File) => {
    try {
      setState({ phase: "uploading" });
      const fileId = await uploadFile(file);

      const taskId = await analyzeFile(fileId);
      setState({ phase: "analyzing", taskId, progress: 0, stage: "" });

      const es = subscribeTask(
        taskId,
        (update) => {
          if (update.status === "completed" || update.status === "failed") {
            setState(
              update.status === "completed"
                ? { phase: "done", taskState: update }
                : { phase: "error", message: update.error || "Analysis failed" }
            );
          } else {
            setState({
              phase: "analyzing",
              taskId,
              progress: update.progress,
              stage: update.stage,
            });
          }
        },
        () => setState({ phase: "error", message: "Connection lost" })
      );
      eventSourceRef.current = es;
    } catch (err) {
      setState({ phase: "error", message: (err as Error).message });
    }
  }, []);

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Chord Analysis</h1>

      <UploadArea
        onFileSelected={handleFile}
        disabled={state.phase === "uploading" || state.phase === "analyzing"}
      />

      {state.phase === "analyzing" && (
        <ProgressIndicator progress={state.progress} stage={state.stage} />
      )}

      {state.phase === "done" && state.taskState.result && (
        <ChordTimeline result={state.taskState.result} />
      )}

      {state.phase === "error" && (
        <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded text-red-200">
          {state.message}
          <button
            className="ml-3 underline"
            onClick={() => setState({ phase: "idle" })}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: add UploadArea, ProgressIndicator, ChordTimeline, and App"
```

---

### Task 14: Docker Setup and Final Verification

**Files:**
- Create: `backend/Dockerfile`
- Create: `frontend/Dockerfile`

- [ ] **Step 1: Write backend/Dockerfile**

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
```

- [ ] **Step 2: Write frontend/Dockerfile**

```dockerfile
FROM node:22-alpine

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
EXPOSE 5173
```

- [ ] **Step 3: Run backend unit tests**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 4: Verify frontend builds**

```bash
cd frontend && npm run build
```

Expected: build succeeds

- [ ] **Step 5: Verify FastAPI starts**

```bash
cd backend && timeout 5 uvicorn app.main:app --port 8000 || true
```

Expected: server starts without import errors

- [ ] **Step 6: Final commit**

```bash
git add backend/Dockerfile frontend/Dockerfile
git commit -m "chore: add Dockerfiles and finalize project"
```
