# Chord Analysis Tool — Design Spec

## Overview

A web application that analyzes audio files (recordings, music) to detect chords, key, and harmonic function. Exposes REST APIs for third-party tool integration.

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, Celery (Redis broker)
- **Audio processing**: librosa, madmom (Chordino CNN model), music21
- **Frontend**: React 18+, TypeScript, Vite, Tailwind CSS
- **Storage**: Local filesystem (audio files + JSON results)

## Architecture

Three layers:

```
Frontend (React SPA) → REST/SSE → FastAPI Backend → Celery Worker
                                                         │
                                    ┌────────────────────┘
                                    ▼
                     Analysis Pipeline (librosa/madmom/music21)
```

### Layer 1: Web Service (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Upload audio, returns `file_id` |
| `/api/analyze` | POST | Submit analysis task (`file_id` → `task_id`) |
| `/api/task/{task_id}` | GET | Task status and result |
| `/api/task/{task_id}/stream` | GET | SSE progress stream |
| `/api/result/{task_id}` | GET | Full result JSON |

Async model: Celery workers consume tasks from Redis queue. Status stored in Redis with key `task:{task_id}`.

### Layer 2: Analysis Pipeline (Celery Worker)

Three sequential stages, each runs in the worker process:

1. **Key Detection**: madmom CNNKeyRecognitionProcessor, fallback to librosa Krumhansl-Schmuckler
2. **Chord Recognition**: madmom CNNChordFeatureProcessor (Chordino) at FPS=2 (~0.5s window)
3. **Harmonic Function Analysis**: music21 RomanNumeral analysis + custom rules for T/S/D labels

Audio preprocessing: all input formats converted to WAV 22050Hz mono before analysis.

### Layer 3: Frontend (React SPA)

Single page, three sections:
- **Upload area**: drag-and-drop, file info display
- **Progress indicator**: SSE-driven, shows current stage
- **Results display**: key banner + chord timeline table (time | chord | function), row expansion for chord details, JSON export button

## Data Flow

```
Upload (audio) → save to disk → return file_id
analyze(file_id) → create task → enqueue Celery job → return task_id
Worker: preprocess → key → chords → functions → save JSON result
Client: SSE/GET task/{id} → poll or stream → display results
```

## API Response Format

```json
{
  "task_id": "uuid",
  "status": "pending|processing|completed|failed",
  "progress": 75,
  "stage": "chord_recognition",
  "result": {
    "key": "C major",
    "key_confidence": 0.92,
    "chords": [
      {"start": 0.0, "end": 2.1, "chord": "Cmaj7", "function": "I (T)"},
      {"start": 2.1, "end": 4.0, "chord": "Am7",  "function": "vi (TS)"}
    ]
  },
  "error": null
}
```

## Supported Audio Formats

WAV, MP3, FLAC, M4A/AAC. Preprocessed to 22050Hz mono WAV.

## Scope & Constraints

- File size limit: 50MB (configurable)
- Max duration: 15 minutes (configurable)
- Single user / no auth (auth can be added later)
- No real-time streaming analysis (offline analysis only, < 2 min per 5 min audio target)
