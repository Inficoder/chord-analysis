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
