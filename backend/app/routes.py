"""Analysis API routes: upload, status, result, download, delete."""
import uuid
import threading
import time
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.config import (
    UPLOAD_DIR, RESULT_DIR, ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE, MAX_DURATION_SECONDS, ANALYSIS_TIMEOUT_SECONDS,
    RESULT_TTL_SECONDS,
)
from app.schemas import (
    UploadResponse, AnalysisStatus, AnalysisStatusEnum,
    Stages, AnalysisResult, ErrorResponse,
)


router = APIRouter(prefix="/api")
_jobs: dict[str, dict] = {}


def _cleanup_expired():
    """Remove expired results."""
    now = time.time()
    expired = [
        aid for aid, job in _jobs.items()
        if job.get("completed_at") and (now - job["completed_at"]) > RESULT_TTL_SECONDS
    ]
    for aid in expired:
        _jobs.pop(aid, None)
        for f in RESULT_DIR.glob(f"{aid}*"):
            f.unlink(missing_ok=True)


def _run_analysis(aid: str, audio_path: Path, language: str, device: str):
    """Background analysis worker."""
    from app.pipeline.runner import PipelineRunner

    job = _jobs.get(aid)
    if not job:
        return
    runner = None
    try:
        job["status"] = AnalysisStatusEnum.processing
        job["stages"] = Stages(
            vocal_sep=True, beat_tracking=True, chord_detection=True,
            key_detection=True, lyrics=True, harmony_analysis=True,
        )
        runner = PipelineRunner(device=device, lyrics_language=language)
        result = runner.run(audio_path, analysis_id=aid)
        job["result"] = result
        job["status"] = AnalysisStatusEnum.done
        job["progress"] = 100
        job["completed_at"] = time.time()
    except Exception as e:
        job["status"] = AnalysisStatusEnum.error
        job["error"] = str(e)
    finally:
        if runner is not None:
            runner.backbone.unload()


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), language: str = Query("zh")):
    """Upload an audio file for analysis. Returns analysis ID and status URL."""
    from app.utils.audio import validate_audio

    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported format: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)} MB")

    aid = uuid.uuid4().hex[:12]
    audio_path = UPLOAD_DIR / f"{aid}{ext}"
    audio_path.write_bytes(contents)

    is_valid, err = validate_audio(audio_path)
    if not is_valid:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(400, detail=err)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    _jobs[aid] = {
        "filename": file.filename,
        "status": AnalysisStatusEnum.queued,
        "progress": 0,
        "stages": Stages(),
        "error": None,
        "result": None,
        "completed_at": None,
    }

    thread = threading.Thread(target=_run_analysis, args=(aid, audio_path, language, device), daemon=True)
    thread.start()

    return UploadResponse(analysis_id=aid, status_url=f"/api/status/{aid}")


@router.get("/status/{analysis_id}", response_model=AnalysisStatus)
async def get_status(analysis_id: str):
    """Poll analysis progress."""
    _cleanup_expired()
    job = _jobs.get(analysis_id)
    if not job:
        raise HTTPException(404, detail="Analysis not found")
    return AnalysisStatus(
        id=analysis_id,
        filename=job["filename"],
        status=job["status"],
        progress=job["progress"],
        stages=job["stages"],
        error=job["error"],
    )


@router.get("/result/{analysis_id}")
async def get_result(analysis_id: str):
    """Get analysis result. Returns 202 if still processing, 404 if not found."""
    _cleanup_expired()
    job = _jobs.get(analysis_id)
    if not job:
        raise HTTPException(404, detail="Analysis not found")
    if job["status"] == AnalysisStatusEnum.error:
        raise HTTPException(500, detail=job.get("error", "Analysis failed"))
    if job["status"] != AnalysisStatusEnum.done:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={"status": job["status"].value, "progress": job["progress"]},
        )
    return job["result"]


@router.get("/download/{analysis_id}")
async def download_result(analysis_id: str):
    """Download analysis result as JSON file."""
    _cleanup_expired()
    job = _jobs.get(analysis_id)
    if not job:
        raise HTTPException(404, detail="Analysis not found")
    if job["status"] != AnalysisStatusEnum.done:
        raise HTTPException(409, detail="Result not ready")
    result: AnalysisResult = job["result"]
    json_path = RESULT_DIR / f"{analysis_id}.json"
    import json
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    from fastapi.responses import FileResponse
    return FileResponse(
        json_path,
        media_type="application/json",
        filename=f"chord-analysis-{analysis_id}.json",
    )


@router.delete("/result/{analysis_id}")
async def delete_result(analysis_id: str):
    """Delete analysis result and uploaded file."""
    job = _jobs.pop(analysis_id, None)
    for f in UPLOAD_DIR.glob(f"{analysis_id}*"):
        f.unlink(missing_ok=True)
    for f in RESULT_DIR.glob(f"{analysis_id}*"):
        f.unlink(missing_ok=True)
    return {"message": "Deleted"}
