from uuid import uuid4, UUID
import json

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import StreamingResponse
import redis.asyncio as aioredis

from app.config import settings
from app.models import AnalyzeRequest, TaskState, TaskStatus
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

    # Initialize task state in Redis
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
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/result/{task_id}")
def get_result(task_id: UUID):
    """Get analysis result from disk."""
    result = load_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result.model_dump(mode="json")
