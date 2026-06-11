from uuid import UUID
import json

from app.celery_config import celery_app
from app.config import settings
from app.pipeline import run_pipeline
from app.storage import get_upload_path, save_result
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
