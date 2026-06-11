from pathlib import Path
from uuid import UUID

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
    """Remove uploaded file from disk."""
    upload = get_upload_path(file_id)
    if upload:
        upload.unlink(missing_ok=True)
