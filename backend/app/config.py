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
