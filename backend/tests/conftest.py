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
