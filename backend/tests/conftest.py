import json
from collections import defaultdict

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class FakeRedis(MagicMock):
    """In-memory Redis mock for testing without a real Redis server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._store: dict[str, str] = {}

    def setex(self, name, time, value):
        self._store[name] = value

    def get(self, name):
        return self._store.get(name)


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
    fake_redis = FakeRedis()

    mock_delay = MagicMock()
    mock_delay.return_value = MagicMock(id="fake-task-id")

    with (
        patch("app.api.redis_client", fake_redis),
        patch("app.api.analyze_audio_task.delay", mock_delay),
    ):
        from app.main import app
        yield TestClient(app)
