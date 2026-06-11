import io
import wave


def test_upload_and_analyze(client):
    """Full flow: upload a WAV, analyze it, get results."""
    # Create minimal WAV in memory
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

    # Analyze
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
