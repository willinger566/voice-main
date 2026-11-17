import asyncio
from pathlib import Path
import threading
import time
import pytest
from fastapi.testclient import TestClient
from api import main
from audio import AudioValidationError


class FakeRecognizer:
    device = "cpu"
    def __init__(self):
        self.paths = []
        self.error = None
        self.delay = 0
    def predict(self, path):
        self.paths.append(path)
        assert Path(path).exists()
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return {"predicted_emotion": "happy", "confidence": .8}


@pytest.fixture
def client(monkeypatch):
    fake = FakeRecognizer()
    monkeypatch.setenv("VOICE_SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("CONFIG_PATH", "models/production/model_config.json")
    monkeypatch.setattr(main, "recognizer", fake)
    monkeypatch.setattr(main, "inference_gate", asyncio.Semaphore(1))
    with TestClient(main.app) as client:
        yield client, fake


def test_same_origin_assets_and_metadata(client):
    api, _ = client
    assert api.get("/").status_code == 200
    assert api.get("/assets/app.js").headers["x-content-type-options"] == "nosniff"
    assert api.get("/info").json()["test_accuracy"] == pytest.approx(139 / 180)
    assert api.get("/info").json()["training_samples"] == 6498
    assert api.get("/samples").json()
    assert api.get("/samples/happy").headers["content-type"] == "audio/wav"
    assert api.get("/samples/not-a-sample").status_code == 404


def test_prediction_extension_and_cleanup(client):
    api, fake = client
    response = api.post("/predict", files={"file": ("UPPER.WAV", b"audio", "audio/wav")})
    assert response.status_code == 200 and response.json()["success"]
    assert fake.paths[0].endswith(".wav") and not Path(fake.paths[0]).exists()
    assert api.post("/predict", files={"file": ("unsafe.exe", b"data")}).status_code == 415


def test_reject_empty_and_oversized_requests(client):
    api, fake = client
    assert api.post("/predict", files={"file": ("empty.wav", b"")}).status_code == 422
    response = api.post("/predict", content=b"x", headers={"content-length": str(main.MAX_UPLOAD + 1048577)})
    assert response.status_code == 413 and not fake.paths
    response = api.post("/predict", content=iter([b"unbounded"]), headers={"transfer-encoding": "chunked"})
    assert response.status_code == 411


def test_cleanup_on_validation_and_unexpected_error(client):
    api, fake = client
    fake.error = AudioValidationError("音频过短")
    assert api.post("/predict", files={"file": ("short.wav", b"x")}).status_code == 422
    fake.error = RuntimeError("private/model/path")
    response = api.post("/predict", files={"file": ("broken.wav", b"x")})
    assert response.status_code == 500 and "private/model/path" not in response.text
    assert all(not Path(path).exists() for path in fake.paths)
    assert not main.inference_gate.locked()


def test_health_during_inference_and_concurrency_limit(client):
    api, fake = client
    fake.delay = .7
    responses = []
    thread = threading.Thread(target=lambda: responses.append(api.post("/predict", files={"file": ("first.wav", b"x")})))
    thread.start()
    for _ in range(100):
        if fake.paths:
            break
        time.sleep(.01)
    started = time.monotonic()
    assert api.get("/health").json()["model_loaded"]
    assert time.monotonic() - started < .4
    assert api.post("/predict", files={"file": ("second.wav", b"x")}).status_code == 429
    thread.join()
    assert responses[0].status_code == 200


def test_truthful_unavailable_state_and_streaming(client, monkeypatch):
    api, _ = client
    monkeypatch.setattr(main, "recognizer", None)
    assert api.get("/health").json()["status"] == "degraded"
    assert api.post("/predict", files={"file": ("audio.wav", b"x")}).status_code == 503
    assert api.post("/predict_stream").status_code == 501
