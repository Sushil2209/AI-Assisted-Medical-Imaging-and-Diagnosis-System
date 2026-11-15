import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.inference_api import main  # noqa: E402


class FakeModel:
    def __call__(self, x):
        return torch.tensor([[2.0]])  # sigmoid(2.0) ~ 0.8808


@pytest.fixture
def client(monkeypatch):
    # No `with`, so the lifespan (real checkpoint loading) never runs
    monkeypatch.setitem(main.state, "model", FakeModel())
    monkeypatch.setattr(main, "orthanc_get", lambda path: SimpleNamespace(content=b"dicom"))
    monkeypatch.setattr(main, "dicom_to_image", lambda data: Image.new("RGB", (32, 32)))
    monkeypatch.setattr(main, "build_transform", lambda size: lambda img: torch.zeros(3, size, size))
    return TestClient(main.app)


def fake_explain_factory(calls, prob=0.3, fail=False):
    def fake_explain(model, image, image_size, device):
        calls.append((model, image_size, device))
        if fail:
            raise RuntimeError("boom")
        return prob, np.full((8, 8, 3), 200, dtype=np.uint8)
    return fake_explain


def test_predict_without_gradcam(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "explain", fake_explain_factory(calls))
    body = client.post("/predict/abc").json()
    assert calls == []
    assert set(body) == {"instance_id", "predicted_class", "pneumonia_probability", "threshold", "disclaimer"}
    assert body["instance_id"] == "abc"
    assert body["pneumonia_probability"] == 0.8808
    assert body["predicted_class"] == "Pneumonia"
    assert body["threshold"] == main.CLS["threshold"]
    assert body["disclaimer"] == main.DISCLAIMER


def test_predict_with_gradcam(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "explain", fake_explain_factory(calls, prob=0.3))
    body = client.post("/predict/abc?gradcam=true").json()
    assert calls == [(main.state["model"], main.CLS["image_size"], "cpu")]
    assert body["pneumonia_probability"] == 0.3  # probability comes from explain()
    assert body["predicted_class"] == "No pneumonia"
    assert body["disclaimer"] == main.DISCLAIMER
    png = Image.open(io.BytesIO(base64.b64decode(body["gradcam_png_base64"])))
    assert png.format == "PNG" and png.size == (8, 8)


def test_gradcam_failure_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "explain", fake_explain_factory([], fail=True))
    resp = client.post("/predict/abc?gradcam=true")
    assert resp.status_code == 500
    assert "gradcam_png_base64" not in resp.json()
