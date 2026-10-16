import httpx
import numpy as np
import pytest
from visual_embeddings.ollama_client import OllamaClient, OllamaError, MODEL, validate_embeddings


@pytest.mark.parametrize("value", [{}, {"embeddings": [[]]}, {"embeddings": [[0] * 768]}, {"embeddings": [[float("nan")] * 768]}, {"embeddings": [[1] * 767]}, {"embeddings": [[1] * 768] * 2}])
def test_reject_invalid_response(value):
    with pytest.raises(OllamaError):
        validate_embeddings(value, 1)


def test_http_contract():
    import json
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        if isinstance(payload["input"], dict):
            assert payload["input"] == {"image": "base64"}
        else:
            assert payload["input"] == ["Una fotografía de un gato"]
        return httpx.Response(200, json={"embeddings": [[1.] * 768]})
    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert client.embed_image("base64").shape == (768,)
    assert client.embed_text(["Una fotografía de un gato"]).shape == (1, 768)


@pytest.mark.parametrize("status,match", [(400, "Actualiza"), (404, "ollama pull"), (500, "Actualiza")])
def test_http_errors(status, match):
    client = OllamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(status)))
    with pytest.raises(OllamaError, match=match):
        client.embed_image("x")


def test_connectivity_and_timeout():
    for error, message in [(httpx.ConnectError("offline"), "ollama serve"), (httpx.ReadTimeout("slow"), "Tiempo de espera")]:
        def handler(request):
            raise error
        with pytest.raises(OllamaError, match=message):
            OllamaClient(transport=httpx.MockTransport(handler)).identity()


def test_no_remote_inference():
    with pytest.raises(ValueError, match="local"):
        OllamaClient("https://example.com")


def test_missing_model():
    client = OllamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"models": []})))
    with pytest.raises(OllamaError, match=f"ollama pull {MODEL}"):
        client.identity()
