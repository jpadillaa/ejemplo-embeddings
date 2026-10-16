"""Único punto de acceso a Ollama. No utiliza endpoints generativos."""
import logging
import os
from urllib.parse import urlparse
import httpx
import numpy as np
from numpy.typing import NDArray

MODEL = "embeddinggemma-2:440m"
DIMENSIONS = 768
Vector = NDArray[np.float32]
logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Error seguro para presentar en la interfaz."""


def validate_embeddings(data: dict, count: int) -> Vector:
    try:
        vectors = np.asarray(data["embeddings"], dtype=np.float32)
        if vectors.shape != (count, DIMENSIONS):
            raise ValueError("dimensiones o número de vectores incorrectos")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.isfinite(vectors).all() or not (norms > 0).all():
            raise ValueError("vector no finito o de norma cero")
        return vectors
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        logger.error("Respuesta de embeddings inválida: %s", exc)
        raise OllamaError("Ollama devolvió un vector inválido; se requieren vectores finitos de 768 dimensiones y norma positiva.") from exc


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120,
                 transport: httpx.BaseTransport | None = None):
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password:
            raise ValueError("Ollama debe utilizar una dirección HTTP de loopback local.")
        self.http = httpx.Client(base_url=url, timeout=timeout, trust_env=False, transport=transport)

    def request(self, method: str, endpoint: str, **kwargs) -> dict:
        try:
            response = self.http.request(method, endpoint, **kwargs)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("objeto JSON esperado")
            return result
        except httpx.TimeoutException as exc:
            raise OllamaError("Tiempo de espera agotado. El modelo puede estar cargándose; vuelve a intentarlo.") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OllamaError(f"Modelo o endpoint no disponible. Ejecuta: ollama pull {MODEL}. Si persiste, actualiza Ollama.") from exc
            raise OllamaError("Ollama rechazó la solicitud. Actualiza Ollama y repite la validación multimodal de /api/embed.") from exc
        except httpx.RequestError as exc:
            raise OllamaError("No se puede conectar con Ollama. Inicia el servicio local con ollama serve y vuelve a intentar.") from exc
        except ValueError as exc:
            raise OllamaError("Ollama devolvió una respuesta JSON inválida.") from exc

    def identity(self) -> str:
        data = self.request("GET", "/api/tags")
        for model in data.get("models", []):
            if model.get("name") == MODEL or model.get("model") == MODEL:
                digest = model.get("digest")
                if not digest:
                    raise OllamaError("Ollama no informó el digest del modelo; no se puede validar la caché.")
                version = self.request("GET", "/api/version").get("version", "unknown")
                return f"{MODEL}:{digest}:{version}"
        raise OllamaError(f"Modelo no instalado. Ejecuta: ollama pull {MODEL}")

    def embed_text(self, descriptions: list[str]) -> Vector:
        return validate_embeddings(self.request("POST", "/api/embed", json={
            "model": MODEL, "input": descriptions, "keep_alive": "30m", "truncate": False,
        }), len(descriptions))

    def embed_image(self, image_base64: str) -> Vector:
        return validate_embeddings(self.request("POST", "/api/embed", json={
            "model": MODEL, "input": {"image": image_base64}, "keep_alive": "30m",
        }), 1)[0]
