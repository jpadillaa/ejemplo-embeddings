"""Cachés acotadas de vectores por sesión; nunca conserva fotografías."""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .categories import Category
from .classifier import Match, classify
from .image_processor import prepare_image
from .ollama_client import OllamaClient, Vector


@dataclass
class Result:
    ranking: list[Match]
    image: Vector
    texts: Vector
    visual_seconds: float
    cosine_seconds: float
    total_seconds: float
    image_cached: bool


class EmbeddingEngine:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self.identity: str | None = None
        self.text_cache: OrderedDict[str, Vector] = OrderedDict()
        self.image_cache: OrderedDict[str, Vector] = OrderedDict()

    def refresh(self) -> None:
        identity = self.client.identity()
        if identity != self.identity:
            self.text_cache.clear()
            self.image_cache.clear()
            self.identity = identity

    def texts(self, categories: list[Category]) -> Vector:
        descriptions = [c.description for c in categories]
        missing = list(dict.fromkeys(d for d in descriptions if d not in self.text_cache))
        if missing:
            vectors = self.client.embed_text(missing)
            self.text_cache.update(zip(missing, vectors))
        result = np.stack([self.text_cache[d] for d in descriptions])
        while len(self.text_cache) > 512:
            self.text_cache.popitem(last=False)
        return result

    def prepare(self, categories: list[Category]) -> None:
        self.refresh()
        self.texts(categories)

    def analyze(self, path: str, categories: list[Category]) -> Result:
        start = perf_counter()
        prepared = prepare_image(path)
        self.refresh()
        cached = prepared.digest in self.image_cache
        visual_start = perf_counter()
        if not cached:
            self.image_cache[prepared.digest] = self.client.embed_image(prepared.base64)
        self.image_cache.move_to_end(prepared.digest)
        while len(self.image_cache) > 4:
            self.image_cache.popitem(last=False)
        image = self.image_cache[prepared.digest]
        visual_seconds = perf_counter() - visual_start
        texts = self.texts(categories)
        cosine_start = perf_counter()
        ranking = classify(image, texts, categories)
        cosine_seconds = perf_counter() - cosine_start
        return Result(ranking, image, texts, visual_seconds, cosine_seconds, perf_counter() - start, cached)

    def close(self) -> None:
        self.image_cache.clear()
        self.text_cache.clear()
        self.client.http.close()
