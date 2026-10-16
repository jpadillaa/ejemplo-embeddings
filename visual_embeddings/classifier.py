"""Similitud coseno explícita y ranking; sin dependencias de interfaz o red."""
from dataclasses import dataclass
import numpy as np
from .categories import Category
from .ollama_client import Vector


@dataclass(frozen=True)
class Match:
    category: Category
    score: float
    index: int


def cosine_scores(image: Vector, texts: Vector) -> Vector:
    image = np.asarray(image, dtype=np.float64)
    texts = np.asarray(texts, dtype=np.float64)
    if image.ndim != 1 or texts.ndim != 2 or texts.shape[1] != image.size or not len(texts):
        raise ValueError("Dimensiones incompatibles para similitud coseno.")
    if not np.isfinite(image).all() or not np.isfinite(texts).all():
        raise ValueError("Los vectores deben contener valores finitos.")
    image_norm = np.linalg.norm(image)
    text_norms = np.linalg.norm(texts, axis=1, keepdims=True)
    if image_norm == 0 or (text_norms == 0).any():
        raise ValueError("Los vectores deben tener norma positiva.")
    return (texts / text_norms) @ (image / image_norm)


def classify(image: Vector, texts: Vector, categories: list[Category]) -> list[Match]:
    if len(categories) != len(texts):
        raise ValueError("Se requiere un vector por categoría.")
    scores = cosine_scores(image, texts)
    return [Match(categories[i], float(scores[i]), int(i)) for i in np.argsort(-scores, kind="stable")]
