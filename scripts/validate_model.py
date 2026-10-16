"""Prueba real: guarda evidencia reproducible, nunca inventa puntuaciones."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_embeddings.categories import Category, defaults
from visual_embeddings.embedding_engine import EmbeddingEngine


def main() -> None:
    engine = EmbeddingEngine()
    categories = defaults()
    engine.prepare(categories)
    evidence = {"model_identity": engine.identity, "examples": {}}
    for name in ("gato", "bicicleta", "silla"):
        result = engine.analyze(f"assets/examples/{name}.jpg", categories)
        evidence["examples"][name] = {
            "dimensions": len(result.image), "seconds": result.total_seconds,
            "ranking": [{"label": m.category.label, "score": m.score} for m in result.ranking],
        }
        print(name, result.ranking[0].category.label, result.ranking[0].score, flush=True)
    expanded = engine.analyze("assets/examples/silla.jpg", categories + [Category("Silla", "Una fotografía de una silla")])
    evidence["chair_added"] = {"image_cached": expanded.image_cached,
        "ranking": [{"label": m.category.label, "score": m.score} for m in expanded.ranking]}
    Path("validation.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    engine.close()


if __name__ == "__main__":
    main()
