import os
from pathlib import Path
import pytest
from visual_embeddings.categories import defaults
from visual_embeddings.embedding_engine import EmbeddingEngine


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_INTEGRATION") != "1", reason="Activar RUN_OLLAMA_INTEGRATION=1 para ejecutar inferencia real")
def test_real_multimodal_model():
    engine = EmbeddingEngine()
    engine.prepare(defaults())
    for name in ("gato", "bicicleta", "silla"):
        result = engine.analyze(str(Path(__file__).parents[1] / f"assets/examples/{name}.jpg"), defaults())
        assert result.image.shape == (768,)
        assert result.texts.shape == (8, 768)
        assert len(result.ranking) == 8
    engine.close()
