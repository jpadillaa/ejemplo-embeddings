import numpy as np
from PIL import Image
from visual_embeddings.categories import Category, defaults
from visual_embeddings.embedding_engine import EmbeddingEngine


class ControlledClient:
    def __init__(self):
        self.version = "digest-1"
        self.images = 0
        self.batches = []

    def identity(self):
        return self.version

    def embed_image(self, payload):
        self.images += 1
        return np.arange(1, 769, dtype=np.float32)

    def embed_text(self, descriptions):
        self.batches.append(descriptions)
        return np.stack([np.arange(1, 769, dtype=np.float32)] * len(descriptions))


def test_edit_reuses_image_batches_only_missing_and_digest_invalidates(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (3, 3)).save(path)
    client = ControlledClient()
    engine = EmbeddingEngine(client)
    catalog = defaults()
    engine.prepare(catalog)
    engine.analyze(str(path), catalog)
    changed = catalog + [Category("Silla", "Una fotografía de una silla")]
    result = engine.analyze(str(path), changed)
    assert result.image_cached
    assert client.images == 1
    assert len(result.ranking) == 9
    assert client.batches[-1] == ["Una fotografía de una silla"]
    client.version = "digest-2"
    result = engine.analyze(str(path), changed)
    assert not result.image_cached
    assert client.images == 2
    assert len(client.batches[-1]) == 9
