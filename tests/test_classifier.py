import numpy as np
import pytest
from visual_embeddings.classifier import cosine_scores, classify
from visual_embeddings.categories import Category


def test_cosine_matches_reference_and_ranking():
    image = np.array([2., -1., 3.])
    texts = np.array([[1., 0., 0.], [-2., 1., -3.], [2., -1., 3.]])
    expected = [np.dot(image, t) / (np.linalg.norm(image) * np.linalg.norm(t)) for t in texts]
    np.testing.assert_allclose(cosine_scores(image, texts), expected)
    matches = classify(image, texts, [Category(str(i), str(i)) for i in range(3)])
    assert [m.index for m in matches] == [2, 0, 1]
    assert matches[1].score == pytest.approx(expected[0])


@pytest.mark.parametrize("image,texts", [([0, 0], [[1, 1]]), ([1, 1], [[0, 0]]), ([1, np.nan], [[1, 1]]), ([1, 1], [[1]])])
def test_invalid_vectors(image, texts):
    with pytest.raises(ValueError):
        cosine_scores(image, texts)
