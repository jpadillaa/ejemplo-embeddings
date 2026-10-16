import pytest
from visual_embeddings.categories import defaults, parse_categories, rows


def test_defaults_and_edit():
    original = defaults()
    assert len(original) == 8
    assert "Silla" not in [c.label for c in original]
    updated = rows(original) + [["Silla", "Una fotografía de una silla"]]
    assert parse_categories(updated)[-1].label == "Silla"
    assert len(defaults()) == 8


@pytest.mark.parametrize("catalog", [[], [["", "text"]], [["x", ""]], [["Gato", "a"], ["gato", "b"]]])
def test_invalid(catalog):
    with pytest.raises(ValueError):
        parse_categories(catalog)
