import numpy as np
from visual_embeddings.categories import Category
from visual_embeddings.classifier import Match
from visual_embeddings.embedding_engine import Result
from visual_embeddings.interface import display, plot_ranking, run_analysis, build_app


def result():
    categories = [Category("Gato", "Una fotografía de un gato"), Category("Perro", "Una fotografía de un perro")]
    return Result([Match(categories[0], .321456, 0), Match(categories[1], -.23456, 1)],
                  np.ones(768), np.ones((2, 768)), .01, .001, .02, False)


def test_chart_preserves_real_scores_descriptions_and_order():
    figure = plot_ranking(result())
    assert list(figure.data[0].x) == [.321456, -.23456]
    assert list(figure.data[0].y) == ["Gato", "Perro"]
    assert list(figure.data[0].text) == ["0.321", "-0.235"]
    assert figure.data[0].customdata[0] == "Una fotografía de un gato"
    outputs = display(result())
    assert "Resultado: Gato" in outputs[0]
    assert len(outputs[5].data[0].z[0]) == 16
    assert "768 dimensiones" in outputs[4]


def test_error_clears_all_results():
    final = list(run_analysis(None, [["Gato", "Un gato"]], "test"))[-1]
    assert "Error" in final[0]
    assert final[2] is None
    assert final[5] == ""
    assert final[6] is None


def test_build_ui():
    app = build_app()
    assert app.config["analytics_enabled"] is False
    assert len(app.config["components"]) > 30
