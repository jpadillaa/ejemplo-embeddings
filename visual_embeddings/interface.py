"""Interfaz Gradio. Todo resultado visible procede del motor local."""
import html
import logging
import uuid
from pathlib import Path
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from .categories import defaults, parse_categories, rows
from .embedding_engine import EmbeddingEngine, Result
from .image_processor import prepare_image
from .ollama_client import OllamaError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [(str(ROOT / f"assets/examples/{name}.jpg"), label) for name, label in
            [("gato", "01 · Gato"), ("bicicleta", "02 · Bicicleta"), ("silla", "03 · Fuera del catálogo")]]
SESSIONS: dict[str, EmbeddingEngine] = {}
logger = logging.getLogger(__name__)

CSS = """
.gradio-container {max-width:1320px!important; margin:auto!important}
body {background:#f4f6f8!important}
#hero {padding:28px 0 18px;border-bottom:1px solid #dce3e9;margin-bottom:12px}
.eyebrow {font:600 12px monospace;letter-spacing:2px;color:#52776b}
#hero h1 {font-size:38px;letter-spacing:-1.6px;margin:10px 0;color:#132e34}
#hero p {color:#53676e;font-size:17px;max-width:850px;line-height:1.5}
.tag {display:inline-block;border:1px solid #cbdad5;border-radius:20px;padding:6px 12px;margin:10px 6px 0 0;color:#426258;font-size:12px}
.result-card {background:#133d35;border-radius:16px;padding:24px 28px;color:#f6fbf7;min-height:145px}
.result-card h2 {font-size:32px;margin:6px 0;color:white}
.result-card p {margin:6px 0;color:#d6e8df}
.result-card > span {color:#d6e8df}
.result-card .score {font:600 25px monospace;color:#c4e9aa}
.pipeline {display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}
.lane {display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.stage {background:#edf1f4;border:1px solid #d7e0e4;border-radius:10px;padding:15px;font-size:14px;flex:1;text-align:center;color:#455d66}
.done .stage {background:#e5f1e7;border-color:#9abda4;color:#204e39}
.merge {grid-column:1/-1;display:flex;align-items:center;gap:12px;justify-content:center}
.merge .stage {flex:0 1 320px}
.caption {font-size:13px;color:#61727b}
@media(max-width:700px){#hero h1{font-size:29px}.pipeline{grid-template-columns:1fr}.merge{grid-column:1}.stage{padding:10px}}
"""


def pipeline(done: bool = False) -> str:
    mark = "✓ " if done else ""
    return f'''<div class="pipeline {'done' if done else ''}" aria-label="Flujo de embeddings">
    <div class="lane"><div class="stage">{mark}IMAGEN</div>→<div class="stage">{mark}EmbeddingGemma 2</div>→<div class="stage">{mark}Vector imagen [768]</div></div>
    <div class="lane"><div class="stage">{mark}DESCRIPCIONES</div>→<div class="stage">{mark}EmbeddingGemma 2</div>→<div class="stage">{mark}Vectores texto [768]</div></div>
    <div class="merge">↘<div class="stage">{mark}Similitud coseno · 768 componentes</div>→<div class="stage">{mark}Ranking</div>↙</div></div>
    <p class="caption">Un espacio vectorial compartido. El modelo produce vectores; NumPy compara y ordena. No se genera una respuesta textual.</p>'''


def empty_card() -> str:
    return '<div class="result-card"><span>CATEGORÍA MÁS SIMILAR</span><h2>Tu imagen, en vectores</h2><p>Selecciona una fotografía y ejecuta el análisis.</p></div>'


def plot_ranking(result: Result) -> go.Figure:
    matches = result.ranking
    figure = go.Figure(go.Bar(
        x=[m.score for m in matches], y=[m.category.label for m in matches], orientation="h",
        marker_color=["#326e52"] + ["#bcced3"] * (len(matches) - 1),
        text=[f"{m.score:.3f}" for m in matches], textposition="outside", cliponaxis=False,
        customdata=[html.escape(m.category.description) for m in matches],
        hovertemplate="%{y}<br>%{customdata}<br>Similitud: %{x:.3f}<extra></extra>",
    ))
    figure.update_layout(template="plotly_white", height=max(320, len(matches) * 37 + 85),
        margin=dict(l=10, r=60, t=15, b=45), font=dict(size=15, family="Arial", color="#243f45"),
        xaxis=dict(title="Similitud coseno", range=[min(0, min(m.score for m in matches)) - .04, max(m.score for m in matches) + .16], zeroline=True),
        yaxis=dict(autorange="reversed"), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    return figure


def display(result: Result) -> tuple:
    best = result.ranking[0]
    card = f'''<div class="result-card"><span>CATEGORÍA MÁS SIMILAR</span><h2>Resultado: {html.escape(best.category.label)}</h2>
    <p>Entre las {len(result.ranking)} opciones disponibles.</p><span class="score">{best.score:.3f}</span> · similitud coseno</div>'''
    info = f"""| Métrica | Valor |
|---|---|
| Modelo | EmbeddingGemma 2 · 440M |
| Modalidad de entrada | Imagen |
| Dimensiones | 768 |
| Categorías comparadas | {len(result.ranking)} |
| Tiempo de procesamiento | {result.total_seconds:.3f} s |
| Embedding visual | {result.visual_seconds:.3f} s · {'reutilizado de caché' if result.image_cached else 'inferencia real'} |
| Cálculo de similitud | {result.cosine_seconds * 1000:.3f} ms |
| Métrica | Similitud coseno |"""
    selected = result.texts[best.index]
    vectors = ("Embedding de la imagen · 768 dimensiones\n" + np.array2string(result.image, threshold=768, precision=5) +
               f"\n\nEmbedding de {best.category.label} · 768 dimensiones\n" + np.array2string(selected, threshold=768, precision=5))
    sample = np.stack([result.image[:16], selected[:16]])
    bound = max(float(np.abs(sample).max()), 1e-9)
    heatmap = go.Figure(go.Heatmap(z=sample.tolist(), x=list(range(1, 17)), y=["Imagen", best.category.label],
        colorscale="BrBG", zmin=-bound, zmax=bound, zmid=0,
        hovertemplate="%{y} · componente %{x}<br>%{z:.5f}<extra></extra>"))
    heatmap.update_layout(template="plotly_white", height=220, margin=dict(l=10, r=10, t=15, b=30), xaxis=dict(dtick=1), font=dict(size=14))
    return card, plot_ranking(result), info, pipeline(True), vectors, heatmap


def engine_for(session: str) -> EmbeddingEngine:
    if session not in SESSIONS:
        SESSIONS[session] = EmbeddingEngine()
    return SESSIONS[session]


def release(session: str) -> None:
    engine = SESSIONS.pop(session, None)
    if engine:
        engine.close()


def initial_outputs(status: str) -> tuple:
    return status, empty_card(), None, "", pipeline(), "", None


def run_analysis(path: str | None, catalog: list, session: str):
    yield initial_outputs("**Procesando** · Preparando vectores y comparando categorías…")
    try:
        categories = parse_categories(catalog)
        if not path:
            raise ValueError("Selecciona una fotografía antes de analizar.")
        result = engine_for(session).analyze(path, categories)
        yield ("**Completado** · Mejor coincidencia dentro del catálogo disponible.", *display(result))
    except (OllamaError, ValueError, OSError) as exc:
        logger.warning("Análisis interrumpido: %s", exc)
        yield initial_outputs(f"**Error** · {html.escape(str(exc))}")


def initialize(catalog: list, session: str):
    yield "**Inicializando** · Verificando Ollama y disponibilidad del modelo…", gr.update(interactive=False)
    try:
        engine = engine_for(session)
        engine.refresh()
        yield "**Preparando** · Generando embeddings de las categorías…", gr.update(interactive=False)
        engine.texts(parse_categories(catalog))
        yield "**Listo** · Ollama conectado. Selecciona una imagen.", gr.update(interactive=True)
    except (OllamaError, ValueError) as exc:
        yield f"**Error** · {html.escape(str(exc))}", gr.update(interactive=False)


def apply_catalog(path: str | None, catalog: list, session: str):
    if path:
        yield from run_analysis(path, catalog, session)
    else:
        yield initial_outputs("**Preparando** · Actualizando embeddings textuales…")
        try:
            engine_for(session).prepare(parse_categories(catalog))
            yield initial_outputs("**Listo** · Categorías actualizadas sin entrenamiento. Selecciona una imagen.")
        except (OllamaError, ValueError) as exc:
            yield initial_outputs(f"**Error** · {html.escape(str(exc))}")


def image_changed(path: str | None):
    if path:
        try:
            prepare_image(path)
        except (ValueError, OSError) as exc:
            return None, *initial_outputs(f"**Error** · {html.escape(str(exc))}")
    return path, *initial_outputs("**Listo** · Imagen seleccionada; ejecuta el análisis." if path else "**Listo** · Selecciona una imagen.")


def add_category(catalog: list, label: str, description: str):
    try:
        updated = rows(parse_categories(catalog + [[label, description]]))
        if not label.strip() or not description.strip():
            raise ValueError("Escribe el nombre y la descripción de la categoría.")
        return updated, "", ""
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Visual Embeddings Lab", analytics_enabled=False, delete_cache=(60, 3600)) as demo:
        session = gr.State(value=lambda: uuid.uuid4().hex, time_to_live=3600, delete_callback=release)
        gr.HTML('''<header id="hero"><span class="eyebrow">LABORATORIO MULTIMODAL / 01</span><h1>Visual Embeddings Lab<span style="color:#54836b">.</span></h1>
        <p>De una fotografía a un espacio compartido. Explora cómo una imagen se relaciona con palabras mediante similitud matemática.</p>
        <span class="tag">EmbeddingGemma 2 · 440M</span><span class="tag">768 dimensiones</span><span class="tag">Inferencia local</span></header>''')
        with gr.Row():
            status = gr.Markdown("**Inicializando** · Verificando el servicio local…", scale=5)
            retry = gr.Button("Verificar conexión", size="sm", scale=1)
        with gr.Row(equal_height=False):
            with gr.Column(scale=45, min_width=340):
                gr.Markdown("### 01 / Fotografía de entrada")
                upload = gr.File(label="Arrastra una fotografía o haz clic para cargar", file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
                preview = gr.Image(label="Vista previa", interactive=False, height=300, buttons=[])
                gr.Markdown("JPG, PNG o WEBP · máximo 10 MB", elem_classes="caption")
                with gr.Row():
                    analyze = gr.Button("Analizar imagen", variant="primary", interactive=False)
                    clear = gr.Button("Limpiar")
                gr.Markdown("**Prueba una fotografía de ejemplo**")
                gallery = gr.Gallery(value=EXAMPLES, columns=3, rows=1, height=160, object_fit="cover", show_label=False, preview=False, allow_preview=False, interactive=False, buttons=[])
                gr.Markdown("Fotos: [Unsplash](https://unsplash.com/license) · atribuciones en assets/examples/ATTRIBUTION.md", elem_classes="caption")
            with gr.Column(scale=55, min_width=340):
                gr.Markdown("### 02 / Comparación semántica")
                card = gr.HTML(empty_card())
                gr.Markdown("**Ranking de similitud**")
                chart = gr.Plot(label="Ranking completo", show_label=False)
                with gr.Accordion("Información técnica", open=False):
                    metrics = gr.Markdown()
                gr.Markdown("La mejor coincidencia depende de las categorías disponibles. La similitud coseno no equivale a una probabilidad.", elem_classes="caption")
        gr.Markdown("### ¿Cómo funciona?")
        process = gr.HTML(pipeline())
        with gr.Accordion("Explorar vectores", open=False):
            gr.Markdown("**Primeros 16 componentes** · Esta muestra es visual. El cálculo utiliza los **768 componentes** de cada vector.")
            heatmap = gr.Plot(label="Muestra de componentes")
            gr.Markdown(r"$$\operatorname{sim}(u,v)=\frac{u\cdot v}{\|u\|\,\|v\|}$$")
            vector_text = gr.Textbox(label="Vectores reales completos", lines=8, max_lines=16, interactive=False)
        with gr.Accordion("Configurar categorías", open=True):
            gr.Markdown("Edita las celdas o agrega una categoría. Para eliminar, usa **Eliminar categoría**. Aplica los cambios para recalcular; se reutiliza el vector visual cuando la imagen permanece igual.")
            catalog = gr.Dataframe(value=rows(defaults()), headers=["Categoría", "Descripción textual"], datatype="str", type="array", column_count=2, interactive=True, wrap=True, column_widths=["25%", "75%"], label="Catálogo de comparación")
            with gr.Row():
                label = gr.Textbox(label="Nueva categoría", placeholder="Silla", scale=1)
                description = gr.Textbox(label="Descripción", placeholder="Una fotografía de una silla", scale=3)
                add = gr.Button("Agregar y comparar", scale=1)
            with gr.Row():
                remove_label = gr.Textbox(label="Categoría a eliminar", placeholder="Nombre exacto de la categoría")
                remove = gr.Button("Eliminar categoría")
                apply = gr.Button("Aplicar cambios y comparar", variant="primary")
                reset = gr.Button("Restablecer originales")
        outputs = [status, card, chart, metrics, process, vector_text, heatmap]
        inputs = [upload, catalog, session]
        demo.load(initialize, [catalog, session], [status, analyze])
        retry.click(initialize, [catalog, session], [status, analyze])
        analyze.click(run_analysis, inputs, outputs)
        apply.click(apply_catalog, inputs, outputs)
        upload.change(image_changed, upload, [preview, *outputs])
        def select_example(event: gr.SelectData):
            return EXAMPLES[event.index][0]
        gallery.select(select_example, outputs=upload)
        clear.click(lambda: None, outputs=upload).then(lambda s: engine_for(s).image_cache.clear(), session)
        add.click(add_category, [catalog, label, description], [catalog, label, description]).success(apply_catalog, inputs, outputs)
        def delete_category(table, name):
            categories = parse_categories(table)
            updated = [c for c in categories if c.label.casefold() != name.strip().casefold()]
            if len(updated) == len(categories):
                raise gr.Error("No existe una categoría con ese nombre.")
            if not updated:
                raise gr.Error("Debe permanecer al menos una categoría.")
            return rows(updated), ""
        remove.click(delete_category, [catalog, remove_label], [catalog, remove_label]).success(apply_catalog, inputs, outputs)
        reset.click(lambda: rows(defaults()), outputs=catalog).then(apply_catalog, inputs, outputs)
        catalog.input(lambda: initial_outputs("**Catálogo modificado** · Aplica los cambios para actualizar el ranking."), outputs=outputs)
    return demo.queue(default_concurrency_limit=1)


def presentation_theme():
    """Paleta clara consistente para proyección, también con preferencia OS oscura."""
    theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", font=["Arial", "sans-serif"], font_mono=["Courier New", "monospace"])
    values = theme.to_dict()["theme"]
    theme.set(**{key: values[key[:-5]] for key in values if key.endswith("_dark") and key[:-5] in values})
    return theme
