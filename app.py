"""Ejecutar con uv run python app.py."""
import logging
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from visual_embeddings.interface import CSS, ROOT, build_app, presentation_theme
import gradio as gr

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_app().launch(server_name="127.0.0.1", server_port=7860, share=False,
        max_file_size="10mb", allowed_paths=[str(ROOT / "assets/examples")],
        theme=presentation_theme(),
        css=CSS, footer_links=[], ssr_mode=False)
