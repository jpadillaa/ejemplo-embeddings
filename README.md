# Visual Embeddings Lab

Demo académica local de selección *zero-shot*: EmbeddingGemma 2 representa una fotografía y descripciones en un espacio compartido de 768 dimensiones; NumPy calcula similitud coseno y construye el ranking. No utiliza generación textual, clasificadores externos ni entrenamiento con las categorías del usuario.

## Requisitos e instalación

Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/) y [Ollama actualizado](https://ollama.com/download). Se recomienda macOS con Apple Silicon y memoria disponible para el modelo de aproximadamente 713 MB.

```bash
ollama serve
# En otra terminal:
ollama pull embeddinggemma-2:440m
uv sync
uv run python app.py
```

Abre http://127.0.0.1:7860. No se habilitan enlaces públicos. Ollama debe escuchar solamente en loopback. `OLLAMA_BASE_URL` permite cambiar el puerto local; se rechazan hosts externos. `.env.example` documenta esta variable; no se carga automáticamente.

La instalación inicial necesita Internet; después, dependencias, modelo, fotografías, fuentes y aplicación se ejecutan localmente. No hay llamadas de inferencia externas, telemetría de Gradio ni uso del Hub. No se usa Docker.

**Compatibilidad comprobada:** Ollama 0.40.5. La instalación encontrada inicialmente (0.35.0) rechazó el modelo con HTTP 412 y exigió actualización. Para validar sin reemplazar el software del sistema se descargó el binario oficial 0.40.5 en `.runtime/ollama/` (ignorado). Durante esta sesión se inició con:

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 .runtime/ollama/ollama serve
```

En otro equipo, instala la versión actual oficial. `.runtime` no es parte del código entregable ni un requisito de la aplicación.

## Uso en clase

1. Espera el estado **Listo** y selecciona la miniatura del gato.
2. Pulsa **Analizar imagen** y observa las similitudes originales, sin porcentajes.
3. Abre **Explorar vectores**: incluye los dos vectores completos y una muestra de sus primeros 16 componentes.
4. Selecciona la silla y analiza con el catálogo original de ocho categorías.
5. En **Configurar categorías**, escribe «Silla» y «Una fotografía de una silla»; pulsa **Agregar y comparar**.
6. Observa el nuevo ranking y en **Información técnica** comprueba que el vector visual fue reutilizado.

Puedes editar nombres y descripciones en la tabla y pulsar **Aplicar cambios y comparar**. Para eliminar una categoría, escribe su nombre en el campo correspondiente y pulsa **Eliminar categoría**. **Restablecer originales** recupera las ocho categorías y recalcula. Las modificaciones son por sesión y no se guardan en disco. Para cambiar el catálogo inicial de forma permanente, edita `visual_embeddings/categories.json` y reinicia.

## Pipeline y cachés

Pillow valida el archivo original (JPG, PNG, WEBP; 10 MB), corrige EXIF y prepara una copia RGB proporcional de máximo 1024 px. Se envía a `POST /api/embed` como `input: {"image": "BASE64"}`. Las descripciones se envían sin prefijos en una solicitud por lote que incluye solamente las entradas nuevas.

Cada respuesta debe contener el número esperado de vectores de 768 componentes, finitos y de norma positiva. Se normalizan con NumPy y se multiplican los vectores textuales por el vector visual. El gráfico conserva las puntuaciones obtenidas; no las transforma en porcentajes.

La caché en memoria conserva hasta cuatro vectores visuales y 512 descripciones por sesión. Antes de cada análisis verifica el digest del modelo y la versión del servidor; si cambian, invalida ambas cachés. El modelo permanece cargado hasta 30 minutos desde su último uso. Al abrir la interfaz se precalculan las categorías. **Limpiar** borra la selección y los vectores visuales de la sesión.

Las fotografías no se guardan como datos permanentes del proyecto. Gradio utiliza archivos temporales para cargas y vistas previas; se configura su limpieza periódica (cada 60 segundos se eliminan archivos con más de una hora de antigüedad, para conservar la selección durante la clase). Los vectores de sesión expiran después de una hora. Las fotos originales seleccionadas por el usuario no se modifican.

## Pruebas y validación real

```bash
uv run pytest -q
RUN_OLLAMA_INTEGRATION=1 uv run pytest -q
uv run python scripts/validate_model.py
```

La primera orden usa datos controlados y omite explícitamente la integración. La segunda requiere Ollama y el modelo real. La tercera prueba las tres fotografías, agrega Silla, verifica la reutilización y escribe evidencia real en `validation.json`. Los datos controlados existen únicamente en pruebas; la aplicación nunca presenta resultados simulados.

Validación realizada el 16 de octubre de 2026 con el digest registrado en `validation.json`:

| Imagen | Mejor coincidencia | Similitud |
|---|---|---|
| Gato | Gato | 0.688 |
| Bicicleta | Bicicleta | 0.697 |
| Silla, catálogo original | Planta | 0.601 |
| Silla, categoría agregada | Silla | 0.684 |

Las fotos y su licencia están en `assets/examples/ATTRIBUTION.md`.

## Limitaciones y errores

La similitud coseno no equivale a una probabilidad de clasificación. Siempre se selecciona una categoría disponible, incluso si el objeto real no está representado. No hay detección de objetos múltiples, calibración ni umbral universal. Descripciones, fondos, idiomas y fotografías afectan al ranking; estos ejemplos no constituyen una evaluación general del modelo.

Si Ollama está apagado, inicia `ollama serve` y pulsa **Verificar conexión**. Si falta el modelo, ejecuta `ollama pull embeddinggemma-2:440m`. Si Ollama rechaza las imágenes o el modelo requiere otra versión, actualiza el servidor y vuelve a validar. El tiempo de espera de inferencia es de 120 segundos; un fallo limpia los resultados y permite repetir. Una respuesta inválida interrumpe el análisis. Los registros no incluyen fotos, vectores ni Base64.

## Fuentes técnicas

- [Modelo oficial Ollama](https://ollama.com/library/embeddinggemma-2)
- [API de embeddings multimodales](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings)
- [Google: embeddings multimodales](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
