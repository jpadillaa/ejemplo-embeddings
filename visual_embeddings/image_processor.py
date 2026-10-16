"""Valida el archivo original y prepara una copia RGB en memoria."""
import base64
import hashlib
import io
import warnings
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class PreparedImage:
    base64: str
    digest: str
    size: tuple[int, int]


def prepare_image(path: str | Path) -> PreparedImage:
    source = Path(path)
    if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("Formatos admitidos: JPG, PNG y WEBP.")
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("La imagen supera el límite de 10 MB.")
    raw = source.read_bytes()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("El contenido del archivo no es JPG, PNG o WEBP.")
                image.load()
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format="PNG")
                payload = output.getvalue()
                return PreparedImage(base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest(), image.size)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("La imagen está dañada o excede los límites de seguridad de Pillow.") from exc
