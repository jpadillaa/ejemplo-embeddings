import base64
import io
import pytest
from PIL import Image
from visual_embeddings.image_processor import prepare_image, MAX_BYTES


@pytest.mark.parametrize("fmt,ext", [("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp")])
def test_formats_resize_rgb_and_original_preserved(tmp_path, fmt, ext):
    path = tmp_path / f"image.{ext}"
    Image.new("RGB", (2048, 1024), "red").save(path, fmt)
    original = path.read_bytes()
    result = prepare_image(path)
    assert result.size == (1024, 512)
    with Image.open(io.BytesIO(base64.b64decode(result.base64))) as image:
        assert image.mode == "RGB"
    assert path.read_bytes() == original
    assert prepare_image(path).digest == result.digest


def test_exif_orientation(tmp_path):
    path = tmp_path / "image.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (20, 10)).save(path, exif=exif)
    assert prepare_image(path).size == (10, 20)


def test_invalid_and_oversize(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="dañada"):
        prepare_image(path)
    path.write_bytes(b"0" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="10 MB"):
        prepare_image(path)
    gif = tmp_path / "hidden.png"
    Image.new("RGB", (2, 2)).save(gif, "GIF")
    with pytest.raises(ValueError, match="contenido"):
        prepare_image(gif)
