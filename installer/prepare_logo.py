"""Extract the user's sigma emblem without redrawing its pixels.

One-time artwork preparation, not an installer/runtime dependency. The original
PNG is retained. Requires Pillow, NumPy and SciPy from the development runtime.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent


def extract_emblem(source: Path, destination: Path):
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float64)
    components, _ = ndimage.label(rgb.min(axis=2) < 200)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    # The navy sigma body is the largest connected component. Its enclosed
    # white outlines and teal cells belong to the emblem; the lower wordmark
    # consists of separate components and is excluded.
    body = components == sizes.argmax()
    mask = ndimage.binary_fill_holes(body)
    interior = ndimage.binary_erosion(mask, iterations=2)
    distance, nearest = ndimage.distance_transform_edt(~interior, return_indices=True)
    band = (distance <= 4) & ~interior
    alpha = mask.astype(np.float64)
    nearby_color = rgb[nearest[0], nearest[1]]
    white_delta = 255 - nearby_color
    denominator = np.sum(white_delta * white_delta, axis=2)
    coverage = np.divide(np.sum((255 - rgb) * white_delta, axis=2), denominator,
                         out=np.zeros_like(alpha), where=denominator > 0)
    alpha[band] = np.clip(coverage[band], 0, 1)
    alpha[alpha < .02] = 0
    # Remove the original white matte at anti-aliased exterior pixels. Interior
    # artwork stays opaque and byte-identical, including the white separators.
    output = rgb.copy()
    edges = (alpha > 0) & (alpha < 1)
    output[edges] = (rgb[edges] - 255 * (1 - alpha[edges, None])) / alpha[edges, None]
    output[alpha == 0] = 0
    rgba = np.dstack((np.clip(np.rint(output), 0, 255).astype(np.uint8),
                      np.rint(alpha * 255).astype(np.uint8)))
    image = Image.fromarray(rgba)
    bbox = image.getbbox()
    if bbox is None or bbox[3] > rgb.shape[0] * .8:
        raise ValueError("Emblem extraction included the lower wordmark")
    emblem = image.crop(bbox)
    side = max(1024, emblem.width + 128, emblem.height + 128)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    # Paste without a mask so alpha is copied once, not multiplied twice.
    canvas.paste(emblem, ((side - emblem.width) // 2, (side - emblem.height) // 2))
    canvas.save(destination)
    print(f"Saved {destination}: {canvas.size}, emblem bounds in original: {bbox}")


if __name__ == "__main__":
    extract_emblem(HERE / "assets/sigma-logo-original.png", HERE / "assets/sigma-logo.png")
