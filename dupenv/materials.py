# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Procedural, tileable textures (asphalt, grass, bark, leaves, facades...).

Everything is generated with numpy (periodic FFT noise), so no texture files
need to be downloaded and every texture tiles without seams.
"""
import os

import numpy as np
from PIL import Image, ImageDraw


def _noise(n, beta, rng, aniso=(1.0, 1.0)):
    """Periodic 1/f^beta noise in [0, 1]."""
    w = rng.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(f * aniso[0], f * aniso[1], indexing="ij")
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0
    img = np.real(np.fft.ifft2(np.fft.fft2(w) / r ** beta))
    img -= img.min()
    return img / max(img.max(), 1e-9)


def _rgb(base, *layers):
    out = np.ones(layers[0][0].shape + (3,)) * np.asarray(base, float)
    for layer, amp in layers:
        out += (layer[..., None] - 0.5) * np.asarray(amp, float)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def asphalt(rng, n=1024):
    grain = _noise(n, 0.3, rng)
    patches = _noise(n, 1.6, rng)
    img = _rgb((0.24, 0.24, 0.25), (grain, (0.16, 0.16, 0.16)), (patches, (0.06, 0.06, 0.05)))
    a = np.asarray(img).astype(float)
    speck = rng.random((n, n)) > 0.985
    a[speck] = np.clip(a[speck] * 1.5 + 25, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def grass(rng, n=1024):
    return _rgb((0.30, 0.40, 0.18), (_noise(n, 0.4, rng), (0.12, 0.14, 0.06)),
                (_noise(n, 1.8, rng), (0.10, 0.10, 0.05)))


def bark(rng, n=512):
    streaks = _noise(n, 1.2, rng, aniso=(0.08, 1.0))
    return _rgb((0.30, 0.22, 0.15), (streaks, (0.22, 0.17, 0.12)), (_noise(n, 0.4, rng), (0.08, 0.06, 0.05)))


def leaves(rng, n=512, base=(0.20, 0.38, 0.12)):
    clumps = _noise(n, 1.0, rng)
    fine = _noise(n, 0.2, rng)
    return _rgb(base, (clumps, (0.20, 0.26, 0.10)), (fine, (0.10, 0.14, 0.06)))


def concrete(rng, n=512):
    return _rgb((0.64, 0.63, 0.60), (_noise(n, 0.5, rng), (0.10, 0.10, 0.10)), (_noise(n, 1.8, rng), (0.08, 0.08, 0.07)))


def gravel(rng, n=512):
    return _rgb((0.48, 0.44, 0.38), (_noise(n, 0.1, rng), (0.35, 0.33, 0.30)))


def pavers(rng, n=512):
    img = _rgb((0.62, 0.60, 0.56), (_noise(n, 0.5, rng), (0.10, 0.10, 0.10)))
    d = ImageDraw.Draw(img)
    step = n // 8
    for k in range(0, n, step):
        d.line([(0, k), (n, k)], fill=(95, 92, 88), width=3)
        d.line([(k, 0), (k, n)], fill=(95, 92, 88), width=3)
    return img


def facade(rng, wall, n=512, window=(40, 52, 64)):
    """6 m x 6 m of wall: two floors, two windows per floor (u along, v up)."""
    base = np.asarray(wall, float)
    img = _rgb(base, (_noise(n, 0.6, rng), (0.08, 0.08, 0.08)))
    d = ImageDraw.Draw(img)
    for fl in range(2):
        for w in range(2):
            x0 = int((w * 3 + 0.9) / 6 * n)
            x1 = int((w * 3 + 2.1) / 6 * n)
            # image rows go down, v goes up: floor 0 at the bottom
            y1 = n - int((fl * 3 + 0.9) / 6 * n)
            y0 = n - int((fl * 3 + 2.3) / 6 * n)
            d.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], fill=(215, 212, 205))
            d.rectangle([x0, y0, x1, y1], fill=window)
            d.line([((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1)], fill=(200, 200, 200), width=4)
    return img


def meadow(rng, n=1024):
    img = _rgb((0.40, 0.46, 0.20), (_noise(n, 0.35, rng), (0.16, 0.16, 0.08)),
               (_noise(n, 1.5, rng), (0.14, 0.12, 0.06)))
    a = np.asarray(img).astype(float)
    flowers = rng.random((n, n)) > 0.996
    a[flowers] = [225, 215, 120]
    return Image.fromarray(a.astype(np.uint8))


def lawn(rng, n=1024):
    return _rgb((0.26, 0.44, 0.16), (_noise(n, 0.25, rng), (0.10, 0.14, 0.05)),
                (_noise(n, 1.6, rng), (0.06, 0.08, 0.03)))


def rows(rng, n=1024, base=(0.36, 0.50, 0.18), between=(0.40, 0.33, 0.22), k=24):
    """Crop rows (or furrows): stripes along v with noise."""
    v = np.linspace(0, 2 * np.pi * k, n, endpoint=False)
    stripe = (0.5 + 0.5 * np.sin(v))[None, :] ** 2
    stripe = np.repeat(stripe, n, axis=0)
    a = np.asarray(base)[None, None] * stripe[..., None] + np.asarray(between)[None, None] * (1 - stripe[..., None])
    a += (_noise(n, 0.4, rng)[..., None] - 0.5) * 0.12
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def forest_floor(rng, n=512):
    return _rgb((0.26, 0.24, 0.14), (_noise(n, 0.5, rng), (0.18, 0.16, 0.08)),
                (_noise(n, 1.4, rng), (0.10, 0.14, 0.05)))


def dirt(rng, n=512):
    return _rgb((0.50, 0.43, 0.33), (_noise(n, 0.4, rng), (0.16, 0.14, 0.12)),
                (_noise(n, 1.6, rng), (0.10, 0.08, 0.06)))


def paving(rng, n=512):
    img = _rgb((0.55, 0.54, 0.52), (_noise(n, 0.5, rng), (0.10, 0.10, 0.10)))
    d = ImageDraw.Draw(img)
    for k in range(0, n, n // 4):
        d.line([(0, k), (n, k)], fill=(110, 108, 104), width=2)
        d.line([(k, 0), (k, n)], fill=(110, 108, 104), width=2)
    return img


def facade_style(rng, style, wall, n=512):
    """6 x 6 m of facade in a given style (two storeys, u along, v up)."""
    wall = np.asarray(wall, float)
    if style == "brick":
        img = _rgb(wall, (_noise(n, 0.3, rng), (0.10, 0.06, 0.05)))
        d = ImageDraw.Draw(img)
        for yy in range(0, n, 12):
            d.line([(0, yy), (n, yy)], fill=(170, 160, 150), width=2)
            off = 0 if (yy // 12) % 2 else 12
            for xx in range(off, n, 24):
                d.line([(xx, yy), (xx, yy + 12)], fill=(170, 160, 150), width=2)
    elif style == "metal":
        img = _rgb(wall, (_noise(n, 1.0, rng, aniso=(0.02, 1.0)), (0.06, 0.06, 0.06)))
        d = ImageDraw.Draw(img)
        for xx in range(0, n, 16):
            d.line([(xx, 0), (xx, n)], fill=tuple(int(c * 190) for c in wall), width=4)
        d.rectangle([n * 0.1, n * 0.62, n * 0.9, n * 0.72], fill=(60, 70, 80))  # ribbon window
        return img
    elif style == "glass":
        img = _rgb((0.30, 0.42, 0.52), (_noise(n, 1.5, rng), (0.10, 0.12, 0.14)))
        d = ImageDraw.Draw(img)
        for k in range(0, n, n // 6):
            d.line([(k, 0), (k, n)], fill=(190, 195, 200), width=5)
        for k in range(0, n, n // 4):
            d.line([(0, k), (n, k)], fill=(190, 195, 200), width=7)
        return img
    else:
        img = _rgb(wall, (_noise(n, 0.6, rng), (0.08, 0.08, 0.08)))
    d = ImageDraw.Draw(img)
    shutters = style in ("render", "stone")
    for fl in range(2):
        for w in range(2):
            x0 = int((w * 3 + 0.9) / 6 * n)
            x1 = int((w * 3 + 2.1) / 6 * n)
            y1 = n - int((fl * 3 + 0.9) / 6 * n)
            y0 = n - int((fl * 3 + 2.3) / 6 * n)
            d.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], fill=(220, 216, 208))
            d.rectangle([x0, y0, x1, y1], fill=(38, 50, 62))
            d.line([((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1)], fill=(205, 205, 205), width=4)
            if shutters:
                col = [(70, 110, 130), (130, 60, 50), (90, 110, 70), (200, 200, 195)][rng.integers(4)]
                sw = (x1 - x0) // 2
                d.rectangle([x0 - 8 - sw, y0, x0 - 8, y1], fill=col)
                d.rectangle([x1 + 8, y0, x1 + 8 + sw, y1], fill=col)
    return img


def roof_tiles(rng, n=512, base=(0.62, 0.30, 0.20)):
    img = _rgb(base, (_noise(n, 0.3, rng), (0.10, 0.06, 0.04)))
    d = ImageDraw.Draw(img)
    for yy in range(0, n, 16):
        d.line([(0, yy), (n, yy)], fill=tuple(int(c * 150) for c in base), width=3)
    return img


def roof(rng, n=512):
    return _rgb((0.42, 0.30, 0.26), (_noise(n, 0.9, rng, aniso=(1.0, 0.15)), (0.15, 0.10, 0.08)),
                (_noise(n, 0.3, rng), (0.08, 0.06, 0.05)))


def water(rng, n=512):
    return _rgb((0.16, 0.28, 0.36), (_noise(n, 1.2, rng), (0.08, 0.10, 0.10)))


TEXTURES = {
    "M_Grass": (lawn, (0.26, 0.44, 0.16)),
    "M_Meadow": (meadow, (0.40, 0.46, 0.20)),
    "M_Crop": (lambda r: rows(r), (0.36, 0.48, 0.18)),
    "M_Field": (lambda r: rows(r, base=(0.52, 0.42, 0.30), between=(0.40, 0.31, 0.22), k=40), (0.48, 0.40, 0.28)),
    "M_ForestFloor": (forest_floor, (0.26, 0.24, 0.14)),
    "M_Paving": (paving, (0.55, 0.54, 0.52)),
    "M_Dirt": (dirt, (0.50, 0.43, 0.33)),
    "M_RoofTiles": (roof_tiles, (0.62, 0.30, 0.20)),
    "M_RoofSlate": (lambda r: roof_tiles(r, base=(0.30, 0.32, 0.35)), (0.30, 0.32, 0.35)),
    "M_RoofFlat": (lambda r: _rgb((0.40, 0.40, 0.40), (_noise(512, 0.2, r), (0.12, 0.12, 0.12))), (0.40, 0.40, 0.40)),
    "M_RoofMetal": (lambda r: _rgb((0.62, 0.64, 0.66), (_noise(512, 1.0, r, aniso=(1.0, 0.02)), (0.08, 0.08, 0.08))),
                    (0.62, 0.64, 0.66)),
    "M_Asphalt": (asphalt, (0.22, 0.22, 0.23)),
    "M_Terrain": (grass, (0.33, 0.42, 0.22)),
    "M_Bark": (bark, (0.33, 0.24, 0.16)),
    "M_Leaves": (lambda r: leaves(r), (0.22, 0.40, 0.14)),
    "M_LeavesDark": (lambda r: leaves(r, base=(0.13, 0.28, 0.09)), (0.14, 0.30, 0.10)),
    "M_Conifer": (lambda r: leaves(r, base=(0.08, 0.24, 0.12)), (0.10, 0.28, 0.14)),
    "M_Bush": (lambda r: leaves(r, base=(0.24, 0.38, 0.14)), (0.28, 0.42, 0.18)),
    "M_Concrete": (concrete, (0.66, 0.65, 0.62)),
    "M_Gravel": (gravel, (0.52, 0.47, 0.40)),
    "M_Ballast": (gravel, (0.45, 0.42, 0.38)),
    "M_Sidewalk": (pavers, (0.60, 0.58, 0.55)),
    "M_Roof": (roof, (0.45, 0.40, 0.38)),
    "M_Water": (water, (0.18, 0.30, 0.38)),
}

FACADES = {
    "M_WallDefault": (0.82, 0.78, 0.72), "M_Wall_house": (0.88, 0.82, 0.70),
    "M_Wall_residential": (0.86, 0.80, 0.68), "M_Wall_detached": (0.90, 0.84, 0.72),
    "M_Wall_apartments": (0.80, 0.78, 0.74), "M_Wall_commercial": (0.74, 0.76, 0.80),
    "M_Wall_retail": (0.80, 0.78, 0.76), "M_Wall_industrial": (0.70, 0.72, 0.74),
    "M_Wall_warehouse": (0.72, 0.73, 0.74), "M_Wall_church": (0.84, 0.80, 0.70),
    "M_Wall_train_station": (0.82, 0.78, 0.70), "M_Wall_office": (0.72, 0.76, 0.82),
}


# facade styles: (name, style, wall colour)
FACADE_STYLES = [
    ("render_cream", "render", (0.90, 0.85, 0.72)), ("render_white", "render", (0.93, 0.92, 0.88)),
    ("render_ochre", "render", (0.86, 0.72, 0.52)), ("render_grey", "plain", (0.78, 0.78, 0.76)),
    ("stone", "stone", (0.80, 0.75, 0.64)), ("brick_red", "brick", (0.62, 0.32, 0.24)),
    ("brick_brown", "brick", (0.50, 0.34, 0.26)), ("concrete", "plain", (0.72, 0.71, 0.68)),
    ("glass", "glass", (0.30, 0.42, 0.52)), ("metal_grey", "metal", (0.70, 0.72, 0.74)),
    ("metal_blue", "metal", (0.45, 0.55, 0.66)), ("metal_beige", "metal", (0.80, 0.76, 0.66)),
]


def register(scene, tex_dir, seed=3, textured=True):
    """Create the texture files and register textured materials on `scene`."""
    rng = np.random.default_rng(seed)
    os.makedirs(tex_dir, exist_ok=True)
    for name, (fn, col) in TEXTURES.items():
        tex = None
        if textured:
            fname = f"{name[2:].lower()}.jpg"
            fn(rng).save(os.path.join(tex_dir, fname), quality=90)
            tex = f"textures/{fname}"
        scene.materials[name] = {"color": col, "texture": tex}
    for fname_, style, col in FACADE_STYLES:
        name = f"M_Facade_{fname_}"
        tex = None
        if textured:
            fn = f"facade_{fname_}.jpg"
            facade_style(rng, style, col).save(os.path.join(tex_dir, fn), quality=90)
            tex = f"textures/{fn}"
        scene.materials[name] = {"color": col, "texture": tex}
    for name, col in FACADES.items():
        tex = None
        if textured:
            fname = "facade_" + name.replace("M_Wall_", "").replace("M_WallDefault", "default").lower() + ".jpg"
            facade(rng, col).save(os.path.join(tex_dir, fname), quality=90)
            tex = f"textures/{fname}"
        scene.materials[name] = {"color": col, "texture": tex}
