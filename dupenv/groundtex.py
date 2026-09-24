# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Ground look for the three quality levels.

low     one baked colour texture per 500 m terrain tile: flat colours per
        ground type, blended over a few metres. No download.
medium  photo textures (ambientCG, CC0) per ground type, 1K source. Where two
        ground types meet, the terrain mesh is refined to 2 m and the border
        follows noisy, organic shapes over a band of several metres, so one
        type thins out into the other instead of a straight 4 m staircase.
high    the same with 2K sources and a 1 m transition mesh, plus 3D details:
        grass tufts and flowers on green ground, small rocks on dirt and
        forest floor (see features.build_ground_details).

ambientCG and Poly Haven ("ph:" names) textures are CC0 (public domain).
Each is downloaded once; only its colour map is kept, in cache/textures/.
"""
import io
import os
import threading
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from . import progress as prog
from .net import CACHE_DIR, UA

ACG_URL = "https://ambientcg.com/get?file={asset}_{res}-JPG.zip"
PH_URL = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/{r}/{asset}/{asset}_diff_{r}.jpg"

# ground class -> (material, texture, colour for the low level = the photo's average colour)
# class ids follow landcover: GRASS, MEADOW, CROP, FIELD, FOREST, PAVED, BARE
CLASSES = [
    ("M_Grass", "Grass004", (0.38, 0.43, 0.19)),
    ("M_Meadow", "Ground037", (0.60, 0.58, 0.35)),
    ("M_Crop", "Ground104", (0.51, 0.42, 0.27)),
    ("M_Field", "Ground048", (0.33, 0.24, 0.20)),
    ("M_ForestFloor", "ph:brown_mud_leaves_01", (0.40, 0.33, 0.20)),
    ("M_Paving", "Gravel043", (0.42, 0.42, 0.41)),
    ("M_Dirt", "Ground023", (0.40, 0.35, 0.29)),
]
EXTRA = {"M_Rock": "Rock030"}
QUALITY = {
    # source resolution, texture pixels for TILE_M metres, transition mesh subdivision of a 4 m cell
    "low": {"res": None, "px": 0, "sub": 1, "details": False},
    "medium": {"res": "1K", "px": 2048, "sub": 2, "details": False},
    "high": {"res": "2K", "px": 4096, "sub": 3, "details": True},
}
TILE_M = 4.0  # metres covered by one ground texture (UV = xy / TILE_M)
COMPOSITES = {}  # material -> texture file of this job (filled by apply)
_lock = threading.Lock()


def tex_dir():
    d = os.path.join(CACHE_DIR, "textures")
    os.makedirs(d, exist_ok=True)
    return d


def fetch(asset, res, log=print):
    """Colour map of an ambientCG material (path), downloaded once."""
    dest = os.path.join(tex_dir(), f"{asset.replace(':', '_')}_{res}.jpg")
    if os.path.exists(dest):
        return dest
    if asset.startswith("ph:"):
        url = PH_URL.format(asset=asset[3:], r=res.lower())
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        Image.open(io.BytesIO(data)).verify()
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        return dest
    url = ACG_URL.format(asset=asset, res=res)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    buf = io.BytesIO()
    with urllib.request.urlopen(req, timeout=120) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf.write(chunk)
    with zipfile.ZipFile(buf) as z:
        names = [n for n in z.namelist() if n.lower().endswith(("_color.jpg", "_color.jpeg"))]
        if not names:
            raise RuntimeError(f"no colour map in {url}")
        data = z.read(names[0])
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return dest


def prefetch(quality, log=print, task="groundtex"):
    """Download every texture the quality level needs (parallel, cached)."""
    q = QUALITY[quality]
    if not q["res"]:
        return {}
    assets = [a for _, a, _ in CLASSES] + list(EXTRA.values())
    c = prog.Counter(task, len(assets), "texture")
    out = {}

    def one(a):
        try:
            out[a] = fetch(a, q["res"], log)
        except Exception as e:
            log(f"  texture {a} unavailable ({str(e)[:60]}), using a generated one")
        c.tick()
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, assets))
    return out


def _periodic_noise(n, beta, rng):
    w = rng.standard_normal((n, n)).astype(np.float32)
    f = np.fft.fftfreq(n).astype(np.float32)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0
    img = np.real(np.fft.ifft2(np.fft.fft2(w) / r ** beta)).astype(np.float32)
    img -= img.min()
    return img / max(float(img.max()), 1e-9)


def composite(src_path, out_px, seed):
    """Seamless out_px texture from a tileable photo texture (2 x 2 source
    tiles), with three shifted / rotated copies mixed through soft noise
    masks and a slight brightness drift, so the repetition doesn't show."""
    key = f"{os.path.basename(src_path)[:-4]}_{out_px}_{seed}.jpg"
    cached = os.path.join(tex_dir(), "comp_" + key)
    if os.path.exists(cached):
        return Image.open(cached)
    rng = np.random.default_rng(seed)
    half = out_px // 2
    src = np.asarray(Image.open(src_path).convert("RGB").resize((half, half), Image.LANCZOS), np.float32)
    base = np.tile(src, (2, 2, 1))
    layers = [base]
    for k in range(2):
        rot = np.rot90(base, k=int(rng.integers(1, 4)))
        layers.append(np.roll(rot, (int(rng.integers(out_px)), int(rng.integers(out_px))), axis=(0, 1)))
    small = max(64, out_px // 8)  # masks at low resolution, then upsampled (smooth anyway)
    m = np.stack([_periodic_noise(small, 1.6, rng) for _ in layers])
    m = np.exp(6.0 * m)
    m /= m.sum(axis=0, keepdims=True)
    m = np.stack([np.asarray(Image.fromarray(x).resize((out_px, out_px), Image.BILINEAR)) for x in m])
    out = np.zeros_like(base)
    for w, L in zip(m, layers):
        out += w[..., None] * L
    drift = np.asarray(Image.fromarray(_periodic_noise(small, 2.0, rng)).resize((out_px, out_px), Image.BILINEAR))
    out *= (0.93 + 0.14 * drift)[..., None]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    with _lock:
        img.save(cached, quality=90)
    return img


def apply(scene, tex_dir_out, quality, fetched, log=print):
    """Point the ground materials at the photo composites (medium / high).
    Missing downloads keep the generated texture."""
    q = QUALITY[quality]
    if not q["res"]:
        return 0
    jobs = [(mat, asset) for mat, asset, _ in CLASSES] + list(EXTRA.items())
    n = [0]

    def one(job):
        mat, asset = job
        src = fetched.get(asset)
        if not src:
            return
        px = q["px"] if mat != "M_Rock" else q["px"] // 4
        img = composite(src, px, seed=zlib.crc32(asset.encode()) % 1000)
        fn = f"ground_{mat[2:].lower()}_{quality}.jpg"
        img.save(os.path.join(tex_dir_out, fn), quality=88)
        with _lock:
            COMPOSITES[mat] = os.path.join(tex_dir_out, fn)
        col = tuple(float(v) / 255 for v in np.asarray(img.resize((1, 1), Image.BOX)).reshape(3))
        with _lock:
            scene.materials[mat] = {"color": col, "texture": f"textures/{fn}"}
            n[0] += 1
    with ThreadPoolExecutor(3) as ex:  # a 4K composite peaks near 1 GB
        list(ex.map(one, jobs))
    return n[0]


def mix_material(scene, tex_dir_out, mat_a, mat_b, quality, share=0.5):
    """Material between two ground types: both photos mixed through a blotchy
    mask, `share` of the area showing mat_a. Two of them per pair (0.7, 0.3)
    make the band where one type thins out into the other."""
    name = f"M_Mix_{mat_a[2:]}_{mat_b[2:]}_{int(round(share * 100))}"
    if name in scene.materials:
        return name
    pa, pb = COMPOSITES.get(mat_a), COMPOSITES.get(mat_b)
    if not pa or not pb:
        return mat_a if share >= 0.5 else mat_b
    n = Image.open(pa).size[0] // 2  # half resolution: these only cover the bands
    A = np.asarray(Image.open(pa).convert("RGB").resize((n, n), Image.LANCZOS), np.float32)
    B = np.asarray(Image.open(pb).convert("RGB").resize((n, n), Image.LANCZOS), np.float32)
    rng = np.random.default_rng(zlib.crc32(f"{mat_a}{mat_b}".encode()))  # same blotches for both shares
    small = max(128, n // 4)
    noise = _periodic_noise(small, 1.3, rng)
    thr = float(np.quantile(noise, 1 - share))
    m = np.clip((noise - thr) * 6.0 + 0.5, 0, 1)  # blotches with soft edges
    m = np.asarray(Image.fromarray(m).resize((n, n), Image.BILINEAR))[..., None]
    img = Image.fromarray(np.clip(A * m + B * (1 - m), 0, 255).astype(np.uint8))
    fn = f"ground_mix_{mat_a[2:].lower()}_{mat_b[2:].lower()}_{int(round(share * 100))}_{quality}.jpg"
    img.save(os.path.join(tex_dir_out, fn), quality=88)
    col = tuple(np.array(scene.materials[mat_a]["color"]) * share + np.array(scene.materials[mat_b]["color"]) * (1 - share))
    with _lock:
        scene.materials[name] = {"color": col, "texture": f"textures/{fn}"}
    return name


# ------------------------------------------------------------ world noise

def _hash2(ix, iy, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2654435761) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def value_noise(x, y, scale, seed):
    """Smooth noise in [0, 1] of world coordinates: the same value whichever
    tile asks, so transitions continue across tile borders."""
    u, v = np.asarray(x, np.float64) / scale, np.asarray(y, np.float64) / scale
    i0, j0 = np.floor(u), np.floor(v)
    fu, fv = u - i0, v - j0
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    i0, j0 = i0.astype(np.int64), j0.astype(np.int64)
    a = _hash2(i0, j0, seed)
    b = _hash2(i0 + 1, j0, seed)
    c = _hash2(i0, j0 + 1, seed)
    d = _hash2(i0 + 1, j0 + 1, seed)
    return (a * (1 - fu) * (1 - fv) + b * fu * (1 - fv) + c * (1 - fu) * fv + d * fu * fv).astype(np.float32)


def fractal(x, y, scale, seed, octaves=2):
    tot, amp, norm = 0.0, 1.0, 0.0
    for k in range(octaves):
        tot = tot + amp * value_noise(x, y, scale / (2.3 ** k), seed + 101 * k)
        norm += amp
        amp *= 0.5
    return tot / norm


def low_palette():
    return np.array([c for _, _, c in CLASSES], np.float32)
