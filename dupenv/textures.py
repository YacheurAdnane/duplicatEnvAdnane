# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Procedural textures for road panels (speed limits, stop, exit panels...)."""
import hashlib
import os
import re

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

RED = (200, 20, 30)
BLUE_MW = (20, 60, 150)    # motorway panels (FR/DE)
BLUE = (0, 90, 170)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GREEN = (0, 110, 60)
YELLOW = (250, 200, 0)


def _font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _center_text(d, box, text, color, max_size):
    x0, y0, x1, y1 = box
    size = max_size
    while size > 8:
        f = _font(size)
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        if r - l <= (x1 - x0) and b - t <= (y1 - y0):
            break
        size -= 2
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text(((x0 + x1 - (r - l)) / 2 - l, (y0 + y1 - (b - t)) / 2 - t), text, font=f, fill=color)


def classify(value, tags=None):
    """Map an OSM traffic_sign value to (kind, number/text)."""
    v = (value or "").strip()
    low = v.lower()
    tags = tags or {}
    num = re.search(r"\[(\d+)\]", v)
    code = v.split(":")[-1].upper() if ":" in v else v.upper()
    if "maxspeed" in low or code.startswith("B14") or code.startswith("274") or code.startswith("R2-1"):
        n = num.group(1) if num else re.sub(r"\D", "", str(tags.get("maxspeed", ""))) or "50"
        return "speed", n
    if code in ("AB4", "206", "R1-1") or low == "stop":
        return "stop", None
    if code.startswith("AB3") or code in ("205", "R1-2") or low in ("give_way", "yield"):
        return "giveway", None
    if code in ("B1", "267"):
        return "noentry", None
    if low.startswith("city_limit") or code.startswith("EB10") or code.startswith("310"):
        return "town", tags.get("name", "")
    if ":" in v:
        cc = v.split(":")[0].upper()
        if cc == "FR":
            if code.startswith("A"):
                return "danger", None
            if code.startswith("B21") or code.startswith("B22"):
                return "mandatory", None
            if code.startswith("B"):
                return "prohib", None
            if code.startswith("C") or code.startswith("CE"):
                return "info", code
            if code.startswith("D") or code.startswith("E"):
                return "direction", code
        if cc == "DE":
            if code.startswith("1"):
                return "danger", None
            if code.startswith("2") and code[:3] in ("209", "211", "214", "215", "222", "237", "239", "240", "241"):
                return "mandatory", None
            if code.startswith("2"):
                return "prohib", None
            if code.startswith("3") or code.startswith("4"):
                return "info", code
    return "generic", None


SHAPES = {
    "speed": "circle", "prohib": "circle", "noentry": "circle", "mandatory": "circle",
    "stop": "octagon", "giveway": "triangle_down", "danger": "triangle",
    "info": "rect", "direction": "rect", "town": "rect", "generic": "rect", "exit": "rect",
}


class PanelFactory:
    def __init__(self, scene, tex_dir, tex_rel="textures"):
        self.scene = scene
        self.dir = tex_dir
        self.rel = tex_rel
        os.makedirs(tex_dir, exist_ok=True)
        scene.material("M_PanelBack", (0.55, 0.56, 0.58))
        self.aspect = {}

    def _save(self, key, img):
        fn = f"panel_{key}.png"
        img.save(os.path.join(self.dir, fn))
        mat = f"M_Panel_{key}"
        self.scene.material(mat, (1, 1, 1), texture=f"{self.rel}/{fn}")
        self.aspect[mat] = img.width / img.height
        return mat

    def material(self, kind, value=None, lines=None, colour=None):
        """Returns (material name, shape, width/height). Textures are created once."""
        key = kind
        if value:
            key += "_" + re.sub(r"[^A-Za-z0-9]", "", str(value))[:20]
        if lines:
            key += "_" + hashlib.md5("|".join(lines).encode()).hexdigest()[:8]
        if "sign" in key.lower() or "light" in key.lower():  # CARLA drops such names
            key = kind + "_" + hashlib.md5(key.encode()).hexdigest()[:8]
        mat = f"M_Panel_{key}"
        shape = SHAPES.get(kind, "rect")
        if mat in self.scene.materials:
            return mat, shape, self.aspect.get(mat, 1.0)
        S = 256
        if kind in ("speed", "prohib", "noentry", "mandatory"):
            img = Image.new("RGB", (S, S), WHITE)
            d = ImageDraw.Draw(img)
            if kind == "mandatory":
                d.ellipse([0, 0, S - 1, S - 1], fill=BLUE)
                d.polygon([(S * .5, S * .2), (S * .72, S * .5), (S * .58, S * .5), (S * .58, S * .8),
                           (S * .42, S * .8), (S * .42, S * .5), (S * .28, S * .5)], fill=WHITE)
            elif kind == "noentry":
                d.ellipse([0, 0, S - 1, S - 1], fill=RED)
                d.rectangle([S * .18, S * .42, S * .82, S * .58], fill=WHITE)
            else:
                d.ellipse([0, 0, S - 1, S - 1], fill=RED)
                d.ellipse([S * .12, S * .12, S * .88, S * .88], fill=WHITE)
                if kind == "speed":
                    _center_text(d, (S * .2, S * .25, S * .8, S * .75), str(value), BLACK, 120)
        elif kind == "stop":
            img = Image.new("RGB", (S, S), RED)
            d = ImageDraw.Draw(img)
            _center_text(d, (S * .1, S * .3, S * .9, S * .7), "STOP", WHITE, 90)
        elif kind in ("giveway", "danger"):
            img = Image.new("RGB", (S, S), RED)
            d = ImageDraw.Draw(img)
            if kind == "danger":
                d.polygon([(S * .5, S * .2), (S * .86, S * .88), (S * .14, S * .88)], fill=WHITE)
                _center_text(d, (S * .4, S * .45, S * .6, S * .82), "!", BLACK, 100)
            else:
                d.polygon([(S * .14, S * .12), (S * .86, S * .12), (S * .5, S * .8)], fill=WHITE)
        elif kind in ("exit", "direction", "info", "town"):
            W, H = 512, 256 if not lines else max(256, 90 * len(lines) + 40)
            bg = {"exit": colour or BLUE_MW, "direction": colour or BLUE, "info": BLUE, "town": WHITE}[kind]
            fg = BLACK if kind == "town" else WHITE
            img = Image.new("RGB", (W, H), bg)
            d = ImageDraw.Draw(img)
            d.rectangle([6, 6, W - 7, H - 7], outline=fg if kind != "town" else RED, width=8)
            text = lines or [str(value or "")]
            lh = (H - 40) / max(1, len(text))
            for i, t in enumerate(text):
                _center_text(d, (30, 20 + i * lh, W - 30, 20 + (i + 1) * lh), t, fg, int(lh * 0.8))
        else:
            img = Image.new("RGB", (S, S), WHITE)
            d = ImageDraw.Draw(img)
            d.rectangle([4, 4, S - 5, S - 5], outline=(90, 90, 90), width=10)
        mat = self._save(key, img)
        return mat, shape, self.aspect[mat]
