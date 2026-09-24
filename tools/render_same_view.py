#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Render several outputs from the same camera, for side-by-side comparisons.

    python3 tools/render_same_view.py <cameras.json> <view> <out.png> output/A output/B ...

LABELS="Low,Medium,High" replaces the output names in the image.
The cameras come from preview/cameras.json of a run (views: street, aerial,
closeup, transition, verge). Each output is rendered from its preview GLB and
the images are placed side by side, labelled with the output name.
"""
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_preview import Renderer, load_glb  # noqa: E402


def main(cam_file, view, out_png, *outputs, size=(960, 600)):
    cam = json.load(open(cam_file))[view]
    tiles = []
    labels = os.environ.get("LABELS", "").split(",") if os.environ.get("LABELS") else None
    for k, o in enumerate(outputs):
        name = os.path.basename(os.path.realpath(o))
        glb_name = name
        if labels:
            name = labels[k]
        prims = load_glb(os.path.join(o, "preview", f"{glb_name}.glb"))
        r = Renderer(prims, size=size)
        img = r.render(cam["eye"], cam["target"], **{k: v for k, v in cam.items() if k not in ("eye", "target")})
        ImageDraw.Draw(img).rectangle((0, 0, 8 * len(name) + 16, 26), fill=(255, 255, 255))
        ImageDraw.Draw(img).text((8, 7), name, fill=(0, 0, 0))
        tiles.append(img)
    sheet = Image.new("RGB", (size[0] * len(tiles), size[1]), "white")
    for k, im in enumerate(tiles):
        sheet.paste(im, (k * size[0], 0))
    sheet.save(out_png, quality=88)
    print(out_png)


if __name__ == "__main__":
    main(*sys.argv[1:])
