#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Count road surfaces that overlap (the flickering "two roads at once" glitch).

    python3 tools/check_overlaps.py output/<name>

Takes the centre of every upward road triangle in the preview and checks
whether another road triangle lies under/over it within 30 cm of height.
"""
import os
import sys

import numpy as np
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_preview import load_glb  # noqa: E402


def main(out):
    name = os.path.basename(os.path.realpath(out))
    prims = load_glb(os.path.join(out, "preview", f"{name}.glb"))
    tris, owner = [], []
    for k, (nm, pos, nrm, uv, idx, col, img) in enumerate(prims):
        if not nm.startswith(("Road_Road", "Road_Sidewalk")):
            continue
        T = pos[idx.reshape(-1, 3)]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        up = n[:, 2] > 1e-6 * np.linalg.norm(n, axis=1).clip(1e-9)
        up &= np.abs(n[:, 2]) > 0.5 * np.linalg.norm(n, axis=1)  # skip curbs / side faces
        T = T[up]
        tris.append(T)
        owner.append(np.full(len(T), k))
    T = np.concatenate(tris)
    owner = np.concatenate(owner)
    polys = shapely.polygons(T[:, :, :2])
    area = shapely.area(polys)
    ok = area > 0.05
    T, polys, owner = T[ok], polys[ok], owner[ok]
    cen = T.mean(axis=1)
    tree = shapely.STRtree(polys)
    a, b = tree.query(shapely.points(cen[:, 0], cen[:, 1]), predicate="within")
    other = a != b
    dz = np.abs(cen[a, 2] - T[b].mean(axis=1)[:, 2])
    bad = np.unique(a[other & (dz < 0.3)])
    print(f"{len(T)} road triangles, {len(bad)} overlapped by another road surface "
          f"({len(bad) / len(T) * 100:.2f}%)")
    if "-v" in sys.argv:
        import collections
        names = [p[0] for p in prims]
        pairs = collections.Counter()
        sel = other & (dz < 0.3)
        for i, j in zip(a[sel], b[sel]):
            k1 = "_".join(names[owner[i]].split("_")[:2])
            k2 = "_".join(names[owner[j]].split("_")[:2])
            pairs[tuple(sorted((k1, k2)))] += 1
        print(pairs.most_common(6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
