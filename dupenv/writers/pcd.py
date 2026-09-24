# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Point cloud map (binary PCD) sampled from the generated meshes, for Autoware NDT."""
import numpy as np

INTENSITY = {
    "Road_Road": 20, "Road_Marking": 200, "Road_Sidewalk": 40, "Terrain": 60, "Ground": 60,
    "Building": 90, "Vegetation": 70, "GuardRail": 180, "Panel": 250, "Pole": 150, "Wall": 100,
    "Fence": 110, "Bridge": 100, "RailTrack": 80, "Water": 5, "Static": 120,
}


def sample_objects(objects, density, rng, terrain_density=None):
    pts, inten = [], []
    for obj in objects:
        cat = obj["category"]
        d = terrain_density if (terrain_density and cat in ("Terrain", "Ground")) else density
        for _, V, F, _ in obj["parts"]:
            A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
            lam = area * d
            n = np.floor(lam).astype(np.int64) + (rng.random(len(lam)) < (lam - np.floor(lam)))
            tot = int(n.sum())
            if tot == 0:
                continue
            tri = np.repeat(np.arange(len(F)), n)
            u, v = rng.random(tot), rng.random(tot)
            flip = u + v > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            P = A[tri] + (B[tri] - A[tri]) * u[:, None] + (C[tri] - A[tri]) * v[:, None]
            pts.append(P.astype(np.float32))
            inten.append(np.full(tot, INTENSITY.get(cat, 100), np.float32))
    if not pts:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    return np.concatenate(pts), np.concatenate(inten)


def voxel_filter(P, I, leaf):
    if len(P) == 0 or leaf <= 0:
        return P, I
    key = np.floor(P / leaf).astype(np.int64)
    key -= key.min(axis=0)
    mx = key.max(axis=0) + 1
    lin = (key[:, 0] * mx[1] + key[:, 1]) * mx[2] + key[:, 2]
    _, first = np.unique(lin, return_index=True)
    return P[first], I[first]


def write_pcd(path, P, I):
    n = len(P)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
        "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    data = np.empty((n, 4), np.float32)
    data[:, :3] = P
    data[:, 3] = I
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())
    return n
