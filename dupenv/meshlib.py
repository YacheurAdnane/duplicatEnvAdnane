# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Scene container and low-poly mesh primitives (numpy only).

Convention: right-handed local ENU frame in metres, Z up, triangles are
counter-clockwise seen from outside.
"""
import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry.polygon import orient

# Mesh or material names containing these words are dropped by CARLA's
# PrepareAssetsForCooking commandlet, so they must never appear in names.
FORBIDDEN_WORDS = ("sign", "light")


def safe_name(name):
    low = name.lower()
    for w in FORBIDDEN_WORDS:
        if w in low:
            raise ValueError(f"name '{name}' contains '{w}', CARLA would drop it")
    return name


class Scene:
    def __init__(self, chunk=250.0):
        self.chunk = chunk
        self.objects = {}
        self.materials = {}
        # semantic records (trees, barriers, signs, props) for exporters that
        # place library assets instead of meshes (RoadRunner HD Map)
        self.records = []
        # meshes kept one by one as well (buildings), for exporters that place
        # each as its own movable object (RoadRunner props)
        self.items = defaultdict(list)

    def material(self, name, color, texture=None):
        safe_name(name)
        if name not in self.materials:
            self.materials[name] = {"color": tuple(color), "texture": texture}
        return name

    def record(self, kind, **data):
        data["kind"] = kind
        self.records.append(data)

    def add(self, category, material, V, F, UV=None, name=None, smooth=False, sub=None, item=None):
        # float32 / int32 halve the memory of big scenes; 2 mm precision at 20 km
        V = np.asarray(V, np.float32)
        F = np.asarray(F, np.int32)
        if len(F) == 0:
            return
        if material not in self.materials:
            raise KeyError(f"unknown material {material}")
        if name is None:
            c = V[:, :2].mean(axis=0)
            cx, cy = int(math.floor(c[0] / self.chunk)), int(math.floor(c[1] / self.chunk))
            fx = f"{'m' if cx < 0 else ''}{abs(cx)}"
            fy = f"{'m' if cy < 0 else ''}{abs(cy)}"
            name = f"{category}_{sub}_{fx}_{fy}" if sub else f"{category}_{fx}_{fy}"
        safe_name(name)
        if UV is None:
            UV = V[:, :2] / 4.0
        obj = self.objects.get(name)
        if obj is None:
            obj = self.objects[name] = {"category": category, "smooth": smooth, "parts": defaultdict(list)}
        obj["parts"][material].append((V, F, np.asarray(UV, np.float32)))
        if item is not None:
            self.items[item].append((material, V, F, np.asarray(UV, np.float32)))

    def finalize(self):
        """List of objects with parts merged per material.

        The per-piece lists are released while merging, so the scene is never
        held twice in memory."""
        out = []
        for name in sorted(self.objects):
            obj = self.objects.pop(name)
            parts = []
            for mat, pieces in obj["parts"].items():
                Vs, Fs, UVs, off = [], [], [], 0
                for V, F, UV in pieces:
                    Vs.append(V)
                    Fs.append(F + off)
                    UVs.append(UV)
                    off += len(V)
                parts.append((mat, np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs)))
                pieces.clear()
            out.append({"name": name, "category": obj["category"], "smooth": obj["smooth"], "parts": parts})
        return out

    def stats(self):
        tris = 0
        per = defaultdict(int)
        for obj in self.objects.values():
            for pieces in obj["parts"].values():
                for _, F, _ in pieces:
                    tris += len(F)
                    per[obj["category"]] += len(F)
        return tris, dict(per)


# ---------------------------------------------------------------- normals

def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return vn / ln


# ---------------------------------------------------------------- primitives

def ribbon(A, B, u_scale=4.0):
    """Triangle strip between polylines A and B (n,3). Normal = (B-A) x dA."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n = len(A)
    V = np.empty((2 * n, 3))
    V[0::2] = A
    V[1::2] = B
    i = np.arange(n - 1)
    a0, b0, a1, b1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
    F = np.concatenate([np.stack([a0, b0, b1], 1), np.stack([a0, b1, a1], 1)])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))])
    w = np.linalg.norm(B - A, axis=1)
    UV = np.empty((2 * n, 2))
    UV[0::2, 0] = 0
    UV[1::2, 0] = w / u_scale
    UV[0::2, 1] = s / u_scale
    UV[1::2, 1] = s / u_scale
    return V, F, UV


def merge(*meshes):
    Vs, Fs, UVs, off = [], [], [], 0
    for V, F, UV in meshes:
        if len(F) == 0:
            continue
        Vs.append(V)
        Fs.append(F + off)
        UVs.append(UV)
        off += len(V)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2))
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs)


def box(sx, sy, sz):
    """Axis aligned box centred on x/y with its base at z=0."""
    x, y = sx / 2, sy / 2
    faces = [
        ([-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0], True),       # bottom (flip)
        ([-x, -y, sz], [x, -y, sz], [x, y, sz], [-x, y, sz], False),  # top
        ([-x, -y, 0], [x, -y, 0], [x, -y, sz], [-x, -y, sz], False),  # -y
        ([x, y, 0], [-x, y, 0], [-x, y, sz], [x, y, sz], False),      # +y
        ([x, -y, 0], [x, y, 0], [x, y, sz], [x, -y, sz], False),      # +x
        ([-x, y, 0], [-x, -y, 0], [-x, -y, sz], [-x, y, sz], False),  # -x
    ]
    V, F, UV = [], [], []
    for k, (a, b, c, d, flip) in enumerate(faces):
        o = 4 * k
        V += [a, b, c, d]
        UV += [[0, 0], [1, 0], [1, 1], [0, 1]]
        if flip:
            F += [[o, o + 2, o + 1], [o, o + 3, o + 2]]
        else:
            F += [[o, o + 1, o + 2], [o, o + 2, o + 3]]
    return np.array(V, float), np.array(F), np.array(UV, float)


def cylinder(r0, r1, h, seg=8, cap=True):
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    bot = np.stack([r0 * c, r0 * s, np.zeros(seg)], 1)
    top = np.stack([r1 * c, r1 * s, np.full(seg, h)], 1)
    V = np.concatenate([bot, top])
    i = np.arange(seg)
    j = (i + 1) % seg
    F = np.concatenate([np.stack([i, j, seg + j], 1), np.stack([i, seg + j, seg + i], 1)])
    UV = np.concatenate([np.stack([i / seg, np.zeros(seg)], 1), np.stack([i / seg, np.ones(seg)], 1)])
    if cap and r1 > 0:
        V = np.concatenate([V, [[0, 0, h]]])
        UV = np.concatenate([UV, [[0.5, 0.5]]])
        ci = 2 * seg
        F = np.concatenate([F, np.stack([seg + i, seg + j, np.full(seg, ci)], 1)])
    return V, F, UV


def icosphere(radius=1.0, subdiv=1):
    t = (1 + 5 ** 0.5) / 2
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(subdiv):
        cache = {}
        newF = []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = newF
    V = np.array(V) * radius
    UV = V[:, :2] * 0.5 + 0.5
    return V, np.array(F), UV


def transform(V, yaw=0.0, t=(0, 0, 0), scale=(1, 1, 1)):
    c, s = math.cos(yaw), math.sin(yaw)
    V = V * np.asarray(scale, float)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return V @ R.T + np.asarray(t, float)


def instances(mesh, pos, yaw, scale):
    """Stamp a template mesh at many positions (vectorised)."""
    V, F, UV = mesh
    k = len(pos)
    if k == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2))
    c, s = np.cos(yaw), np.sin(yaw)
    Vs = V[None, :, :] * scale[:, None, :]
    X = Vs[:, :, 0] * c[:, None] - Vs[:, :, 1] * s[:, None]
    Y = Vs[:, :, 0] * s[:, None] + Vs[:, :, 1] * c[:, None]
    Vo = np.stack([X, Y, Vs[:, :, 2]], axis=2) + pos[:, None, :]
    Fo = F[None, :, :] + (np.arange(k) * len(V))[:, None, None]
    UVo = np.broadcast_to(UV, (k,) + UV.shape)
    return Vo.reshape(-1, 3), Fo.reshape(-1, 3), UVo.reshape(-1, 2)


def polygon_cap(poly, z):
    """Triangulate a shapely polygon (holes allowed) as an up-facing face."""
    tris = shapely.get_parts(shapely.constrained_delaunay_triangles(poly))
    if len(tris) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2))
    coords = shapely.get_coordinates(tris).reshape(len(tris), 4, 2)[:, :3, :]
    # keep only triangles inside (CDT may return hull triangles for odd rings)
    cent = coords.mean(axis=1)
    inside = shapely.contains_xy(poly, cent[:, 0], cent[:, 1])
    coords = coords[inside]
    area = (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 2, 1] - coords[:, 0, 1]) - (
        coords[:, 2, 0] - coords[:, 0, 0]
    ) * (coords[:, 1, 1] - coords[:, 0, 1])
    cw = area < 0
    coords[cw] = coords[cw][:, ::-1]
    V2 = coords.reshape(-1, 2)
    V = np.column_stack([V2, np.broadcast_to(np.asarray(z, float), (len(V2),))])
    F = np.arange(len(V2)).reshape(-1, 3)
    return V, F, V2 / 4.0


def extrude_polygon(poly, z0, z1, roof=True, tile=6.0):
    """Walls (outward) and flat roof for a polygon with holes.

    Wall UVs: u = distance along the wall, v = height, both / tile metres,
    so a facade texture of `tile` x `tile` metres repeats with the storeys."""
    poly = orient(poly, 1.0)
    walls = []
    for ring in [poly.exterior] + list(poly.interiors):
        P = np.array(ring.coords)
        A = np.column_stack([P, np.full(len(P), z1)])
        B = np.column_stack([P, np.full(len(P), z0)])
        V, F, UV = ribbon(A, B, u_scale=1.0)
        UV = np.column_stack([UV[:, 1] / tile, (V[:, 2] - z0) / tile])
        walls.append((V, F, UV))
    parts = walls + ([polygon_cap(poly, z1)] if roof else [])
    return merge(*parts)


def wall_along(P, base_z, height, thickness, u_scale=2.0):
    """Closed wall of given thickness following polyline P (n,2)."""
    from .geo import left_normals, tangents

    P = np.asarray(P, float)[:, :2]
    if len(P) < 2:
        return merge()
    N = left_normals(tangents(P))
    zb = np.broadcast_to(np.asarray(base_z, float), (len(P),))
    zt = zb + np.broadcast_to(np.asarray(height, float), (len(P),))
    L = P + N * thickness / 2
    R = P - N * thickness / 2
    Lb, Lt = np.column_stack([L, zb]), np.column_stack([L, zt])
    Rb, Rt = np.column_stack([R, zb]), np.column_stack([R, zt])
    parts = [
        ribbon(Lb, Lt, u_scale),   # left face
        ribbon(Rt, Rb, u_scale),   # right face
        ribbon(Lt, Rt, u_scale),   # top
    ]
    for i, sgn in ((0, -1), (-1, 1)):
        q = np.array([Lb[i], Rb[i], Rt[i], Lt[i]])
        F = np.array([[0, 1, 2], [0, 2, 3]]) if sgn < 0 else np.array([[0, 2, 1], [0, 3, 2]])
        parts.append((q, F, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)))
    return merge(*parts)


def panel_outline(shape, w, h, seg=20):
    """2D outline (x across, z up, centred at 0) of a sign panel shape."""
    if shape == "circle":
        a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        return np.stack([w / 2 * np.cos(a), h / 2 * np.sin(a)], 1)
    if shape == "octagon":
        a = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        return np.stack([w / 2 * np.cos(a) / math.cos(np.pi / 8), h / 2 * np.sin(a) / math.cos(np.pi / 8)], 1)
    if shape == "triangle":
        return np.array([[-w / 2, -h / 2 + 0.0], [w / 2, -h / 2], [0, h / 2]])
    if shape == "triangle_down":
        return np.array([[-w / 2, h / 2], [0, -h / 2], [w / 2, h / 2]])
    if shape == "diamond":
        return np.array([[0, -h / 2], [w / 2, 0], [0, h / 2], [-w / 2, 0]])
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])


def panel(shape, w, h, thickness=0.03):
    """Sign panel in its local frame: faces -y (front) and +y (back).

    Returns (front mesh, back mesh). The front UVs map the bounding box of the
    outline to [0,1], so a texture drawn on a square canvas lines up.
    """
    O = panel_outline(shape, w, h)
    n = len(O)
    # front at y = -t/2 facing -y
    fx, fz = O[:, 0], O[:, 1]
    front_V = np.column_stack([fx, np.full(n, -thickness / 2), fz])
    center = np.array([[0.0, -thickness / 2, 0.0]])
    V = np.concatenate([front_V, center])
    i = np.arange(n)
    j = (i + 1) % n
    # outline is CCW in (x,z); seen from -y (x right, z up) that is CCW, so normal = -y
    F = np.stack([np.full(n, n), i, j], 1)
    UV = np.column_stack([(V[:, 0] + w / 2) / w, (V[:, 2] + h / 2) / h])
    front = (V, F, UV)
    back_V = np.concatenate([front_V + [0, thickness, 0], center + [0, thickness, 0]])
    Fb = np.stack([np.full(n, n), i, j], 1)
    Fb = Fb[:, [0, 2, 1]]
    top = back_V[:n]
    rim = ribbon(np.concatenate([front_V, front_V[:1]]), np.concatenate([top, top[:1]]))
    back = merge((back_V, Fb, np.zeros((n + 1, 2))), rim)
    return front, back


def cylinder_between(p0, p1, r0, r1, seg=6):
    """Tapered cylinder from point p0 to p1 (no caps)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    V, F, UV = cylinder(r0, r1, L, seg=seg, cap=False)
    z = d / L
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return V @ R.T + p0, F, UV


def blob(radius, rng, subdiv=0, jitter=0.18):
    """Irregular leaf cluster: an icosphere with radial noise."""
    V, F, UV = icosphere(1.0, subdiv)
    V = V * (1 + rng.uniform(-jitter, jitter, (len(V), 1))) * radius
    UV = np.column_stack([np.arctan2(V[:, 1], V[:, 0]) / np.pi, V[:, 2] / max(radius, 1e-6)])
    return V, F, UV
