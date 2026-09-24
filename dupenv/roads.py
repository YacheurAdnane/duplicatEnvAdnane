# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Road models (lanes, widths, elevation) and their meshes.

One `RoadModel` is built for the selected route and one for every other OSM
highway inside the corridor. The same model feeds the mesh, the OpenDRIVE
writer and the Lanelet2 writer, so lanes match everywhere.
"""
import re

import numpy as np
from scipy.spatial import cKDTree

from . import meshlib as ml
from .geo import (box_smooth, cumlen, gaussian_smooth, left_normals, resample, runs,
                  smooth_polyline, tangents)

STEP = 2.0          # sampling step along roads (m)
EPS_W = 0.05        # lanes narrower than this do not exist
MARK_Z = 0.02       # markings float above the asphalt

# lane width, default lanes per direction, right shoulder, left (inner) shoulder,
# default speed, OpenDRIVE road type, draw edge lines, rank
CLASSES = {
    "motorway":       (3.50, 2, 2.5, 1.0, 130, "motorway", True, 10),
    "trunk":          (3.50, 2, 2.0, 0.5, 110, "rural", True, 9),
    "motorway_link":  (3.50, 1, 1.5, 0.5, 70, "motorway", True, 8),
    "trunk_link":     (3.50, 1, 1.0, 0.5, 70, "rural", True, 8),
    "primary":        (3.25, 1, 0.5, 0.3, 80, "rural", True, 7),
    "primary_link":   (3.25, 1, 0.5, 0.3, 50, "rural", True, 7),
    "secondary":      (3.00, 1, 0.4, 0.3, 80, "rural", True, 6),
    "secondary_link": (3.00, 1, 0.4, 0.3, 50, "rural", True, 6),
    "tertiary":       (3.00, 1, 0.3, 0.3, 80, "rural", False, 5),
    "tertiary_link":  (3.00, 1, 0.3, 0.3, 50, "rural", False, 5),
    "unclassified":   (2.75, 1, 0.2, 0.2, 80, "rural", False, 4),
    "residential":    (2.75, 1, 0.2, 0.2, 50, "town", False, 4),
    "living_street":  (2.75, 1, 0.0, 0.0, 20, "lowSpeed", False, 3),
    "road":           (3.00, 1, 0.2, 0.2, 50, "town", False, 3),
    "busway":         (3.25, 1, 0.2, 0.2, 50, "town", False, 3),
    "service":        (3.00, 1, 0.0, 0.0, 30, "lowSpeed", False, 2),
}
PATHS = {"footway": 1.8, "path": 1.5, "cycleway": 2.0, "pedestrian": 4.0, "steps": 1.5,
         "bridleway": 2.0, "track": 3.0}
SKIP = {"construction", "proposed", "platform", "raceway", "corridor", "elevator", "bus_stop",
        "rest_area", "services", "abandoned", "disused", "escape", "emergency_bay", "via_ferrata"}

COUNTRY_SPEED = {  # maxspeed zone values like FR:urban
    "urban": 50, "rural": 80, "motorway": 130, "trunk": 110, "living_street": 20, "zone30": 30,
}


def _num(v, default=None):
    if v is None:
        return default
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(v))
    return float(m.group(1)) if m else default


def parse_speed(tags, default):
    v = tags.get("maxspeed")
    if not v:
        return default
    v = str(v).split(";")[0].strip()
    n = _num(v)
    if n is not None:
        return n * 1.609 if "mph" in v else n
    zone = v.split(":")[-1].lower()
    return COUNTRY_SPEED.get(zone, default)


def oneway_dir(tags, hclass):
    ow = str(tags.get("oneway", "")).lower()
    if ow in ("yes", "true", "1"):
        return 1
    if ow in ("-1", "reverse"):
        return -1
    if ow == "no":
        return 0
    if hclass in ("motorway", "motorway_link", "trunk_link") or tags.get("junction") in ("roundabout", "circular"):
        return 1
    return 0


def lanes_split(tags, hclass):
    """Returns (lanes forward, lanes backward, oneway direction)."""
    d = CLASSES.get(hclass, CLASSES["road"])
    ow = oneway_dir(tags, hclass)
    total = _num(tags.get("lanes"))
    fw = _num(tags.get("lanes:forward"))
    bw = _num(tags.get("lanes:backward"))
    if ow != 0:
        n = int(total) if total else d[1]
        return max(1, n), 0, ow
    if hclass == "service" and not total:
        return 1, 0, 0  # single shared lane
    if fw is None and bw is None:
        if total:
            t = int(total)
            if t == 1:
                return 1, 0, 0
            fw, bw = t - t // 2, t // 2
        else:
            fw = bw = 1
    elif fw is None:
        fw = max(1, (int(total) if total else 2) - int(bw))
    elif bw is None:
        bw = max(1, (int(total) if total else 2) - int(fw))
    return int(fw), int(bw), 0


def lane_width(tags, hclass, total_lanes):
    d = CLASSES.get(hclass, CLASSES["road"])
    w = _num(tags.get("width"))
    if w and total_lanes:
        return float(np.clip(w / total_lanes, 2.4, 4.0))
    return d[0]


def is_bridge(tags):
    return str(tags.get("bridge", "no")).lower() not in ("no", "") or tags.get("man_made") == "bridge"


def is_tunnel(tags):
    return str(tags.get("tunnel", "no")).lower() in ("yes", "building_passage", "avalanche_protector")


class RoadModel:
    """Sampled road with per-sample lane widths.

    wr[:, k] is the width of the k-th lane right of the centre lane (driving in
    the direction of the samples), wl[:, k] the k-th lane left of it (opposite
    direction). c0 is the lateral offset of the centre lane from the sampled
    polyline (positive = left).
    """

    def __init__(self, name, kind, P, s):
        self.name = name
        self.kind = kind            # route | road | path | track
        self.x, self.y = P[:, 0].copy(), P[:, 1].copy()
        self.s = s
        self.T = tangents(P)
        self.N = left_normals(self.T)
        self.hdg = np.arctan2(self.T[:, 1], self.T[:, 0])
        n = len(s)
        self.z = np.zeros(n)
        self.wr = np.zeros((n, 0))
        self.wl = np.zeros((n, 0))
        self.c0 = np.zeros(n)
        self.rsh = np.zeros(n)
        self.lsh = np.zeros(n)
        self.speed = np.full(n, 50.0)
        self.bridge = np.zeros(n, bool)
        self.tunnel = np.zeros(n, bool)
        self.hclass = ["road"] * n
        self.osm_id = None
        self.oneway = True
        self.xodr_type = "town"
        self.edge_lines = False
        self.rank = 3

    @property
    def n(self):
        return len(self.s)

    def right_lane_edge(self):
        return self.c0 - self.wr.sum(axis=1)

    def left_lane_edge(self):
        return self.c0 + self.wl.sum(axis=1)

    def surface_edges(self):
        """(left, right) lateral offsets of the paved surface."""
        return self.left_lane_edge() + self.lsh, self.right_lane_edge() - self.rsh

    def point(self, offset, dz=0.0, idx=slice(None)):
        o = np.broadcast_to(np.asarray(offset, float), (self.n,))[idx]
        x = self.x[idx] + self.N[idx, 0] * o
        y = self.y[idx] + self.N[idx, 1] * o
        return np.column_stack([x, y, self.z[idx] + dz])

    def subset(self, a, b):
        m = RoadModel.__new__(RoadModel)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray) and v.shape[:1] == (self.n,):
                setattr(m, k, v[a:b].copy())
            elif isinstance(v, list) and len(v) == self.n:
                setattr(m, k, v[a:b])
            else:
                setattr(m, k, v)
        m.s = m.s - m.s[0]
        return m


# ---------------------------------------------------------------- elevation

def road_elevation(model, elev, sigma_m):
    z = elev.sample_xy(model.x, model.y)
    n = model.n
    for flag in (model.bridge, model.tunnel):
        for a, b in runs(flag):
            a0 = max(a - 3, 0)
            b0 = min(b + 2, n - 1)
            if b0 <= a0:
                continue
            t = (model.s[a:b] - model.s[a0]) / max(model.s[b0] - model.s[a0], 1e-6)
            z[a:b] = z[a0] + t * (z[b0] - z[a0])
    model.z = gaussian_smooth(z, sigma_m / STEP)


def _clean_runs(v, min_len):
    """Merge runs shorter than min_len samples into their neighbour."""
    v = np.asarray(v).copy()
    changed = True
    while changed:
        changed = False
        edges = np.nonzero(np.diff(v) != 0)[0] + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [len(v)]])
        for a, b in zip(starts, ends):
            if b - a < min_len and not (a == 0 and b == len(v)):
                v[a:b] = v[a - 1] if a > 0 else v[b]
                changed = True
                break
    return v


# ---------------------------------------------------------------- route

def _motor_way_points(osm):
    pts, wids, dirs = [], [], []
    for wid, w in osm.ways.items():
        h = w["tags"].get("highway")
        if h not in CLASSES or w["tags"].get("area") == "yes":
            continue
        P = osm.way_xy(wid)
        if P is None:
            continue
        for a, b in zip(P[:-1], P[1:]):
            L = np.hypot(*(b - a))
            if L < 1e-3:
                continue
            k = max(1, int(L / 2.0))
            t = (np.arange(k) + 0.5) / k
            pts.append(a + (b - a) * t[:, None])
            wids.append(np.full(k, wid))
            dirs.append(np.repeat(((b - a) / L)[None], k, axis=0))
    if not pts:
        return None
    return np.concatenate(pts), np.concatenate(wids), np.concatenate(dirs)


def build_route_model(route_xy, osm, elev, cfg, log):
    P, _ = resample(route_xy, 1.0)
    P = smooth_polyline(P, 7)
    P, s = resample(P, STEP)
    m = RoadModel("route", "route", P, s)
    n = m.n

    # match every 10 m of route to the OSM way it drives on
    mp = _motor_way_points(osm)
    way_of = np.full(n, -1, np.int64)
    fwd = np.ones(n, bool)
    if mp is not None:
        pts, wids, dirs = mp
        tree = cKDTree(pts)
        qi = np.arange(0, n, 5)
        dist, idx = tree.query(np.column_stack([m.x[qi], m.y[qi]]), k=32, distance_upper_bound=25)
        for r, i in enumerate(qi):
            best, bw, bf = 1e9, -1, True
            for d, j in zip(dist[r], idx[r]):
                if not np.isfinite(d):
                    break
                wid = wids[j]
                tags = osm.ways[wid]["tags"]
                ow = oneway_dir(tags, tags.get("highway"))
                dot = float(dirs[j] @ m.T[i])
                if ow != 0:
                    ok = ow * dot > 0.6
                else:
                    ok = abs(dot) > 0.6
                cost = d + (0 if ok else 1000)
                if cost < best:
                    best, bw, bf = cost, wid, dot > 0
            if best < 1000:
                way_of[i] = bw
                fwd[i] = bf
        # spread to all samples
        known = qi[way_of[qi] >= 0]
        if len(known):
            near = known[np.clip(np.searchsorted(known, np.arange(n)), 0, len(known) - 1)]
            prev = known[np.clip(np.searchsorted(known, np.arange(n)) - 1, 0, len(known) - 1)]
            pick = np.where(np.abs(near - np.arange(n)) < np.abs(prev - np.arange(n)), near, prev)
            way_of = way_of[pick]
            fwd = fwd[pick]
    matched = (way_of >= 0).mean()
    log(f"  route matched to OSM ways on {matched * 100:.0f}% of its length")

    nR = np.ones(n, int)
    nL = np.zeros(n, int)
    w = np.full(n, 3.5)
    rsh = np.zeros(n)
    lsh = np.zeros(n)
    spd = np.full(n, 50.0)
    rank = np.zeros(n, int)
    for i in range(n):
        wid = way_of[i]
        tags = osm.ways[wid]["tags"] if wid >= 0 else {"highway": cfg.get("default_class", "primary")}
        h = tags.get("highway", "road")
        if h not in CLASSES:
            h = "road"
        d = CLASSES[h]
        f, b, ow = lanes_split(tags, h)
        if ow != 0:
            nR[i], nL[i] = f, 0
            lsh[i] = d[3]
        else:
            nR[i], nL[i] = (f, b) if fwd[i] else (b, f)
            lsh[i] = d[2]
            if b == 0:  # single shared lane
                nL[i] = 0
        w[i] = lane_width(tags, h, f + b)
        rsh[i] = d[2]
        spd[i] = parse_speed(tags, d[4])
        m.bridge[i] = is_bridge(tags)
        m.tunnel[i] = is_tunnel(tags)
        m.hclass[i] = h
        rank[i] = d[7]

    clean = int(40 / STEP)
    nR = _clean_runs(nR, clean)
    nL = _clean_runs(nL, clean)
    spd = _clean_runs(np.round(spd).astype(int), int(100 / STEP)).astype(float)
    m.bridge = _clean_runs(m.bridge.astype(int), 3).astype(bool)
    m.tunnel = _clean_runs(m.tunnel.astype(int), 3).astype(bool)

    ramp_half = max(1, int(cfg.get("lane_taper", 80.0) / 2 / STEP))
    KR, KL = int(nR.max()), int(nL.max())
    wr = np.stack([w * (k < nR) for k in range(KR)], axis=1)
    wl = np.stack([w * (k < nL) for k in range(KL)], axis=1) if KL else np.zeros((n, 0))
    m.wr = box_smooth(wr, ramp_half)
    m.wl = box_smooth(wl, ramp_half) if KL else wl
    m.wr[m.wr < EPS_W] = 0
    if KL:
        m.wl[m.wl < EPS_W] = 0
    m.c0 = (m.wr.sum(1) - m.wl.sum(1)) / 2
    m.rsh = box_smooth(rsh, ramp_half)
    m.lsh = box_smooth(lsh, ramp_half)
    m.speed = spd
    m.oneway = KL == 0
    top = m.hclass[int(np.argmax(rank))]
    d = CLASSES[top]
    m.xodr_type, m.edge_lines, m.rank = d[5], True, 11
    road_elevation(m, elev, cfg.get("route_z_sigma", 20.0))
    return m


# ---------------------------------------------------------------- other roads

def build_way_models(osm, elev, route, corridor, cfg, log):
    from shapely import contains_xy

    rtree = cKDTree(np.column_stack([route.x, route.y]))
    rl, rr = route.surface_edges()
    r_half = np.maximum(np.abs(rl), np.abs(rr))
    clip = corridor.buffer(5)
    models = []
    for wid, w in osm.ways.items():
        tags = w["tags"]
        h = tags.get("highway")
        if not h or h in SKIP or tags.get("area") == "yes":
            continue
        if h not in CLASSES and h not in PATHS:
            continue
        if not cfg.get("paths", True) and h in PATHS and h != "track":
            continue
        P = osm.way_xy(wid)
        if P is None or cumlen(P)[-1] < 3:
            continue
        if h in CLASSES:
            f, b, ow = lanes_split(tags, h)
            if ow == -1:
                P = P[::-1]
            kind = "road"
        else:
            kind = "track" if h == "track" else "path"
            ow = 1
        Q, s = resample(P, STEP)
        m = RoadModel(f"way{wid}", kind, Q, s)
        m.osm_id = wid
        m.hclass = [h] * m.n
        m.bridge[:] = is_bridge(tags)
        m.tunnel[:] = is_tunnel(tags)
        if kind == "road":
            d = CLASSES[h]
            lw = lane_width(tags, h, f + b)
            m.wr = np.full((m.n, f), lw)
            m.wl = np.full((m.n, b), lw)
            m.c0[:] = (f - b) * lw / 2
            m.rsh[:] = d[2]
            m.lsh[:] = d[3] if b == 0 else d[2]
            m.speed[:] = parse_speed(tags, d[4])
            m.oneway = b == 0
            m.xodr_type, m.edge_lines, m.rank = d[5], d[6], d[7]
        else:
            width = _num(tags.get("width"), PATHS[h])
            m.wr = np.full((m.n, 1), width)
            m.c0[:] = width / 2
            m.speed[:] = 10
            m.rank = 1
        road_elevation(m, elev, 12.0 if kind == "road" else 6.0)

        # drop the parts that duplicate the route surface
        d, j = rtree.query(np.column_stack([m.x, m.y]))
        lat = np.abs((m.x - route.x[j]) * route.N[j, 0] + (m.y - route.y[j]) * route.N[j, 1])
        dot = np.einsum("ij,ij->i", m.T, route.T[j])
        par = np.abs(dot) > 0.9 if (kind != "road" or not m.oneway) else dot > 0.9
        dup = (lat < r_half[j] + 0.5) & par & (np.abs(m.z - route.z[j]) < 3) & (d < r_half[j] + 3)
        keep = ~dup & contains_xy(clip, m.x, m.y)
        for a, b in runs(keep):
            if b - a >= 3:
                models.append(m.subset(a, b))
    log(f"  {len(models)} other road/path pieces")
    return models


# ---------------------------------------------------------------- meshes

def _pieces(n, per=64):
    a = 0
    while a < n - 1:
        b = min(n, a + per + 1)
        yield a, b
        a = b - 1


def _add_strip(scene, cat, mat, A, B, a, b, name=None):
    if b - a >= 2:
        scene.add(cat, mat, *ml.ribbon(A[a:b], B[a:b]), name=name)


def _dash_quads(m, off, mask, width, dash, gap, zoff):
    """Vectorised dashes along boundary `off` where `mask` holds."""
    period = dash + gap
    s = m.s
    starts = np.arange(s[0] + gap / 2, s[-1] - dash, period)
    if len(starts) == 0:
        return None
    ends = starts + dash
    okm = mask.astype(float)
    ok = (np.interp(starts, s, okm) > 0.999) & (np.interp(ends, s, okm) > 0.999)
    starts, ends = starts[ok], ends[ok]
    if len(starts) == 0:
        return None

    def at(sv):
        x = np.interp(sv, s, m.x)
        y = np.interp(sv, s, m.y)
        z = np.interp(sv, s, m.z) + zoff
        nx = np.interp(sv, s, m.N[:, 0])
        ny = np.interp(sv, s, m.N[:, 1])
        ln = np.hypot(nx, ny)
        nx, ny = nx / ln, ny / ln
        o = np.interp(sv, s, off)
        px, py = x + nx * o, y + ny * o
        return px, py, z, nx, ny

    x0, y0, z0, nx0, ny0 = at(starts)
    x1, y1, z1, nx1, ny1 = at(ends)
    h = width / 2
    V = np.stack([
        np.column_stack([x0 + nx0 * h, y0 + ny0 * h, z0]),
        np.column_stack([x0 - nx0 * h, y0 - ny0 * h, z0]),
        np.column_stack([x1 - nx1 * h, y1 - ny1 * h, z1]),
        np.column_stack([x1 + nx1 * h, y1 + ny1 * h, z1]),
    ], axis=1)  # (k,4,3) L0 R0 R1 L1
    k = len(starts)
    base = (np.arange(k) * 4)[:, None]
    F = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]])
    UV = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), (k, 1))
    return V.reshape(-1, 3), F, UV, (x0 + x1) / 2, (y0 + y1) / 2


def _markings(scene, m, cfg):
    dash, gap = cfg.get("dash", 3.0), cfg.get("gap", 10.0)
    wsep, wedge = cfg.get("line_width", 0.15), cfg.get("edge_width", 0.2)
    lines = []  # (offset array, mask, style, width)
    cr = np.cumsum(m.wr, axis=1)
    KR = m.wr.shape[1]
    KL = m.wl.shape[1]
    big = 1.0
    has_r = m.wr[:, 0] > big if KR else np.zeros(m.n, bool)
    if KR:
        lines.append((m.c0 - cr[:, -1], m.wr.sum(1) > big, "solid" if m.edge_lines else None, wedge))
        for k in range(KR - 1):
            mk = (m.wr[:, k] > big) & (m.wr[:, k + 1] > big)
            lines.append((m.c0 - cr[:, k], mk, "dashed", wsep))
    if KL:
        cl = np.cumsum(m.wl, axis=1)
        lines.append((m.c0, has_r & (m.wl[:, 0] > big), "dashed" if KR + KL <= 3 else "solid", wsep))
        lines.append((m.c0 + cl[:, -1], m.wl.sum(1) > big, "solid" if m.edge_lines else None, wedge))
        for k in range(KL - 1):
            mk = (m.wl[:, k] > big) & (m.wl[:, k + 1] > big)
            lines.append((m.c0 + cl[:, k], mk, "dashed", wsep))
    elif m.oneway and m.edge_lines:
        lines.append((m.c0, has_r, "solid", wedge))

    mat = "M_Marking_White"
    for off, mask, style, width in lines:
        if style is None:
            continue
        if style == "solid":
            for a, b in runs(mask):
                for pa, pb in _pieces(b - a):
                    idx = slice(a + pa, a + pb)
                    A = m.point(off + width / 2, MARK_Z, idx)
                    B = m.point(off - width / 2, MARK_Z, idx)
                    if len(A) >= 2:
                        scene.add("Road_Marking", mat, *ml.ribbon(A, B))
        else:
            r = _dash_quads(m, off, mask, width, dash, gap, MARK_Z)
            if r is None:
                continue
            V, F, UV, cx, cy = r
            key = np.floor(cx / scene.chunk).astype(int) * 100000 + np.floor(cy / scene.chunk).astype(int)
            for kk in np.unique(key):
                q = np.nonzero(key == kk)[0]
                vi = (q[:, None] * 4 + np.arange(4)).ravel()
                remap = np.full(len(V), -1)
                remap[vi] = np.arange(len(vi))
                fsel = np.isin(F[:, 0] // 4, q)
                scene.add("Road_Marking", mat, V[vi], remap[F[fsel]], UV[vi])


def road_meshes(scene, m, elev, cfg):
    L, R = m.surface_edges()
    if m.kind == "path":
        cat, mat = "Road_Sidewalk", "M_Sidewalk"
    elif m.kind == "track":
        cat, mat = "Road_Road", "M_Gravel"
    else:
        cat, mat = "Road_Road", "M_Asphalt"
    A = m.point(L)
    B = m.point(R)
    for a, b in _pieces(m.n):
        _add_strip(scene, cat, mat, A, B, a, b)

    # side skirts hide gaps with the terrain; bridges get a thick deck instead
    depth = np.where(m.bridge, 1.2, 1.2)
    Ab = A - np.column_stack([np.zeros((m.n, 2)), depth])
    Bb = B - np.column_stack([np.zeros((m.n, 2)), depth])
    skirt_cat = cat
    for a, b in _pieces(m.n):
        _add_strip(scene, skirt_cat, mat, Ab, A, a, b)
        _add_strip(scene, skirt_cat, mat, B, Bb, a, b)

    for a, b in runs(m.bridge):
        if b - a < 2:
            continue
        for pa, pb in _pieces(b - a):
            i0, i1 = a + pa, a + pb
            scene.add("Bridge", "M_Concrete", *ml.ribbon(Bb[i0:i1], Ab[i0:i1]))
        if m.kind != "path":
            for off, sgn in ((L, -1), (R, 1)):
                P = m.point(off + sgn * 0.2)[a:b]
                for pa, pb in _pieces(b - a):
                    seg = P[pa:pb]
                    if len(seg) >= 2:
                        scene.add("Bridge", "M_Concrete", *ml.wall_along(seg[:, :2], seg[:, 2] - 0.05, 1.0, 0.3))
        # pillars
        if cfg.get("pillars", True):
            step = max(1, int(30 / STEP))
            idx = np.arange(a + step // 2, b, step)
            if len(idx):
                ground = elev.sample_xy(m.x[idx], m.y[idx])
                for i, g in zip(idx, ground):
                    h = m.z[i] - depth[i] - g
                    if h < 2.5:
                        continue
                    wdt = max(1.5, (L[i] - R[i]) * 0.6)
                    V, F, UV = ml.box(1.2, wdt, h + 1.0)
                    c = (L[i] + R[i]) / 2
                    V = ml.transform(V, m.hdg[i], (m.x[i] + m.N[i, 0] * c, m.y[i] + m.N[i, 1] * c, g - 1.0))
                    scene.add("Bridge", "M_Concrete", V, F, UV)

    if cfg.get("tunnel_tubes", True) and m.kind != "path":
        for a, b in runs(m.tunnel):
            if b - a < 2:
                continue
            H = 5.5
            for pa, pb in _pieces(b - a):
                i0, i1 = a + pa, a + pb
                At, Bt = A[i0:i1] + [0, 0, H], B[i0:i1] + [0, 0, H]
                # walls facing inwards and a ceiling facing down
                Ab0, Bb0 = A[i0:i1] - [0, 0, 0.5], B[i0:i1] - [0, 0, 0.5]
                scene.add("Wall", "M_Concrete", *ml.ribbon(At, Ab0))
                scene.add("Wall", "M_Concrete", *ml.ribbon(Bb0, Bt))
                scene.add("Wall", "M_Concrete", *ml.ribbon(Bt, At))

    if m.kind in ("road", "route") and cfg.get("markings", True):
        _markings(scene, m, cfg)


def harmonize_levels(models, blend=15.0, spacing=2.0, log=print):
    """Make roads that meet at the same level share one height.

    Roads are processed from the most important down. Where a road overlaps a
    more important one (within 1.5 m of height, so bridges are left alone)
    its height is set just under the other road, and the correction fades out
    over `blend` metres along the road. This removes both the flicker of two
    surfaces and the step a car would hit at a junction.
    """
    order = sorted(range(len(models)), key=lambda i: -models[i].rank)
    # big KD-tree over already-processed roads, rebuilt only when the small
    # "pending" part has grown by 25 %; both are queried
    big_pts, big_z, big_tree = np.zeros((0, 2)), np.zeros(0), None
    pend_pts, pend_z = [], []
    changed = 0

    def nearest_z(xy, hw):
        best = np.full(len(xy), np.inf)
        trees = []
        if big_tree is not None:
            trees.append((big_tree, big_z))
        if pend_pts:
            pp = np.concatenate(pend_pts)
            trees.append((cKDTree(pp), np.concatenate(pend_z)))
        for tree, zall in trees:
            d, j = tree.query(xy, k=8, distance_upper_bound=float(hw.max()) + 1.0)
            hit = np.isfinite(d) & (d < (hw[:, None] + 0.5))
            best = np.minimum(best, np.where(hit, zall[np.minimum(j, len(zall) - 1)], np.inf).min(axis=1))
        return best

    for n, i in enumerate(order):
        m = models[i]
        ok = ~(m.bridge | m.tunnel)
        if (big_tree is not None or pend_pts) and ok.any():
            L, R = m.surface_edges()
            hw = np.maximum(np.abs(L), np.abs(R))
            zn = nearest_z(np.column_stack([m.x, m.y]), hw)
            over = ok & np.isfinite(zn) & (np.abs(zn - m.z) < 1.5)
            if over.any():
                delta_at = (zn - 0.03 - m.z)[over]
                s_at = m.s[over]
                k = np.clip(np.searchsorted(s_at, m.s), 1, len(s_at) - 1) if len(s_at) > 1 else np.zeros(m.n, int)
                if len(s_at) > 1:
                    near = np.where(np.abs(m.s - s_at[k - 1]) < np.abs(m.s - s_at[k]), k - 1, k)
                else:
                    near = np.zeros(m.n, int)
                ds = np.abs(m.s - s_at[near])
                w = np.clip(1 - ds / blend, 0, 1)
                corr = delta_at[near] * w
                corr[over] = delta_at
                m.z = m.z + np.where(ok, corr, 0.0)
                changed += 1
        if ok.any():
            L, R = m.surface_edges()
            idx = np.nonzero(ok)[0]
            width = (L - R)[idx]
            kk = max(2, int(np.ceil(width.max() / spacing)) + 1)
            t = np.linspace(0, 1, kk)
            off = R[idx][:, None] + width[:, None] * t[None, :]
            x = m.x[idx][:, None] + m.N[idx, 0][:, None] * off
            y = m.y[idx][:, None] + m.N[idx, 1][:, None] * off
            pend_pts.append(np.column_stack([x.ravel(), y.ravel()]))
            pend_z.append(np.repeat(m.z[idx], kk))
            if sum(len(p) for p in pend_pts) > max(20000, 0.25 * len(big_z)):
                big_pts = np.concatenate([big_pts] + pend_pts)
                big_z = np.concatenate([big_z] + pend_z)
                big_tree = cKDTree(big_pts)
                pend_pts, pend_z = [], []
    log(f"  junction heights matched on {changed} road pieces")
