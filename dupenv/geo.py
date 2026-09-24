# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Local metric frame and polyline helpers.

All geometry is built in a local East-North-Up frame (metres) given by a
transverse Mercator projection centred on the route. The same frame is used
for the FBX mesh, the OpenDRIVE file (x, y) and the Lanelet2 local_x/local_y
tags, so every output lines up.
"""
import threading

import numpy as np
from pyproj import Transformer


_geom_local = threading.local()


def thread_geom(g):
    """This thread's own prepared copy of a shapely geometry. GEOS builds the
    index of a prepared geometry lazily on first use, which corrupts memory
    when two threads query the same geometry at once."""
    if g is None:
        return None
    import shapely
    d = getattr(_geom_local, "d", None)
    if d is None:
        d = _geom_local.d = {}
    c = d.get(id(g))
    if c is None or c[0] is not g:
        copy = shapely.from_wkb(shapely.to_wkb(g))
        shapely.prepare(copy)
        c = d[id(g)] = (g, copy)
    return c[1]


class LocalFrame:
    def __init__(self, lat0, lon0):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.proj_string = (
            f"+proj=tmerc +lat_0={self.lat0:.9f} +lon_0={self.lon0:.9f} +k=1 "
            "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
        )
        # pyproj transformers must not be shared between threads (the
        # downloads and terrain tiles run in parallel): one pair per thread
        self._local = threading.local()

    def _tr(self):
        loc = self._local
        if not hasattr(loc, "fwd"):
            loc.fwd = Transformer.from_crs("EPSG:4326", self.proj_string, always_xy=True)
            loc.inv = Transformer.from_crs(self.proj_string, "EPSG:4326", always_xy=True)
        return loc

    def to_xy(self, lon, lat):
        x, y = self._tr().fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._tr().inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)

    def __getstate__(self):
        return {"lat0": self.lat0, "lon0": self.lon0, "proj_string": self.proj_string}

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._local = threading.local()


def cumlen(P):
    P = np.asarray(P)
    if len(P) < 2:
        return np.zeros(len(P))
    d = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def dedupe(P, eps=0.05):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i, :2] - P[keep[-1], :2])) > eps:
            keep.append(i)
    return P[keep]


def resample(P, step):
    """Resample a polyline at a (nearly) constant step. Returns (points, s)."""
    P = dedupe(P)
    s = cumlen(P)
    L = s[-1]
    n = max(2, int(np.ceil(L / step)) + 1)
    ss = np.linspace(0.0, L, n)
    out = np.stack([np.interp(ss, s, P[:, k]) for k in range(P.shape[1])], axis=1)
    return out, ss


def smooth_polyline(P, window):
    """Moving average on x/y that keeps both end points fixed."""
    if window < 3 or len(P) < window:
        return P
    k = np.ones(window) / window
    pad = window // 2
    out = P.copy()
    for c in range(2):
        v = np.pad(P[:, c], pad, mode="reflect", reflect_type="odd")
        out[:, c] = np.convolve(v, k, mode="valid")[: len(P)]
    out[0], out[-1] = P[0], P[-1]
    return out


def tangents(P):
    P = np.asarray(P)[:, :2]
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    n = np.linalg.norm(T, axis=1, keepdims=True)
    n[n == 0] = 1
    return T / n


def left_normals(T):
    return np.stack([-T[:, 1], T[:, 0]], axis=1)


def offset_points(P, N, d):
    """Offset points P along left normals N by distance d (scalar or per point)."""
    d = np.broadcast_to(np.asarray(d, float), (len(P),))
    return P[:, :2] + N * d[:, None]


def gaussian_smooth(v, sigma_samples):
    if sigma_samples <= 0.5 or len(v) < 3:
        return v.copy()
    r = int(3 * sigma_samples)
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma_samples) ** 2)
    k /= k.sum()
    vp = np.pad(v, r, mode="edge")
    return np.convolve(vp, k, mode="valid")


def box_smooth(v, half):
    """Box filter along axis 0 (turns steps into linear ramps)."""
    if half < 1 or len(v) < 3:
        return v.copy()
    k = np.ones(2 * half + 1) / (2 * half + 1)
    if v.ndim == 1:
        return np.convolve(np.pad(v, half, mode="edge"), k, mode="valid")
    return np.stack([box_smooth(v[:, j], half) for j in range(v.shape[1])], axis=1)


def dp_simplify(P, tol):
    """Douglas-Peucker on 2D points. Returns sorted indices of kept points."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        A, B = P[a], P[b]
        seg = B - A
        L = np.hypot(*seg)
        Q = P[a + 1 : b]
        if L < 1e-9:
            d = np.hypot(Q[:, 0] - A[0], Q[:, 1] - A[1])
        else:
            d = np.abs(seg[0] * (Q[:, 1] - A[1]) - seg[1] * (Q[:, 0] - A[0])) / L
        i = int(np.argmax(d))
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.nonzero(keep)[0]


def runs(mask):
    """Start/end (exclusive) index pairs of True runs in a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))
