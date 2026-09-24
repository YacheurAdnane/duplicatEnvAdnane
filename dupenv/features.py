# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Everything around the road: terrain, buildings, vegetation, barriers,
panels, poles, railways, water, exit panels and inferred guard rails."""
import math
import re
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box as sbox
from shapely.ops import unary_union

from . import meshlib as ml
from . import groundtex
from .geo import thread_geom, cumlen, resample, runs
from .net import http_get
from .roads import STEP, RoadModel, _num, is_bridge, is_tunnel, road_elevation
from .textures import PanelFactory, classify

IGN_WMS = "https://data.geopf.fr/wms-r/wms"


COMMON_MATERIALS = {
    "M_Metal": (0.62, 0.64, 0.66), "M_Concrete": (0.66, 0.65, 0.62), "M_Wood": (0.40, 0.30, 0.20),
    "M_Dark": (0.12, 0.12, 0.13), "M_Orange": (0.95, 0.45, 0.05), "M_WallDefault": (0.82, 0.78, 0.72),
    "M_Roof": (0.45, 0.40, 0.38), "M_Bush": (0.28, 0.42, 0.18),
}


def register_common_materials(scene):
    for k, v in COMMON_MATERIALS.items():
        scene.material(k, v)


# ================================================================ road surfaces

def surface_samples(models, which, spacing=1.5):
    """Points covering road surfaces: which = ground | bridge | tunnel.

    Returns points (k,3), model index (k,), unit direction (k,2)."""
    P, M, D = [], [], []
    for mi, m in enumerate(models):
        if which == "ground":
            ok = ~(m.bridge | m.tunnel)
        elif which == "bridge":
            ok = m.bridge
        else:
            ok = m.tunnel
        if not ok.any():
            continue
        L, R = m.surface_edges()
        idx = np.nonzero(ok)[0]
        width = (L - R)[idx]
        k = max(2, int(np.ceil(width.max() / spacing)) + 1)
        t = np.linspace(0, 1, k)
        off = R[idx][:, None] + width[:, None] * t[None, :]
        x = m.x[idx][:, None] + m.N[idx, 0][:, None] * off
        y = m.y[idx][:, None] + m.N[idx, 1][:, None] * off
        z = np.repeat(m.z[idx][:, None], k, axis=1)
        P.append(np.column_stack([x.ravel(), y.ravel(), z.ravel()]))
        M.append(np.full(x.size, mi))
        D.append(np.repeat(m.T[idx], k, axis=0))
    if not P:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros((0, 2))
    return np.concatenate(P), np.concatenate(M), np.concatenate(D)


class Ground:
    """Terrain height = DEM, flattened under roads and blended back to the DEM."""

    def __init__(self, elev, models, res, blend=10.0):
        self.elev = elev
        self.fp, self.fp_mid, self.fp_dir = surface_samples(models, "ground")
        self.bp, _, _ = surface_samples(models, "bridge")
        self.tp, _, _ = surface_samples(models, "tunnel")
        self.tree = cKDTree(self.fp[:, :2]) if len(self.fp) else None
        road_idx = [k for k, m in enumerate(models) if m.kind in ("route", "road", "track")]
        is_road = np.isin(self.fp_mid, road_idx)
        self.atree = cKDTree(self.fp[is_road, :2]) if is_road.any() else None
        self.btree = cKDTree(self.bp[:, :2]) if len(self.bp) else None
        self.ttree = cKDTree(self.tp[:, :2]) if len(self.tp) else None
        self.inner = 0.8 + 0.6 * res
        self.outer = self.inner + blend
        # any terrain vertex this close to a road point stays below it, so no
        # terrain triangle can rise above the asphalt (grid diagonal + margin)
        self.clamp_r = res * 1.5 + 0.5

    @classmethod
    def from_network(cls, elev, P, D, asphalt, bridge, tunnel, res, blend=10.0, extra=None):
        """Same terrain model, but fitted to the OpenDRIVE surface points.
        `extra` (list of RoadModel, e.g. railways and footpaths) is added."""
        g = cls.__new__(cls)
        g.elev = elev
        ok = ~(bridge | tunnel)
        fp, fdir, fmid = P[ok], D[ok], np.zeros(ok.sum(), int)
        is_road = asphalt[ok]
        if extra:
            ep, emid, edir = surface_samples(extra, "ground")
            fp = np.concatenate([fp, ep])
            fdir = np.concatenate([fdir, edir])
            fmid = np.concatenate([fmid, emid + 1])
            is_road = np.concatenate([is_road, np.zeros(len(ep), bool)])
        g.fp, g.fp_dir, g.fp_mid = fp, fdir, fmid
        g.bp = P[bridge]
        g.tp = P[tunnel]
        g.tree = cKDTree(g.fp[:, :2]) if len(g.fp) else None
        g.btree = cKDTree(g.bp[:, :2]) if len(g.bp) else None
        g.ttree = cKDTree(g.tp[:, :2]) if len(g.tp) else None
        g.atree = cKDTree(g.fp[is_road, :2]) if is_road.any() else None
        g.inner = 0.8 + 0.6 * res
        g.outer = g.inner + blend
        g.clamp_r = res * 1.5 + 0.5
        return g

    def height(self, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        z = self.elev.sample_xy(x, y)
        if self.tree is not None:
            d, j = self.tree.query(np.column_stack([x, y]), distance_upper_bound=self.outer)
            m = np.isfinite(d)
            if m.any():
                zt = self.fp[j[m], 2] - 0.12
                t = np.clip((d[m] - self.inner) / (self.outer - self.inner), 0, 1)
                t = t * t * (3 - 2 * t)
                z[m] = zt * (1 - t) + z[m] * t
            d, j = self.tree.query(np.column_stack([x, y]), k=32, distance_upper_bound=self.clamp_r)
            near = np.isfinite(d)
            if near.any():
                zz = np.where(near, self.fp[np.minimum(j, len(self.fp) - 1), 2], np.inf).min(axis=1)
                z = np.minimum(z, zz - 0.15)
        if self.btree is not None:  # keep the ground under bridge decks (1.2 m thick)
            d, j = self.btree.query(np.column_stack([x, y]), k=16, distance_upper_bound=self.clamp_r)
            near = np.isfinite(d)
            if near.any():
                zz = np.where(near, self.bp[np.minimum(j, len(self.bp) - 1), 2], np.inf).min(axis=1)
                z = np.minimum(z, zz - 1.5)
        return z

    def surface(self, x, y):
        """Height of whatever is walkable/drivable at x,y (road, bridge deck, ground)."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        z = self.height(x, y)
        xy = np.column_stack([x, y])
        if self.tree is not None:
            d, j = self.tree.query(xy, distance_upper_bound=1.2)
            m = np.isfinite(d)
            z[m] = self.fp[j[m], 2]
        if self.btree is not None:
            d, j = self.btree.query(xy, distance_upper_bound=1.5)
            m = np.isfinite(d)
            z[m] = np.maximum(z[m], self.bp[j[m], 2])
        return z

    def near_road(self, x, y, dist):
        out = np.zeros(len(x), bool)
        for tr in (self.tree, self.btree):
            if tr is not None:
                d, _ = tr.query(np.column_stack([x, y]), distance_upper_bound=dist)
                out |= np.isfinite(d)
        return out

    def on_asphalt(self, x, y, dist):
        """Near a real road surface (railway ballast and footpaths excluded)."""
        if self.atree is None:
            return np.zeros(len(x), bool)
        d, _ = self.atree.query(np.column_stack([x, y]), distance_upper_bound=dist)
        return np.isfinite(d)

    def over_tunnel(self, x, y, dist=1.5):
        if self.ttree is None:
            return np.zeros(len(x), bool)
        d, _ = self.ttree.query(np.column_stack([x, y]), distance_upper_bound=dist)
        return np.isfinite(d)


class RoadIndex:
    """Nearest road centreline sample, for orienting panels and lamps."""

    def __init__(self, models):
        pts, mi, si = [], [], []
        for k, m in enumerate(models):
            if m.kind in ("route", "road"):
                pts.append(np.column_stack([m.x, m.y]))
                mi.append(np.full(m.n, k))
                si.append(np.arange(m.n))
        self.models = models
        if pts:
            self.pts = np.concatenate(pts)
            self.mi = np.concatenate(mi)
            self.si = np.concatenate(si)
            self.tree = cKDTree(self.pts)
        else:
            self.tree = None

    def nearest(self, x, y, maxd=30.0):
        if self.tree is None:
            return None
        d, j = self.tree.query([x, y], distance_upper_bound=maxd)
        if not np.isfinite(d):
            return None
        return self.models[self.mi[j]], int(self.si[j])


def _yaw_facing(fx, fy):
    """Yaw so that a panel's local -y front faces direction (fx, fy)."""
    return math.atan2(fx, -fy)


# ================================================================ terrain

def terrain_tiles(corridor, res, tile=500.0):
    area = corridor.buffer(res * 1.5)
    minx, miny, maxx, maxy = area.bounds
    tx0, ty0 = int(math.floor(minx / tile)), int(math.floor(miny / tile))
    tx1, ty1 = int(math.floor(maxx / tile)), int(math.floor(maxy / tile))
    return [(i, j) for i in range(tx0, tx1 + 1) for j in range(ty0, ty1 + 1)
            if area.intersects(sbox(i * tile, j * tile, (i + 1) * tile, (j + 1) * tile))]


def _grid_class_weights(landcover, buildings_t, X0, Y0, nx, ny, res, margin=3):
    """Blurred one-hot ground classes on the 4 m cell grid of a tile, with a
    margin so blends continue across tile borders. Shape (7, ny+2m, nx+2m)."""
    cx = X0 + (np.arange(-margin, nx + margin) + 0.5) * res
    cy = Y0 + (np.arange(-margin, ny + margin) + 0.5) * res
    CX, CY = np.meshgrid(cx, cy)
    near_b = (shapely.dwithin(buildings_t, shapely.points(CX.ravel(), CY.ravel()), 25.0)
              if buildings_t is not None else None)
    cls = landcover.ground_class(CX.ravel(), CY.ravel(), near_b).reshape(CX.shape)
    from scipy import ndimage
    return np.stack([ndimage.gaussian_filter((cls == k).astype(np.float32), 1.3) for k in range(7)])


def _weights_at(W, px, py, X0, Y0, res, margin=3):
    from scipy import ndimage
    rr = (np.asarray(py) - Y0) / res - 0.5 + margin
    cc = (np.asarray(px) - X0) / res - 0.5 + margin
    return np.stack([ndimage.map_coordinates(w, [rr, cc], order=1, mode="nearest") for w in W])


def _noisy_class(W, px, py, X0, Y0, res, amp=0.75, mix=0.0):
    """Ground class at points: blended weights plus a different fractal noise
    per class, so borders become organic patches over a band of metres.
    With mix > 0, points where the two leading classes are that close get a
    mixed material of the pair (a < b): code 100 + 10 * a + b where a leads,
    200 + 10 * a + b where b leads."""
    w = _weights_at(W, px, py, X0, Y0, res)
    for k in range(len(w)):
        w[k] += amp * (groundtex.fractal(px, py, 5.0, seed=17 * k + 3, octaves=3) - 0.5)
    order = np.argsort(-w, axis=0)
    c1 = order[0].astype(np.int16)
    if mix <= 0:
        return c1
    c2 = order[1].astype(np.int16)
    w1 = np.take_along_axis(w, order[:1], 0)[0]
    w2 = np.take_along_axis(w, order[1:2], 0)[0]
    close = (w1 - w2) < mix
    lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
    return np.where(close, np.where(c1 == lo, 100, 200) + 10 * lo + hi, c1).astype(np.int16)



def build_terrain(scene, ground, corridor, frame, cfg, tex_dir, log, progress=None, landcover=None,
                  buildings=None):
    res = float(cfg.get("terrain_res", 4.0))
    tile = 500.0
    ortho = cfg.get("ortho", False) and cfg.get("ground_style", "landcover") == "ortho"
    area = corridor.buffer(res * 1.5)
    minx, miny, maxx, maxy = area.bounds
    tx0, ty0 = int(math.floor(minx / tile)), int(math.floor(miny / tile))
    tx1, ty1 = int(math.floor(maxx / tile)), int(math.floor(maxy / tile))
    tiles = [(i, j) for i in range(tx0, tx1 + 1) for j in range(ty0, ty1 + 1)
             if area.intersects(sbox(i * tile, j * tile, (i + 1) * tile, (j + 1) * tile))]
    log(f"  terrain: {len(tiles)} tiles of {tile:.0f} m, grid {res} m, ortho={'on' if ortho else 'off'}")
    scene.material("M_Terrain", (0.33, 0.42, 0.22))

    ortho_jobs = []

    quality = cfg.get("ground_quality", "high")
    Q = groundtex.QUALITY.get(quality, groundtex.QUALITY["high"])
    lc = landcover is not None and not ortho
    sub = Q["sub"] if lc else 1
    uv_m = groundtex.TILE_M if (lc and Q["res"]) else 8.0
    want_details = lc and Q["details"] and cfg.get("ground_details", True)
    fine_m = float(cfg.get("ground_fine_m", 30.0))  # fine transition mesh within this distance of a road
    palette = groundtex.low_palette()

    def one_tile(ij):
        """Meshes of one 500 m tile as a list of scene.add() calls, plus scatter
        points for grass tufts and rocks (run in a worker thread)."""
        from .progress import wait_for_ram
        wait_for_ram()
        i, j = ij
        adds, scatter = [], None
        area_t, buildings_t = thread_geom(area), thread_geom(buildings)
        X0, Y0 = i * tile, j * tile
        xs = np.arange(X0, X0 + tile + res * 0.5, res)
        ys = np.arange(Y0, Y0 + tile + res * 0.5, res)
        X, Y = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(area_t, X, Y)
        if not inside.any():
            return adds, scatter
        Z = np.full(X.shape, np.nan)
        Z[inside] = ground.height(X[inside], Y[inside])
        ny, nx = X.shape[0] - 1, X.shape[1] - 1
        ok = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
        if not ok.any():
            return adds, scatter
        tag = f"{'m' if i < 0 else ''}{abs(i)}_{'m' if j < 0 else ''}{abs(j)}"
        W = _grid_class_weights(landcover, buildings_t, X0, Y0, nx, ny, res) if lc else None

        # --- faces on the fine vertex grid (s sub-cells per 4 m cell where ground types mix)
        s = sub
        rr, cc = np.nonzero(ok)
        fcls = None
        if lc and Q["res"]:
            # class of every sub-cell of every cell
            # only cells whose neighbourhood mixes ground types can hold a border:
            # elsewhere one class per cell, from its centre
            wc = _weights_at(W, X0 + (cc + 0.5) * res, Y0 + (rr + 0.5) * res, X0, Y0, res)
            pure = wc.max(axis=0) > 0.92
            cell_cls = np.argmax(wc, axis=0).astype(np.int16)
            a_, b_ = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
            sub_r = (rr[:, None] * s + a_.ravel()[None, :])
            sub_c = (cc[:, None] * s + b_.ravel()[None, :])
            # class per triangle (centroids of the two halves of each sub-cell), so
            # borders get diagonal edges instead of 1 m steps
            p1x, p1y = X0 + (sub_c + 2 / 3) * res / s, Y0 + (sub_r + 1 / 3) * res / s
            p2x, p2y = X0 + (sub_c + 1 / 3) * res / s, Y0 + (sub_r + 2 / 3) * res / s
            t1 = np.repeat(cell_cls[:, None], s * s, axis=1)
            t2 = t1.copy()
            # fine transition mesh only where the car sees it: near the road
            if ground.atree is not None:
                dcell, _ = ground.atree.query(np.column_stack([X0 + (cc + 0.5) * res, Y0 + (rr + 0.5) * res]),
                                              distance_upper_bound=fine_m)
                near = np.isfinite(dcell)
            else:
                near = np.ones(len(rr), bool)
            # far border cells: 4 m cells with a class per triangle
            f_idx = np.nonzero(~pure & ~near)[0]
            cA, cB = cell_cls.copy(), cell_cls.copy()
            if len(f_idx):
                fx, fy = X0 + cc[f_idx] * res, Y0 + rr[f_idx] * res
                two = _noisy_class(W, np.concatenate([fx + res * 2 / 3, fx + res / 3]),
                                   np.concatenate([fy + res / 3, fy + res * 2 / 3]), X0, Y0, res, mix=0.3)
                cA[f_idx], cB[f_idx] = two[:len(f_idx)], two[len(f_idx):]
            b_idx = np.nonzero(~pure & near)[0]
            if len(b_idx):
                n_ = b_idx.size * s * s
                both = _noisy_class(W, np.concatenate([p1x[b_idx].ravel(), p2x[b_idx].ravel()]),
                                    np.concatenate([p1y[b_idx].ravel(), p2y[b_idx].ravel()]), X0, Y0, res, mix=0.3)
                t1[b_idx] = both[:n_].reshape(-1, s * s)
                t2[b_idx] = both[n_:].reshape(-1, s * s)
            scls = t1
            mixed = near & ((t1 != t1[:, :1]).any(axis=1) | (t2 != t1[:, :1]).any(axis=1))
            cA[near], cB[near] = t1[near, 0], t1[near, 0]
        else:
            mixed = np.zeros(len(rr), bool)
            if lc:  # low: class only matters for the baked colours
                scls = None
        vid = lambda R, C: R * (nx * s + 1) + C  # noqa: E731  fine vertex id
        Fl, Cl = [], []
        u = ~mixed
        R0, C0 = rr[u] * s, cc[u] * s
        a, b = vid(R0, C0), vid(R0, C0 + s)
        c, d = vid(R0 + s, C0 + s), vid(R0 + s, C0)
        Fl.append(np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]))
        if lc and Q["res"]:
            Cl.append(np.concatenate([cA[u], cB[u]]))
        if mixed.any():
            m_r, m_c = sub_r[mixed].ravel(), sub_c[mixed].ravel()
            a, b = vid(m_r, m_c), vid(m_r, m_c + 1)
            c, d = vid(m_r + 1, m_c + 1), vid(m_r + 1, m_c)
            Fl.append(np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]))
            Cl.append(np.concatenate([t1[mixed].ravel(), t2[mixed].ravel()]))
        F = np.concatenate(Fl)
        fcls = np.concatenate(Cl) if Cl else None
        used = np.unique(F)
        R, C = used // (nx * s + 1), used % (nx * s + 1)
        # heights: bilinear inside the 4 m cell, so edges shared with plain cells stay straight
        r0, c0 = np.minimum(R // s, ny - 1), np.minimum(C // s, nx - 1)
        fr, fc = (R - r0 * s) / s, (C - c0 * s) / s
        Zc = np.nan_to_num(Z)
        z = (Zc[r0, c0] * (1 - fr) * (1 - fc) + Zc[r0, c0 + 1] * (1 - fr) * fc
             + Zc[r0 + 1, c0] * fr * (1 - fc) + Zc[r0 + 1, c0 + 1] * fr * fc)
        V = np.column_stack([X0 + C * res / s, Y0 + R * res / s, z])
        remap = np.full(int(used.max()) + 1, -1)
        remap[used] = np.arange(len(used))
        F = remap[F]

        cen = V[F].mean(axis=1)
        keep = ~ground.over_tunnel(cen[:, 0], cen[:, 1], 2.0)
        # cells completely under a road are hidden anyway: drop them (corners
        # and centre must all be on asphalt, so narrow medians between two
        # carriageways are kept)
        on_road = ground.on_asphalt(V[:, 0], V[:, 1], 0.8)
        keep &= ~(on_road[F].all(axis=1) & ground.on_asphalt(cen[:, 0], cen[:, 1], 0.5))
        F = F[keep]
        if fcls is not None:
            fcls = fcls[keep]
        if len(F) == 0:
            return adds, scatter
        used = np.unique(F)
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        V, F = V[used], remap[F]
        if ortho:
            lon, lat = frame.to_lonlat(V[:, 0], V[:, 1])
            bb = (lon.min() - 1e-6, lat.min() - 1e-6, lon.max() + 1e-6, lat.max() + 1e-6)
            UV = np.column_stack([(lon - bb[0]) / (bb[2] - bb[0]), (lat - bb[1]) / (bb[3] - bb[1])])
            fn = f"ortho_{tag}.jpg"
            adds.append(("ortho", tag, fn, bb, V, F, UV))
        elif lc and not Q["res"]:
            # low: one baked texture per tile, flat colours blended over metres
            # the texture reaches one cell past the tile on each side, so texture
            # filtering never wraps round at the tile border (that showed a seam)
            N, span, x_0, y_0 = 512, tile + 3 * res, X0 - res, Y0 - res
            pxs = x_0 + (np.arange(N) + 0.5) * span / N
            pys = y_0 + span - (np.arange(N) + 0.5) * span / N  # row 0 = north
            PX, PY = np.meshgrid(pxs, pys)
            w = _weights_at(W, PX.ravel(), PY.ravel(), X0, Y0, res) ** 3  # keep each type's own colour
            col = (w.T @ palette) / np.maximum(w.sum(axis=0), 1e-6)[:, None]
            col *= (0.95 + 0.10 * groundtex.fractal(PX.ravel(), PY.ravel(), 25.0, seed=5))[:, None]
            img = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(N, N, 3))
            fn = f"groundtile_{tag}.jpg"
            img.save(os.path.join(tex_dir, fn), quality=90)
            UV = np.column_stack([(V[:, 0] - x_0) / span, (V[:, 1] - y_0) / span])
            adds.append(("tile", tag, fn, V, F, UV))
        elif fcls is not None:
            for c_id in np.unique(fcls):
                Fc = F[fcls == c_id]
                used_c = np.unique(Fc)
                rm = np.full(len(V), -1)
                rm[used_c] = np.arange(len(used_c))
                # material resolved in the main thread (mixed pairs are made on demand)
                adds.append(("Ground", int(c_id), V[used_c], rm[Fc], V[used_c, :2] / uv_m, f"Ground_{tag}"))
        else:
            adds.append(("Terrain", "M_Terrain", V, F, V[:, :2] / 8.0, f"Terrain_{tag}"))
        if want_details:
            scatter = _scatter_points(W, ok, X0, Y0, res, ground, buildings_t, (i * 7919 + j * 104729) % 2147483647)
        return adds, scatter

    from .progress import workers
    nw = workers("cpu")
    scatter_all = []
    log(f"  {nw} terrain workers, ground quality {quality if lc else 'n/a'}")
    with ThreadPoolExecutor(nw) as ex:
        # map keeps tile order, so the scene is the same as a serial run
        results = []
        for n, res_ in enumerate(ex.map(one_tile, tiles)):
            results.append(res_)
            if progress:
                progress((n + 1) / len(tiles) * (0.5 if ortho else 0.8), f"tile {n + 1}/{len(tiles)}")
        # mixed materials of the pairs that occur, made in parallel
        from .landcover import CLASS_MATERIAL
        codes = sorted({ad[1] for adds, _ in results for ad in adds
                        if ad[0] == "Ground" and isinstance(ad[1], int) and ad[1] >= 100})

        def make_mix(code):
            c = code % 100
            return code, groundtex.mix_material(scene, tex_dir, CLASS_MATERIAL[c // 10], CLASS_MATERIAL[c % 10],
                                                quality, share=0.7 if code < 200 else 0.3)
        mix_mat = dict(ex.map(make_mix, codes))
        if codes:
            log(f"  {len(codes)} transition textures between ground types")
        for n, (adds, sc) in enumerate(results):
            if sc is not None:
                scatter_all.append(sc)
            for ad in adds:
                if ad[0] == "tile":
                    _, tag, fn, V, F, UV = ad
                    mat = f"M_GroundTile_{tag}"
                    scene.material(mat, (0.4, 0.45, 0.25), texture=f"textures/{fn}")
                    scene.add("Ground", mat, V, F, UV, name=f"Ground_{tag}", smooth=True)
                elif ad[0] == "ortho":
                    _, tag, fn, bb, V, F, UV = ad
                    mat = f"M_Ortho_{tag}"
                    scene.material(mat, (1, 1, 1), texture=f"textures/{fn}")
                    ortho_jobs.append((bb, os.path.join(tex_dir, fn)))
                    scene.add("Ground", mat, V, F, UV, name=f"Ground_{tag}", smooth=True)
                else:
                    cat, mat, V, F, UV, nm = ad
                    if isinstance(mat, int):
                        mat = mix_mat[mat] if mat >= 100 else CLASS_MATERIAL[mat]
                    scene.add(cat, mat, V, F, UV, name=nm, smooth=True)
            results[n] = None
        if progress:
            progress(0.6 if ortho else 1.0)

    if ortho_jobs:
        px = int(min(2048, max(256, tile / float(cfg.get("ortho_res", 0.5)))))

        def fetch(job):
            bb, path = job
            url = (f"{IGN_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS"
                   f"&STYLES=&CRS=EPSG:4326&BBOX={bb[1]:.9f},{bb[0]:.9f},{bb[3]:.9f},{bb[2]:.9f}"
                   f"&WIDTH={px}&HEIGHT={px}&FORMAT=image/jpeg")
            try:
                raw = http_get(url, timeout=120, retries=6)
                with open(path, "wb") as fh:
                    fh.write(raw)
                Image.open(path).verify()
            except Exception:
                Image.new("RGB", (8, 8), (84, 107, 56)).save(path)
            return path

        log(f"  downloading {len(ortho_jobs)} IGN orthophoto tiles ({px}x{px} px)")
        with ThreadPoolExecutor(10) as ex:
            for k, _ in enumerate(ex.map(fetch, ortho_jobs)):
                if progress:
                    progress(0.6 + 0.4 * (k + 1) / len(ortho_jobs))
    return scatter_all


# ================================================================ ground details

GREEN = (0, 1)        # GRASS, MEADOW: grass tufts (flowers in meadows)
STONY = (4, 6, 3)     # FOREST, BARE, FIELD: small rocks


def _scatter_points(W, ok, X0, Y0, res, ground, buildings_t, seed):
    """Candidate spots for grass tufts and rocks in one tile: denser near the
    road, where the cameras and the LiDAR look. Returns (kind, x, y, cls, dist)."""
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(ok)
    per_cell = 16  # candidates per 4 m cell (16 m2), thinned below and by the budget
    k = rng.integers(0, len(rr), len(rr) * per_cell)
    x = X0 + (cc[k] + rng.random(len(k))) * res
    y = Y0 + (rr[k] + rng.random(len(k))) * res
    if ground.atree is not None:
        d, _ = ground.atree.query(np.column_stack([x, y]), distance_upper_bound=60.0)
    else:
        d = np.full(len(x), 60.0)
    d = np.minimum(d, 60.0)
    keep = d > 1.2  # never on the asphalt or its edge
    if buildings_t is not None:
        keep &= ~shapely.contains_xy(buildings_t, x, y)
    x, y, d = x[keep], y[keep], d[keep]
    cls = _noisy_class(W, x, y, X0, Y0, res)
    green = np.isin(cls, GREEN)
    stony = np.isin(cls, STONY)
    # expected tufts per m2 near the road, fading over ~12 m; rocks sparser
    p_green = np.clip(4.0 * np.exp(-(d - 1.2) / 12.0), 0, 1) * green
    p_stone = np.clip(0.6 * np.exp(-(d - 1.2) / 20.0) + 0.05, 0, 1) * stony
    u = rng.random(len(x))
    tuft = u < p_green
    rock = (~tuft) & (u < p_stone)
    kind = np.where(tuft, 1, np.where(rock, 2, 0))
    m = kind > 0
    return np.column_stack([kind[m], x[m], y[m], cls[m], d[m]]).astype(np.float32)


def _blade_texture(path):
    """Grass blade colours: 6 greens (dark base to light tip) and 3 flower colours."""
    h = 128
    cols = [((0.10, 0.22, 0.05), (0.42, 0.60, 0.20)), ((0.12, 0.25, 0.06), (0.50, 0.62, 0.22)),
            ((0.14, 0.24, 0.07), (0.62, 0.64, 0.30)), ((0.09, 0.20, 0.06), (0.36, 0.52, 0.18)),
            ((0.16, 0.26, 0.08), (0.70, 0.66, 0.36)), ((0.11, 0.23, 0.05), (0.46, 0.58, 0.16)),
            ((0.95, 0.95, 0.92), (0.98, 0.98, 0.95)), ((0.95, 0.80, 0.15), (0.98, 0.86, 0.25)),
            ((0.60, 0.42, 0.78), (0.70, 0.52, 0.85))]
    img = np.zeros((h, 16 * len(cols), 3), np.float32)
    v = np.linspace(1, 0, h)[:, None]  # image row 0 = top = tip
    for k, (base, tip) in enumerate(cols):
        img[:, 16 * k:16 * (k + 1)] = (np.array(base) * (1 - v) + np.array(tip) * v)[:, None, :]
    Image.fromarray((img * 255).astype(np.uint8)).save(path, quality=92)
    return len(cols)


def build_ground_details(scene, scatter, ground, cfg, tex_dir, rng, log):
    """3D grass tufts (with flowers in meadows) and small rocks, from the
    scatter points of the terrain tiles, within a triangle budget. The
    closest spots to the road are kept first."""
    if not scatter:
        return
    P = np.concatenate(scatter)
    if not len(P):
        return
    n_cols = _blade_texture(os.path.join(tex_dir, "grass_blades.jpg"))
    scene.material("M_GrassBlade", (0.30, 0.45, 0.15), texture="textures/grass_blades.jpg")
    if "M_Rock" not in scene.materials:
        scene.material("M_Rock", (0.45, 0.43, 0.40))
    budget = float(cfg.get("detail_tri_budget", 1.5e6))
    blades = 16
    tuft_tris = 2 * blades
    rock_tris = 20
    T, Rk = P[P[:, 0] == 1], P[P[:, 0] == 2]
    # rocks take at most 20 % of the budget; priority to what is near the road
    n_rock = int(min(len(Rk), 0.2 * budget / rock_tris))
    n_tuft = int(min(len(T), (budget - n_rock * rock_tris) / tuft_tris))
    if n_tuft < len(T):
        T = T[np.argsort(T[:, 4] + rng.random(len(T)) * 6.0)[:n_tuft]]
    if n_rock < len(Rk):
        Rk = Rk[np.argsort(Rk[:, 4] + rng.random(len(Rk)) * 10.0)[:n_rock]]

    def by_chunk(A):
        key = np.floor(A[:, 1] / scene.chunk).astype(np.int64) * 100000 + np.floor(A[:, 2] / scene.chunk).astype(np.int64)
        for kv in np.unique(key):
            yield A[key == kv]

    # a share of the tufts become RoadRunner library plants (small bushes)
    rr = T[rng.random(len(T)) < min(1.0, 4000 / max(len(T), 1))]
    zr = ground.height(rr[:, 1], rr[:, 2]) if len(rr) else []
    for q, zz in zip(rr, zr):
        scene.record("grass", pos=[float(q[1]), float(q[2]), float(zz)],
                     size=float(rng.uniform(0.35, 0.8) * (1.3 if q[3] == 1 else 1.0)), yaw=float(rng.uniform(0, 6.283)))
    # ---- grass tufts: blades as thin double-sided triangles
    for A in by_chunk(T):
        n = len(A)
        z = ground.height(A[:, 1], A[:, 2])
        k = n * blades
        cx = np.repeat(A[:, 1], blades) + rng.normal(0, 0.13, k)
        cy = np.repeat(A[:, 2], blades) + rng.normal(0, 0.13, k)
        cz = np.repeat(z, blades) - 0.02
        meadow = np.repeat(A[:, 3] == 1, blades)
        # mixed heights: mostly short tufts, some knee-high clumps in meadows
        size = np.repeat(rng.choice([1.0, 1.6, 2.4], n, p=[0.7, 0.25, 0.05]), blades)
        size = np.where(meadow, size, np.minimum(size, 1.6))
        h = rng.uniform(0.16, 0.40, k) * size * np.where(meadow, 1.2, 1.0)
        ang = rng.uniform(0, 2 * np.pi, k)
        lean = rng.uniform(0.05, 0.35, k)
        wdt = rng.uniform(0.018, 0.035, k)
        ex, ey = np.cos(ang) * wdt, np.sin(ang) * wdt          # blade width across
        lx, ly = -np.sin(ang) * lean * h, np.cos(ang) * lean * h  # lean
        v0 = np.column_stack([cx - ex, cy - ey, cz])
        v1 = np.column_stack([cx + ex, cy + ey, cz])
        v2 = np.column_stack([cx + lx, cy + ly, cz + h])
        V = np.stack([v0, v1, v2], 1).reshape(-1, 3)
        # blade colours follow the ground: yellow-green on meadows, greener on lawns
        col = np.where(meadow, rng.choice([2, 4, 1], k), rng.choice([0, 1, 3, 5], k))
        u = (col + 0.5) / n_cols
        UV = np.stack([np.column_stack([u, np.full(k, 0.02)]), np.column_stack([u, np.full(k, 0.02)]),
                       np.column_stack([u, np.full(k, 0.98)])], 1).reshape(-1, 2)
        b = np.arange(k) * 3
        F = np.concatenate([np.stack([b, b + 1, b + 2], 1), np.stack([b, b + 2, b + 1], 1)])
        # flowers: a small upward square on some meadow blades
        fl = meadow & (rng.random(k) < 0.03)
        if fl.any():
            q = np.nonzero(fl)[0]
            top = V[b[q] + 2]
            r_ = rng.uniform(0.015, 0.03, len(q))
            corners = np.stack([top + np.column_stack([dx * r_, dy * r_, np.zeros(len(q))])
                                for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))], 1).reshape(-1, 3)
            fcol = rng.integers(6, n_cols, len(q))
            fu = np.repeat((fcol + 0.5) / n_cols, 4)
            fUV = np.column_stack([fu, np.full(len(fu), 0.5)])
            o = len(V) + np.arange(len(q)) * 4
            Ff = np.concatenate([np.stack([o, o + 1, o + 2], 1), np.stack([o, o + 2, o + 3], 1),
                                 np.stack([o, o + 2, o + 1], 1), np.stack([o, o + 3, o + 2], 1)])
            V, UV, F = np.concatenate([V, corners]), np.concatenate([UV, fUV]), np.concatenate([F, Ff])
        scene.add("Vegetation", "M_GrassBlade", V, F, UV, sub="grass")

    # ---- small rocks: jittered, half-buried low-poly stones
    shapes = []
    for _ in range(8):  # a few stone shapes, instanced (building each one was slow)
        V, F, UV = ml.blob(1.0, rng, subdiv=0, jitter=0.3)
        V[:, 2] *= rng.uniform(0.45, 0.7)
        shapes.append((V, F, V[:, :2] / 0.8))
    for A in by_chunk(Rk):
        n = len(A)
        z = ground.height(A[:, 1], A[:, 2])
        r = np.where(rng.random(n) < 0.8, rng.uniform(0.05, 0.14, n), rng.uniform(0.12, 0.28, n))
        pos = np.column_stack([A[:, 1], A[:, 2], z - 0.3 * r])
        yaw = rng.uniform(0, 2 * np.pi, n)
        sh = rng.integers(0, len(shapes), n)
        for s_ in range(len(shapes)):
            m = sh == s_
            if m.any():
                scene.add("Terrain", "M_Rock", *ml.instances(shapes[s_], pos[m], yaw[m], np.repeat(r[m, None], 3, 1)),
                          sub="rocks")
    log(f"  ground details: {len(T)} grass tufts, {len(Rk)} small rocks "
        f"({(len(T) * tuft_tris + len(Rk) * rock_tris) / 1e6:.2f} M triangles)")


# ================================================================ buildings

DEFAULT_HEIGHT = {
    "house": 6.5, "detached": 6.5, "semidetached_house": 6.5, "terrace": 7.0, "residential": 7.0,
    "garage": 3.0, "garages": 3.0, "shed": 2.8, "hut": 2.8, "carport": 2.6, "roof": 5.0,
    "industrial": 9.0, "warehouse": 9.0, "retail": 7.0, "commercial": 9.0, "apartments": 15.0,
    "church": 14.0, "school": 9.0, "farm_auxiliary": 6.0, "barn": 7.0, "greenhouse": 3.5,
    "service": 3.5, "toll_booth": 3.0, "transformer_tower": 7.0,
}


def building_height(tags):
    h = _num(tags.get("height"))
    lv = _num(tags.get("building:levels"))
    rl = _num(tags.get("roof:levels"), 0) or 0
    if h is None and lv is not None:
        h = lv * 3.0 + rl * 2.0 + 1.0
    if h is None:
        h = DEFAULT_HEIGHT.get(tags.get("building", "yes"), 7.0)
    mh = _num(tags.get("min_height"))
    ml_ = _num(tags.get("building:min_level"))
    if mh is None and ml_ is not None:
        mh = ml_ * 3.0
    if tags.get("building") == "roof" and mh is None:
        mh = max(0.0, h - 0.6)
    return float(h), float(mh or 0.0)


CATEGORY = {
    "house": "house", "detached": "house", "semidetached_house": "house", "terrace": "house",
    "bungalow": "house", "villa": "house", "farm": "house", "apartments": "apartments",
    "residential": "house", "dormitory": "apartments", "commercial": "commercial", "retail": "commercial",
    "office": "commercial", "supermarket": "commercial", "hotel": "commercial", "school": "commercial",
    "university": "commercial", "hospital": "commercial", "train_station": "commercial",
    "transportation": "commercial", "public": "commercial", "civic": "commercial",
    "industrial": "industrial", "warehouse": "industrial", "hangar": "industrial", "factory": "industrial",
    "manufacture": "industrial", "service": "industrial", "barn": "farm", "farm_auxiliary": "farm",
    "stable": "farm", "cowshed": "farm", "greenhouse": "farm", "garage": "annex", "garages": "annex",
    "shed": "annex", "carport": "annex", "roof": "annex", "hut": "annex", "church": "religious",
    "chapel": "religious", "cathedral": "religious",
}
BD_USAGE = {"Résidentiel": "house", "Commercial et services": "commercial", "Industriel": "industrial",
            "Agricole": "farm", "Religieux": "religious", "Sportif": "commercial", "Annexe": "annex"}
STYLE_FACADES = {
    "house": ["render_cream", "render_white", "render_ochre", "stone", "brick_red", "render_cream"],
    "apartments": ["render_white", "render_grey", "concrete", "brick_brown", "render_cream"],
    "commercial": ["glass", "concrete", "render_grey", "metal_grey"],
    "industrial": ["metal_grey", "metal_blue", "metal_beige", "concrete"],
    "farm": ["stone", "metal_beige", "brick_brown"],
    "annex": ["render_grey", "concrete", "metal_grey", "render_cream"],
    "religious": ["stone"],
}


def _stable(key):
    import zlib
    return zlib.crc32(key.encode())


def _pick(seq, key):
    return seq[_stable(key) % len(seq)]


def _building_style(tags, height, key, bd_usage=""):
    btype = tags.get("building", "yes")
    cat = CATEGORY.get(btype)
    if cat is None:
        cat = BD_USAGE.get(bd_usage, "house")
    if cat == "house" and height > 11:
        cat = "apartments"
    facade = f"M_Facade_{_pick(STYLE_FACADES[cat], key)}"
    shape = tags.get("roof:shape", "")
    if not shape:
        if cat in ("house", "farm", "religious"):
            shape = _pick(["gabled", "gabled", "hipped"], key + "r")
        elif cat == "annex" and height < 4.5:
            shape = _pick(["gabled", "flat"], key + "r")
        else:
            shape = "flat"
    if shape in ("gabled", "hipped", "pyramidal", "half-hipped", "gambrel", "mansard"):
        shape = "hipped" if shape in ("hipped", "pyramidal", "half-hipped", "mansard") else "gabled"
        roof = "M_RoofMetal" if cat in ("industrial", "farm") and _stable(key) % 2 else \
            _pick(["M_RoofTiles", "M_RoofTiles", "M_RoofSlate"], key + "m")
    else:
        shape = "flat"
        roof = "M_RoofFlat"
    return cat, facade, shape, roof


def _pitched_roof(poly, z_eave, shape, overhang=0.3):
    """Gabled / hipped roof over the footprint's minimum rectangle.

    Returns (roof mesh, gable-end mesh) or None when the footprint is not
    close enough to a rectangle."""
    rect = poly.minimum_rotated_rectangle
    if rect.area <= 0 or poly.area / rect.area < 0.75:
        return None
    R = np.array(rect.exterior.coords)[:4]
    e1, e2 = R[1] - R[0], R[2] - R[1]
    if np.hypot(*e1) < np.hypot(*e2):
        R = np.roll(R, -1, axis=0)
        e1, e2 = R[1] - R[0], R[2] - R[1]
    L, W = np.hypot(*e1), np.hypot(*e2)
    if W < 2.5:
        return None
    u, v = e1 / L, e2 / W
    c = R.mean(axis=0)
    L += 2 * overhang
    W += 2 * overhang
    h = min(0.7 * W / 2, 6.0)  # about 35 degrees, capped for big buildings

    def P(a, b, z):
        return np.array([*(c + u * a + v * b), z])

    A, B = P(-L / 2, -W / 2, z_eave), P(L / 2, -W / 2, z_eave)
    C, D = P(L / 2, W / 2, z_eave), P(-L / 2, W / 2, z_eave)
    inset = 0.0 if shape == "gabled" else min(W / 2, L / 2 - 0.01)
    R1, R2 = P(-L / 2 + inset, 0, z_eave + h), P(L / 2 - inset, 0, z_eave + h)
    roof_q = [(A, B, R2, R1), (C, D, R1, R2)]
    ends = [(B, C, R2), (D, A, R1)]
    center = np.array([*c, z_eave])

    def tri_mesh(tris, outward_from):
        V, F, UV = [], [], []
        for t in tris:
            t = [np.asarray(p, float) for p in t]
            n = np.cross(t[1] - t[0], t[2] - t[0])
            out = np.mean(t, axis=0) - outward_from
            if n @ out < 0:
                t = t[::-1]
            k = len(V)
            V += t
            F.append([k, k + 1, k + 2])
            for p in t:
                UV.append([(p[:2] - c) @ u / 4.0, ((p[:2] - c) @ v) / 4.0 + p[2] / 4.0])
        return np.array(V), np.array(F), np.array(UV)

    quads = []
    for q in roof_q:
        quads += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    roof = tri_mesh(quads, center - [0, 0, 5.0])
    endm = tri_mesh(ends, center)
    if shape == "gabled":
        return roof, endm, None
    return ml.merge(roof, endm), None, None


def build_buildings(scene, osm, ground, corridor, cfg, log, landcover=None):
    polys = []
    count = pitched = 0
    items = []
    for wid, tags, poly in osm.areas(lambda t: "building" in t and t.get("building") != "no"):
        if poly.intersects(corridor) and poly.area >= 4:
            items.append((str(wid), tags, poly, None, ""))
    bd = landcover.bd_buildings if landcover is not None else []
    if bd:
        tree = shapely.STRtree([p for p, _, _ in bd])
        matched = set()
        new_items = []
        for key, tags, poly, _, _ in items:
            hits = tree.query(poly, predicate="intersects")
            best, best_a = None, 0.0
            for k in hits:
                a = bd[k][0].intersection(poly).area
                if a > best_a:
                    best, best_a = k, a
            if best is not None and best_a > 0.3 * poly.area:
                matched.add(best)
                new_items.append((key, tags, poly, bd[best][1], bd[best][2]))
            else:
                new_items.append((key, tags, poly, None, ""))
        added = 0
        osm_union = unary_union([p for _, _, p, _, _ in items]) if items else None
        for k, (p, h, usage) in enumerate(bd):
            if k in matched or p.area < 12 or not p.intersects(corridor):
                continue
            if osm_union is not None and p.intersection(osm_union).area > 0.3 * p.area:
                continue
            new_items.append((f"bd{k}", {"building": "yes"}, p, h, usage))
            added += 1
        items = new_items
        if added:
            log(f"  {added} buildings added from BD TOPO (missing in OSM)")
    for key, tags, poly, bd_h, bd_usage in items:
        h, mh = building_height(tags)
        if bd_h and not tags.get("height") and not tags.get("building:levels"):
            h = float(bd_h)
        for pi_, part in enumerate(getattr(poly, "geoms", [poly])):
            if part.geom_type != "Polygon":
                continue
            item = "Building_" + re.sub(r"[^A-Za-z0-9]", "_", str(key)) + (f"_{pi_}" if pi_ else "")
            ring = np.array(part.exterior.coords)
            zg = ground.height(ring[:, 0], ring[:, 1])
            base = float(zg.min())
            cat, facade, shape, roof_mat = _building_style(tags, h, key, bd_usage)
            part = part.simplify(0.05)
            if part.is_empty or part.geom_type != "Polygon":
                continue
            z0 = base - 0.5 if mh <= 0 else base + mh
            roof = None
            if shape != "flat" and mh <= 0:
                # height is to the eaves for BD TOPO, to the top for OSM
                eave = base + (h if bd_h and not tags.get("height") else max(2.5, h * 0.75))
                roof = _pitched_roof(part, eave, shape)
            if roof is not None:
                z1 = eave
            else:
                z1 = base + h
            if z1 - z0 < 0.3:
                continue
            V, F, UV = ml.extrude_polygon(part, z0, z1, roof=False)
            scene.add("Building", facade, V, F, UV, item=item)
            if roof is not None:
                roof_mesh, ends, _ = roof
                scene.add("Building", roof_mat, *roof_mesh, item=item)
                if ends is not None:
                    scene.add("Building", facade, *ends, item=item)
                pitched += 1
            else:
                scene.add("Building", roof_mat, *ml.polygon_cap(part, z1), item=item)
                if cat in ("apartments", "commercial") and part.area > 60:
                    # low parapet around flat roofs
                    ringp = np.array(part.exterior.coords)
                    scene.add("Building", facade, *ml.wall_along(ringp, np.full(len(ringp), z1), 0.8, 0.25), item=item)
            if mh > 0:
                Vb, Fb, UVb = ml.polygon_cap(part, z0)
                scene.add("Building", roof_mat, Vb, Fb[:, ::-1], UVb, item=item)
            scene.record("building", polygon=[list(c) for c in part.exterior.coords], z0=z0, z1=z1)
            polys.append(part)
            count += 1
    log(f"  {count} buildings ({pitched} with pitched roofs)")
    return unary_union(polys) if polys else None


def _jitter_points(poly, spacing, rng):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx, spacing)
    ys = np.arange(miny, maxy, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    X, Y = np.meshgrid(xs, ys)
    X = X.ravel() + rng.uniform(-0.45, 0.45, X.size) * spacing
    Y = Y.ravel() + rng.uniform(-0.45, 0.45, Y.size) * spacing
    m = shapely.contains_xy(poly, X, Y)
    return np.column_stack([X[m], Y[m]])


def tree_template(kind, rng):
    """Unit-height tree: list of (mesh, material). kind 0 broadleaf, 1 conifer, 2 bush."""
    parts = []
    if kind == 0:
        trunk_top = rng.uniform(0.28, 0.38)
        parts.append((ml.cylinder_between((0, 0, -0.05), (0, 0, trunk_top), 0.035, 0.022, seg=6), "M_Bark"))
        crown_c = np.array([0.0, 0.0, trunk_top + 0.30])
        blobs = []
        for i in range(3):  # main branches
            a = rng.uniform(0, 2 * np.pi)
            tip = crown_c + np.array([np.cos(a) * 0.16, np.sin(a) * 0.16, rng.uniform(-0.10, 0.10)])
            start = (0, 0, trunk_top * rng.uniform(0.75, 1.0))
            parts.append((ml.cylinder_between(start, tip, 0.016, 0.007, seg=4), "M_Bark"))
            blobs.append(tip)
        for i in range(rng.integers(3, 5)):  # leaf clusters inside an ellipsoid crown
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            blobs.append(crown_c + d * np.array([0.16, 0.16, 0.20]) * rng.uniform(0.3, 1.0))
        for c in blobs:
            V, F, UV = ml.blob(rng.uniform(0.15, 0.21), rng, subdiv=0, jitter=0.22)
            parts.append(((V * [1, 1, 1.05] + c, F, UV * 3), "M_Leaves"))
    elif kind == 1:
        parts.append((ml.cylinder_between((0, 0, -0.05), (0, 0, 0.95), 0.028, 0.006, seg=5), "M_Bark"))
        tiers = rng.integers(4, 6)
        for i in range(tiers):
            t = i / tiers
            z0 = 0.18 + t * 0.72
            r = 0.26 * (1 - t) + 0.05
            V, F, UV = ml.cylinder(r, 0.0, 0.30 * (1 - 0.4 * t), seg=7, cap=False)
            V[:7, 2] -= rng.uniform(0.0, 0.04, 7)  # ragged tier edge
            parts.append(((ml.transform(V, rng.uniform(0, 1), (0, 0, z0)), F, UV * 3), "M_Conifer"))
    else:
        for i in range(2):
            c = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0.35])
            V, F, UV = ml.blob(rng.uniform(0.35, 0.5), rng, subdiv=0, jitter=0.25)
            parts.append(((V * [1, 1, 0.75] + c, F, UV * 2), "M_Bush"))
    # merge parts sharing a material
    out = {}
    for mesh, mat in parts:
        out.setdefault(mat, []).append(mesh)
    return [(ml.merge(*meshes), mat) for mat, meshes in out.items()]


def _lumpy(radius, rng, subdiv=1):
    """Leaf clump: an icosphere pushed in and out by smooth noise, so the crown
    has hollows and bulges instead of facets."""
    V, F, UV = ml.icosphere(1.0, subdiv)
    k = rng.normal(size=(3, 3))
    bump = sum(np.sin(V @ k[i] * (2.5 + i) + rng.uniform(0, 6)) for i in range(3)) / 3.0
    V = V * (1 + 0.28 * bump[:, None] + rng.uniform(-0.06, 0.06, (len(V), 1))) * radius
    UV = np.column_stack([np.arctan2(V[:, 1], V[:, 0]) / np.pi, V[:, 2] / max(radius, 1e-6)])
    return V, F, UV


def tree_template_hq(kind, rng):
    """Detailed unit-height tree for the trees near the road (about 1,300
    triangles for a broadleaf, 500 for a conifer, 400 for a bush)."""
    parts = []
    leaf = "M_Leaves" if rng.random() < 0.6 else "M_LeavesDark"
    if kind == 0:
        trunk_top = rng.uniform(0.30, 0.42)
        lean = rng.normal(0, 0.02, 2)
        top = np.array([lean[0], lean[1], trunk_top])
        parts.append((ml.cylinder_between((0, 0, -0.05), (0, 0, 0.04), 0.06, 0.036, seg=9), "M_Bark"))  # root flare
        parts.append((ml.cylinder_between((0, 0, 0.03), top, 0.036, 0.024, seg=9), "M_Bark"))
        crown_c = top + np.array([0, 0, 0.28])
        tips = []
        nb = int(rng.integers(5, 8))
        for i in range(nb):
            a = 2 * np.pi * i / nb + rng.uniform(-0.3, 0.3)
            h0 = trunk_top * rng.uniform(0.7, 1.0)
            start = np.array([lean[0] * h0 / trunk_top, lean[1] * h0 / trunk_top, h0])
            mid = start + np.array([np.cos(a) * 0.09, np.sin(a) * 0.09, rng.uniform(0.10, 0.18)])
            tip = mid + np.array([np.cos(a) * 0.08, np.sin(a) * 0.08, rng.uniform(0.02, 0.12)])
            parts.append((ml.cylinder_between(start, mid, 0.017, 0.011, seg=6), "M_Bark"))
            parts.append((ml.cylinder_between(mid, tip, 0.011, 0.005, seg=5), "M_Bark"))
            for _ in range(1):  # twig
                b = a + rng.uniform(-0.9, 0.9)
                tw = mid + np.array([np.cos(b) * 0.07, np.sin(b) * 0.07, rng.uniform(0.04, 0.12)])
                parts.append((ml.cylinder_between(mid, tw, 0.006, 0.003, seg=4), "M_Bark"))
                tips.append(tw)
            tips.append(tip)
        for i in range(int(rng.integers(3, 6))):  # fill the crown
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            tips.append(crown_c + d * np.array([0.14, 0.14, 0.16]) * rng.uniform(0.2, 0.9))
        for c in tips[:11]:
            V, F, UV = _lumpy(rng.uniform(0.11, 0.16), rng)
            parts.append(((V * [1, 1, 0.85] + c, F, UV * 3), leaf))
    elif kind == 1:
        parts.append((ml.cylinder_between((0, 0, -0.05), (0, 0, 0.97), 0.03, 0.005, seg=7), "M_Bark"))
        tiers = int(rng.integers(9, 13))
        for i in range(tiers):
            t_ = i / tiers
            z0 = 0.14 + t_ * 0.80
            r = 0.27 * (1 - t_) ** 0.9 + 0.035
            h = 0.16 * (1 - 0.5 * t_) + 0.04
            seg = 12
            V, F, UV = ml.cylinder(r, 0.0, h, seg=seg, cap=False)
            ring = np.arange(len(V)) < seg
            jag = np.where(np.arange(len(V)) % 2 == 0, 1.0, 0.72)  # star-shaped, drooping edge
            V[ring, :2] *= jag[ring, None] * rng.uniform(0.9, 1.1, (ring.sum(), 1))
            V[ring, 2] -= rng.uniform(0.02, 0.06, ring.sum())
            parts.append(((ml.transform(V, rng.uniform(0, 2 * np.pi), (0, 0, z0)), F, UV * 3), "M_Conifer"))
    else:
        for i in range(5):
            c = np.array([rng.uniform(-0.28, 0.28), rng.uniform(-0.28, 0.28), rng.uniform(0.28, 0.45)])
            V, F, UV = _lumpy(rng.uniform(0.22, 0.34), rng)
            parts.append(((V * [1, 1, 0.8] + c, F, UV * 2), "M_Bush"))
    out = {}
    for mesh, mat in parts:
        out.setdefault(mat, []).append(mesh)
    return [(ml.merge(*meshes), mat) for mat, meshes in out.items()]


TEMPLATE_WIDTH = {0: 0.7, 1: 0.6, 2: 1.2}  # crown width of a unit-height template


def build_vegetation(scene, osm, ground, corridor, blocked, cfg, rng, log, landcover=None, tiles=None):
    scene.material("M_Bark", (0.33, 0.24, 0.16))
    scene.material("M_Leaves", (0.22, 0.40, 0.14))
    if "M_LeavesDark" not in scene.materials:
        scene.material("M_LeavesDark", (0.14, 0.30, 0.10))
    scene.material("M_Conifer", (0.10, 0.28, 0.14))
    scene.material("M_Bush", (0.28, 0.42, 0.18))
    spacing = float(cfg.get("tree_spacing", 7.0))
    groups = {0: [], 1: [], 2: []}  # broadleaf, conifer, bush; rows are x, y, height, width
    lidar = None
    if landcover is not None and tiles:
        lidar = landcover.detect_trees(tiles, ground, blocked, rng, log)
    if lidar is not None:
        T, kinds = lidar
        if len(T):
            groups[0].append(T[kinds != "conifer"])
            groups[1].append(T[kinds == "conifer"])
        for hp in landcover.bd_hedges:  # hedges: dense bushes
            area = hp.intersection(corridor)
            if not area.is_empty:
                groups[2].append(_jitter_points(area, 2.2, rng))

    for _, tags, poly in (osm.areas(lambda t: t.get("landuse") in ("forest", "orchard")
                                   or t.get("natural") in ("wood", "scrub")) if lidar is None else
                           osm.areas(lambda t: t.get("natural") == "scrub")):
        area = poly.intersection(corridor)
        if area.is_empty:
            continue
        if tags.get("natural") == "scrub":
            groups[2].append(_jitter_points(area, spacing * 0.7, rng))
            continue
        sp = spacing * (0.9 if tags.get("landuse") == "orchard" else 1.0)
        pts = _jitter_points(area, sp, rng)
        lt = tags.get("leaf_type", "")
        if lt == "needleleaved":
            groups[1].append(pts)
        elif lt == "mixed":
            m = rng.random(len(pts)) < 0.5
            groups[1].append(pts[m])
            groups[0].append(pts[~m])
        else:
            groups[0].append(pts)
    for wid, w in (osm.ways.items() if lidar is None else ()):
        if w["tags"].get("natural") == "tree_row":
            P = osm.way_xy(wid)
            if P is not None and cumlen(P)[-1] > 1:
                Q, _ = resample(P, 8.0)
                groups[0].append(Q[:, :2])
    ids = [nid for nid, v in osm.nodes.items() if v[2].get("natural") == "tree" and nid in osm.node_xy]
    if ids and lidar is None:
        groups[0].append(np.array([osm.node_xy[i] for i in ids]))

    cap = int(cfg.get("max_trees", 20000))
    def as4(a):
        a = np.asarray(a, float)
        if a.ndim == 2 and a.shape[1] == 2:
            a = np.column_stack([a, np.full((len(a), 2), np.nan)])
        return a.reshape(-1, 4)
    allpts = {k: (np.concatenate([as4(a) for a in v]) if v else np.zeros((0, 4))) for k, v in groups.items()}
    # variants 0-3 light (far trees), 4-7 detailed (within hq_tree_m of a road)
    variants = {k: [tree_template(k, rng) for _ in range(4)] + [tree_template_hq(k, rng) for _ in range(4)]
                for k in (0, 1, 2)}
    hq_m = float(cfg.get("hq_tree_m", 60.0))
    # the budget counts light models; detailed ones have their own budget (hq_tri_budget)
    tris_per = {k: np.mean([sum(len(m[1]) for m, _ in v) for v in variants[k][:4]]) for k in variants}
    tris_hq = {k: np.mean([sum(len(m[1]) for m, _ in v) for v in variants[k][4:]]) for k in variants}
    hq_left = [float(cfg.get("hq_tri_budget", 3e6))]
    n_hq = [0]
    total = sum(len(v) for v in allpts.values())
    est_tris = sum(len(allpts[k]) * tris_per[k] for k in allpts)
    budget = float(cfg.get("veg_tri_budget", 2.5e6))
    keep_frac = 1.0
    if total:
        keep_frac = min(1.0, cap / total, budget / max(est_tris, 1.0))
    if keep_frac < 1.0:
        log(f"  {total} tree spots, keeping {keep_frac * 100:.0f}% (max_trees {cap}, "
            f"vegetation budget {budget / 1e6:.1f} M triangles)")
    n_trees = 0
    for kind, pts in allpts.items():
        if len(pts) == 0:
            continue
        m = shapely.contains_xy(corridor, pts[:, 0], pts[:, 1])
        m &= ~ground.near_road(pts[:, 0], pts[:, 1], 3.0 if kind != 2 else 2.0)
        if blocked is not None:
            m &= ~shapely.contains_xy(blocked, pts[:, 0], pts[:, 1])
        if keep_frac < 1:
            m &= rng.random(len(pts)) < keep_frac
        pts = pts[m]
        if len(pts) == 0:
            continue
        z = ground.height(pts[:, 0], pts[:, 1]) - 0.15
        k = len(pts)
        n_trees += k
        yaw = rng.uniform(0, 2 * np.pi, k)
        if kind == 2:
            Hr = rng.uniform(1.0, 2.2, k)
        elif kind == 0:
            Hr = rng.uniform(9, 18, k)
        else:
            Hr = rng.uniform(12, 24, k)
        # measured height / crown width (LIDAR) when known, random otherwise
        H = np.where(np.isfinite(pts[:, 2]), pts[:, 2], Hr)
        W = np.where(np.isfinite(pts[:, 3]), pts[:, 3], H * TEMPLATE_WIDTH[kind] * rng.uniform(0.8, 1.2, k))
        if kind == 2:
            W = np.maximum(W, H * 0.8)
        # keep low crowns and bushes off the asphalt (they would be solid in CARLA);
        # tall trees may overhang the road above the vehicles
        if ground.tree is not None:
            dd, _ = ground.tree.query(pts[:, :2])
            need = W / 2 + 0.3 if kind == 2 else np.where(H < 8, W / 2, 1.5)
            okd = dd >= need
            pts, z, H, W, yaw = pts[okd], z[okd], H[okd], W[okd], yaw[okd]
            k = len(pts)
            n_trees -= int((~okd).sum())
            if k == 0:
                continue
        var = rng.integers(0, 4, k)
        if hq_m > 0 and ground.tree is not None:
            # detailed models for the trees nearest a road, while the budget lasts
            dr, _ = ground.tree.query(pts[:, :2], distance_upper_bound=hq_m)
            cand = np.nonzero(np.isfinite(dr))[0]
            cand = cand[np.argsort(dr[cand])][: int(hq_left[0] // tris_hq[kind])]
            var[cand] += 4
            hq_left[0] -= len(cand) * tris_hq[kind]
            n_hq[0] += len(cand)
        sxy = W / TEMPLATE_WIDTH[kind]
        sc = np.column_stack([sxy, sxy, H])
        pos = np.column_stack([pts[:, :2], z])
        for i in range(k):
            scene.record("tree", type=("broadleaf", "conifer", "bush")[kind], pos=pos[i].tolist(),
                         height=float(H[i]), width=float(W[i]), yaw=float(yaw[i]))
        key = np.floor(pts[:, 0] / scene.chunk).astype(int) * 100000 + np.floor(pts[:, 1] / scene.chunk).astype(int)
        for kk in np.unique(key):
            for v in range(8):
                sel = (key == kk) & (var == v)
                if not sel.any():
                    continue
                for tmpl, mat in variants[kind][v]:
                    scene.add("Vegetation", mat, *ml.instances(tmpl, pos[sel], yaw[sel], sc[sel]))
    log(f"  {n_trees} trees/bushes (cap {cap}), {n_hq[0]} of them with the detailed model near the road")


# ================================================================ barriers

def _polyline_pieces(P, per=40):
    n = len(P)
    a = 0
    while a < n - 1:
        b = min(n, a + per + 1)
        yield P[a:b]
        a = b - 1


def guardrail_mesh(P, z, post_step=4.0):
    """W-beam + posts along P (n,2) with ground z (n,)."""
    beam = ml.wall_along(P, z + 0.45, 0.32, 0.08)
    s = cumlen(P)
    if s[-1] < 0.5:
        return beam, ml.merge()
    sp = np.arange(0, s[-1] + 1e-6, post_step)
    px = np.interp(sp, s, P[:, 0])
    py = np.interp(sp, s, P[:, 1])
    pz = np.interp(sp, s, z)
    T = np.gradient(P, axis=0) if len(P) > 1 else np.array([[1.0, 0.0]])
    hdg = np.arctan2(np.interp(sp, s, T[:, 1]), np.interp(sp, s, T[:, 0]))
    posts = ml.instances(ml.box(0.12, 0.1, 0.8), np.column_stack([px, py, pz - 0.1]), hdg,
                         np.ones((len(sp), 3)))
    return beam, posts


BARRIERS = {
    # barrier value: (category, material, height, thickness)
    "guard_rail": ("GuardRail", "M_Metal", 0.8, 0.1),
    "jersey_barrier": ("GuardRail", "M_Concrete", 0.8, 0.5),
    "cable_barrier": ("GuardRail", "M_Metal", 0.75, 0.05),
    "wall": ("Wall", "M_WallStone", 2.0, 0.3),
    "retaining_wall": ("Wall", "M_Concrete", 2.0, 0.4),
    "city_wall": ("Wall", "M_WallStone", 4.0, 0.8),
    "noise_barrier": ("Wall", "M_NoiseWall", 4.0, 0.25),
    "fence": ("Fence", "M_Fence", 1.6, 0.05),
    "handrail": ("Fence", "M_Metal", 1.0, 0.05),
    "hedge": ("Vegetation", "M_Bush", 1.6, 0.9),
}


def build_barriers(scene, osm, ground, corridor, cfg, log):
    scene.material("M_Metal", (0.62, 0.64, 0.66))
    scene.material("M_Concrete", (0.66, 0.65, 0.62))
    scene.material("M_WallStone", (0.62, 0.58, 0.52))
    scene.material("M_NoiseWall", (0.45, 0.52, 0.48))
    scene.material("M_Fence", (0.35, 0.38, 0.35))
    scene.material("M_Bush", (0.28, 0.42, 0.18))
    samples = []
    count = 0
    for wid, w in osm.ways.items():
        t = w["tags"]
        b = t.get("barrier")
        if b is None and t.get("wall") == "noise_barrier":
            b = "noise_barrier"
        if b == "wall" and (t.get("wall") == "noise_barrier" or t.get("noise_barrier") == "yes"):
            b = "noise_barrier"
        if b not in BARRIERS:
            continue
        P = osm.way_xy(wid)
        if P is None or cumlen(P)[-1] < 1:
            continue
        Q, _ = resample(P, 2.0)
        m = shapely.contains_xy(corridor.buffer(20), Q[:, 0], Q[:, 1])
        cat, mat, h, th = BARRIERS[b]
        h = _num(t.get("height"), h)
        for a, e in runs(m):
            if e - a < 2:
                continue
            seg = Q[a:e]
            z = ground.surface(seg[:, 0], seg[:, 1])
            samples.append(seg)
            if b in ("guard_rail", "jersey_barrier", "cable_barrier", "fence", "handrail"):
                road_left = True
                if ground.tree is not None:
                    mid = len(seg) // 2
                    _, j = ground.tree.query(seg[mid])
                    t = seg[min(mid + 1, len(seg) - 1)] - seg[max(mid - 1, 0)]
                    d = ground.fp[j, :2] - seg[mid]
                    road_left = (t[0] * d[1] - t[1] * d[0]) > 0
                scene.record("barrier", type=b, points=np.column_stack([seg, z]).tolist(), road_left=bool(road_left))
            for piece_idx in range(0, len(seg) - 1, 40):
                sl = slice(piece_idx, min(len(seg), piece_idx + 41))
                p, zz = seg[sl], z[sl]
                if len(p) < 2:
                    continue
                if b == "guard_rail":
                    beam, posts = guardrail_mesh(p, zz)
                    scene.add(cat, mat, *beam)
                    scene.add(cat, mat, *posts)
                else:
                    scene.add(cat, mat, *ml.wall_along(p, zz - 0.3, h + 0.3, th),
                              sub="hedge" if b == "hedge" else None)
            count += 1
    log(f"  {count} OSM barriers")
    return np.concatenate(samples) if samples else np.zeros((0, 2))


def infer_guardrails(scene, models, ground, osm_barrier_pts, cfg, log):
    """Guard rails along motorway/trunk edges where OSM has none mapped."""
    scene.material("M_Metal", (0.62, 0.64, 0.66))
    btree = cKDTree(osm_barrier_pts) if len(osm_barrier_pts) else None
    total = 0.0
    for mi, m in enumerate(models):
        if m.kind not in ("route", "road"):
            continue
        cls = np.array([h in ("motorway", "trunk") for h in m.hclass])
        if not cls.any():
            continue
        L, R = m.surface_edges()
        sides = [(R - 0.45, -1)]
        if m.oneway:
            sides.append((L + 0.45, 1))
        for off, side in sides:
            P = m.point(off)
            ok = cls & ~m.bridge & ~m.tunnel
            # skip where another road touches this edge (ramps, crossings)
            if ground.tree is not None:
                nb = ground.tree.query_ball_point(P[:, :2], 4.0)
                for i, lst in enumerate(nb):
                    if not ok[i] or not lst:
                        continue
                    lst = np.asarray(lst)
                    other = ground.fp_mid[lst] != mi
                    if not other.any():
                        continue
                    lst = lst[other]
                    dz = np.abs(ground.fp[lst, 2] - m.z[i])
                    dots = ground.fp_dir[lst] @ m.T[i]
                    # on the median side the opposite carriageway (anti-parallel) is fine
                    touch = (dz < 3) if side < 0 else (dz < 3) & (dots > -0.5)
                    if touch.any():
                        ok[i] = False
            if btree is not None:
                d, _ = btree.query(P[:, :2], distance_upper_bound=3.0)
                ok &= ~np.isfinite(d)
            for a, b in runs(ok):
                if b - a < 10:
                    continue
                seg = P[a:b]
                scene.record("barrier", type="guard_rail", points=seg.tolist(), road_left=bool(side < 0))
                for k in range(0, len(seg) - 1, 40):
                    p = seg[k:k + 41]
                    if len(p) < 2:
                        continue
                    beam, posts = guardrail_mesh(p[:, :2], p[:, 2])
                    scene.add("GuardRail", "M_Metal", *beam)
                    scene.add("GuardRail", "M_Metal", *posts)
                total += (b - a) * STEP
    log(f"  inferred guard rails: {total / 1000:.1f} km")


# ================================================================ panels and poles

def _pole(scene, x, y, z, h, r=0.05, mat="M_Metal", cat="Pole"):
    V, F, UV = ml.cylinder(r, r * 0.8, h + 0.3, seg=6)
    scene.add(cat, mat, ml.transform(V, 0, (x, y, z - 0.3)), F, UV)


def _panel(scene, pf, x, y, zc, yaw, kind, value=None, size=0.8, lines=None, colour=None):
    from .writers.rrhd import rr_sign_asset
    mat, shape, aspect = pf.material(kind, value, lines, colour)
    w = size * (aspect if shape == "rect" else 1.0)
    h = size if shape == "rect" else size * (0.9 if shape.startswith("triangle") else 1.0)
    front, back = ml.panel(shape, w, h)
    rr_asset = None if lines else rr_sign_asset(kind, value)
    for (V, F, UV), m in ((front, mat), (back, "M_PanelBack")):
        # panels RoadRunner has in its library get the sub-name "rr" so the
        # RoadRunner export uses the library sign instead of this mesh
        scene.add("Panel", m, ml.transform(V, yaw, (x, y, zc)), F, UV, sub="rr" if rr_asset else None)
    if rr_asset:
        scene.record("sign", rr_asset=rr_asset, center=[x, y, zc], size=float(max(w, h)),
                     facing=[math.sin(yaw), -math.cos(yaw)])
    return w, h


def _travel_dir(osm, nid, tags, ridx, x, y, way_dir_of_node):
    """Unit travel direction of the traffic a panel at (x, y) is meant for."""
    d = str(tags.get("direction", tags.get("traffic_sign:direction", ""))).lower()
    try:
        deg = float(d)
        th = math.radians(deg)
        return np.array([math.sin(th), math.cos(th)]), None
    except ValueError:
        pass
    if nid in way_dir_of_node:
        t = way_dir_of_node[nid]
        return (-t if d == "backward" else t), True
    nr = ridx.nearest(x, y)
    if nr is None:
        return None, None
    m, i = nr
    T = m.T[i]
    lat = (x - m.x[i]) * m.N[i, 0] + (y - m.y[i]) * m.N[i, 1]
    if d == "backward":
        return -T, False
    if d == "forward" or m.oneway or lat < m.c0[i]:
        return T, False
    return -T, False


def build_point_objects(scene, osm, ground, ridx, corridor, tex_dir, cfg, log):
    pf = PanelFactory(scene, tex_dir)
    scene.material("M_Metal", (0.62, 0.64, 0.66))
    scene.material("M_Wood", (0.40, 0.30, 0.20))
    scene.material("M_Dark", (0.12, 0.12, 0.13))
    scene.material("M_Orange", (0.95, 0.45, 0.05))
    scene.material("M_Concrete", (0.66, 0.65, 0.62))
    scene.material("M_WallDefault", (0.82, 0.78, 0.72))
    scene.material("M_Roof", (0.45, 0.40, 0.38))
    left_hand = cfg.get("drive_left", False)

    # direction of the way at every node of a motor road (for on-road panel nodes)
    way_dir_of_node = {}
    for wid, w in osm.ways.items():
        h = w["tags"].get("highway")
        if not h or h in ("footway", "path", "cycleway", "steps", "pedestrian", "track"):
            continue
        nodes = w["nodes"]
        P = [osm.node_xy.get(n) for n in nodes]
        for k, n in enumerate(nodes):
            if P[k] is None:
                continue
            a = P[max(0, k - 1)] or P[k]
            b = P[min(len(nodes) - 1, k + 1)] or P[k]
            v = np.array(b) - np.array(a)
            ln = np.hypot(*v)
            if ln > 0:
                way_dir_of_node[n] = v / ln

    counts = {}
    for nid, (lon, lat, tags) in osm.nodes.items():
        if not tags or nid not in osm.node_xy:
            continue
        x, y = osm.node_xy[nid]
        if not corridor.contains(Point(x, y)):
            continue
        hw = tags.get("highway")
        ts = tags.get("traffic_sign")
        kind = None
        if ts or hw in ("stop", "give_way"):
            kind = "panel"
        elif hw == "street_lamp":
            kind = "lamp"
        elif hw == "traffic_signals":
            kind = "tsignal"
        elif hw in ("milestone",):
            kind = "milestone"
        elif hw == "speed_camera":
            kind = "camera"
        elif tags.get("amenity") == "emergency_phone" or hw == "emergency_access_point":
            kind = "phone"
        elif tags.get("power") in ("pole", "tower", "portal"):
            kind = "power_" + tags["power"]
        elif tags.get("man_made") in ("street_cabinet", "mast", "flagpole", "surveillance"):
            if tags.get("surveillance") == "indoor" or tags.get("camera:mount") in ("wall", "ceiling", "building"):
                continue  # cameras fixed to a building, not a pole
            kind = tags["man_made"]
        elif tags.get("advertising"):
            kind = "billboard"
        elif tags.get("barrier") in ("bollard", "block", "lift_gate", "gate", "toll_booth"):
            kind = "barrier_" + tags["barrier"]
        if kind is None:
            continue
        # street furniture mapped on a road centreline (milestones, cameras,
        # phones, cabinets, lamps...) goes to the roadside, never on a lane
        if kind not in ("panel", "tsignal") and not kind.startswith("barrier_") \
                and ground.on_asphalt(np.array([x]), np.array([y]), 1.0)[0]:
            nr = ridx.nearest(x, y)
            if nr is not None:
                m, i = nr
                L, R = m.surface_edges()
                side = R[i] - 1.2 if not left_hand else L[i] + 1.2
                x = m.x[i] + m.N[i, 0] * side
                y = m.y[i] + m.N[i, 1] * side
        z = float(ground.surface(np.array([x]), np.array([y]))[0])
        counts[kind] = counts.get(kind, 0) + 1

        if kind == "panel":
            t, on_way = _travel_dir(osm, nid, tags, ridx, x, y, way_dir_of_node)
            if t is None:
                t = np.array([0.0, 1.0])
            if on_way:
                nr = ridx.nearest(x, y)
                if nr is not None:
                    m, i = nr
                    L, R = m.surface_edges()
                    along = float(t @ m.T[i]) > 0
                    right_off = (R[i] - 1.2) if along != left_hand else (L[i] + 1.2)
                    x = m.x[i] + m.N[i, 0] * right_off
                    y = m.y[i] + m.N[i, 1] * right_off
                    z = float(ground.surface(np.array([x]), np.array([y]))[0])
            yaw = _yaw_facing(-t[0], -t[1])
            nr = ridx.nearest(x, y)
            big = nr is not None and nr[0].rank >= 8
            size = 1.0 if big else 0.7
            values = []
            if ts:
                for v in str(ts).replace(",", ";").split(";"):
                    if v.strip() and v.strip().lower() not in ("none",):
                        values.append(v.strip())
            elif hw == "stop":
                values = ["stop"]
            else:
                values = ["give_way"]
            zc = z + 2.3 + size / 2
            top = zc
            for v in values[:3]:
                k, val = classify(v, tags)
                _, hh = _panel(scene, pf, x, y, zc, yaw, k, val, size=size)
                top = zc + hh / 2
                zc -= hh + 0.15
            _pole(scene, x, y, z, top - z, r=0.04)
        elif kind == "lamp":
            nr = ridx.nearest(x, y)
            H = 10.0 if nr is not None and nr[0].rank >= 7 else 7.5
            yaw = 0.0
            if nr is not None:
                m, i = nr
                yaw = math.atan2(m.y[i] - y, m.x[i] - x)
            _pole(scene, x, y, z, H, r=0.08)
            V, F, UV = ml.box(1.8, 0.08, 0.08)
            scene.add("Pole", "M_Metal", ml.transform(V + [0.9, 0, 0], yaw, (x, y, z + H - 0.1)), F, UV)
            V, F, UV = ml.box(0.6, 0.3, 0.15)
            scene.add("Pole", "M_Dark", ml.transform(V + [1.7, 0, 0], yaw, (x, y, z + H - 0.2)), F, UV)
        elif kind == "tsignal":
            t, _ = _travel_dir(osm, nid, tags, ridx, x, y, way_dir_of_node)
            yaw = _yaw_facing(-t[0], -t[1]) if t is not None else 0.0
            _pole(scene, x, y, z, 3.4, r=0.07)
            V, F, UV = ml.box(0.35, 0.3, 1.0)
            scene.add("Pole", "M_Dark", ml.transform(V, yaw, (x, y, z + 2.6)), F, UV)
        elif kind == "milestone":
            V, F, UV = ml.box(0.35, 0.2, 0.9)
            scene.add("Static", "M_Concrete", ml.transform(V, 0, (x, y, z - 0.3)), F, UV)
        elif kind == "camera":
            _pole(scene, x, y, z, 3.0, r=0.1)
            V, F, UV = ml.box(0.6, 0.5, 0.8)
            scene.add("Static", "M_Dark", ml.transform(V, 0, (x, y, z + 2.5)), F, UV)
        elif kind == "phone":
            V, F, UV = ml.box(0.45, 0.35, 1.3)
            scene.add("Static", "M_Orange", ml.transform(V, 0, (x, y, z - 0.2)), F, UV)
        elif kind == "power_pole":
            _pole(scene, x, y, z, float(_num(tags.get("height"), 9.0)), r=0.14, mat="M_Wood")
        elif kind in ("power_tower", "power_portal"):
            H = float(_num(tags.get("height"), 30.0))
            V, F, UV = ml.cylinder(3.0, 0.6, H, seg=4)
            scene.add("Pole", "M_Metal", ml.transform(V, math.pi / 4, (x, y, z - 0.5)), F, UV)
            V, F, UV = ml.box(14.0, 0.6, 0.6)
            scene.add("Pole", "M_Metal", ml.transform(V, 0, (x, y, z + H * 0.8)), F, UV)
        elif kind in ("mast", "flagpole", "surveillance"):
            H = float(_num(tags.get("height"), {"mast": 25.0, "flagpole": 8.0, "surveillance": 5.0}[kind]))
            _pole(scene, x, y, z, H, r=0.25 if kind == "mast" else 0.06)
        elif kind == "street_cabinet":
            V, F, UV = ml.box(0.9, 0.4, 1.3)
            scene.add("Static", "M_Metal", ml.transform(V, 0, (x, y, z - 0.1)), F, UV)
        elif kind == "billboard":
            nr = ridx.nearest(x, y, 60)
            yaw = 0.0
            if nr is not None:
                m, i = nr
                yaw = _yaw_facing(m.x[i] - x, m.y[i] - y)
            if tags.get("advertising") == "column":
                V, F, UV = ml.cylinder(0.6, 0.6, 3.0, seg=12)
                scene.add("Static", "M_Concrete", ml.transform(V, 0, (x, y, z)), F, UV)
            else:
                for dx in (-1.5, 1.5):
                    px, py = x + math.cos(yaw) * dx, y + math.sin(yaw) * dx
                    _pole(scene, px, py, z, 3.2, r=0.08)
                _panel(scene, pf, x, y, z + 4.2, yaw, "generic", size=3.0)
        elif kind.startswith("barrier_"):
            b = kind[8:]
            if b == "bollard":
                V, F, UV = ml.cylinder(0.1, 0.1, 1.2, seg=8)
                scene.add("Static", "M_Metal", ml.transform(V, 0, (x, y, z - 0.2)), F, UV)
            elif b == "block":
                V, F, UV = ml.box(1.0, 1.0, 1.0)
                scene.add("Static", "M_Concrete", ml.transform(V, 0, (x, y, z - 0.2)), F, UV)
            elif b == "toll_booth":
                nr = ridx.nearest(x, y)
                yaw = nr[0].hdg[nr[1]] if nr is not None else 0.0
                V, F, UV = ml.box(2.5, 1.6, 3.2)
                scene.add("Building", "M_WallDefault", ml.transform(V, yaw, (x, y, z - 0.2)), F, UV)
            else:
                _pole(scene, x, y, z, 1.2, r=0.1)
    log("  point objects: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return pf


def build_gantries(scene, osm, ground, ridx, pf, corridor, log):
    n = 0
    for wid, w in osm.ways.items():
        if w["tags"].get("man_made") != "gantry":
            continue
        P = osm.way_xy(wid)
        if P is None or not corridor.intersects(LineString(P)):
            continue
        a, b = P[0], P[-1]
        z = ground.surface(np.array([a[0], b[0]]), np.array([a[1], b[1]]))
        H = 6.8
        for (px, py), pz in zip((a, b), z):
            _pole(scene, px, py, pz, H + 0.8, r=0.3)
        zb = float(max(z)) + H
        seg = np.array([a, b])
        scene.add("Pole", "M_Metal", *ml.wall_along(seg, np.full(2, zb), 1.0, 0.8))
        mid = (a + b) / 2
        nr = ridx.nearest(mid[0], mid[1], 40)
        if nr is not None:
            m, i = nr
            t = m.T[i]
            yaw = _yaw_facing(-t[0], -t[1])
            L = np.hypot(*(b - a))
            k = max(1, int(L // 5))
            for j in range(k):
                c = a + (b - a) * (j + 0.5) / k
                _panel(scene, pf, c[0] - t[0] * 0.5, c[1] - t[1] * 0.5, zb - 0.6, yaw, "direction",
                       value=w["tags"].get("name", " "), size=2.2)
        n += 1
    if n:
        log(f"  {n} gantries")


def build_exit_panels(scene, osm, route, ground, pf, corridor, cfg, log):
    """Exit panels at the gore of each motorway_junction on the route,
    plus advance panels at 1000 m and 500 m (inferred)."""
    lang_exit = "Sortie" if cfg.get("country", "FR") == "FR" else "Exit"
    starts = {}
    for wid, w in osm.ways.items():
        h = w["tags"].get("highway", "")
        if (h.endswith("_link") or h == "service") and w["nodes"]:
            starts.setdefault(w["nodes"][0], []).append(wid)
    rtree = cKDTree(np.column_stack([route.x, route.y]))
    L, R = route.surface_edges()
    n = 0
    for nid, (lon, lat, tags) in osm.nodes.items():
        if tags.get("highway") != "motorway_junction" or nid not in osm.node_xy:
            continue
        x, y = osm.node_xy[nid]
        d, i = rtree.query([x, y])
        if d > 15:
            continue
        links = starts.get(nid, [])
        dest, dref = [], ""
        for wid in links:
            t = osm.ways[wid]["tags"]
            dest += [s.strip() for s in str(t.get("destination", "")).split(";") if s.strip()]
            dref = dref or t.get("destination:ref", "")
        ref = tags.get("ref", "")
        lines = [f"{lang_exit} {ref}".strip()]
        if dref:
            lines[0] += f"  {dref.replace(';', ' ')}"
        lines += dest[:3] or ([tags["name"]] if tags.get("name") else [])
        # gore position: follow the link until it is 5 m clear of the route surface
        gx, gy = None, None
        for wid in links:
            P = osm.way_xy(wid)
            if P is None:
                continue
            Q, _ = resample(P, 2.0)
            _, j = rtree.query(Q)
            lat = (Q[:, 0] - route.x[j]) * route.N[j, 0] + (Q[:, 1] - route.y[j]) * route.N[j, 1]
            clear = np.nonzero(lat < R[j] - 5.0)[0]
            if len(clear):
                k = clear[0]
                jj = j[k]
                ex = route.x[jj] + route.N[jj, 0] * R[jj]
                ey = route.y[jj] + route.N[jj, 1] * R[jj]
                gx, gy = (Q[k, 0] + ex) / 2, (Q[k, 1] + ey) / 2
                i = jj
                break
        t = route.T[i]
        yaw = _yaw_facing(-t[0], -t[1])
        if gx is not None:
            z = float(ground.surface(np.array([gx]), np.array([gy]))[0])
            w, h = _panel(scene, pf, gx, gy, z + 2.2 + 0.9, yaw, "exit", lines=lines, size=1.8)
            for dx in (-w / 3, w / 3):
                _pole(scene, gx + math.cos(yaw) * dx, gy + math.sin(yaw) * dx, z, 2.2, r=0.08)
            n += 1
        if cfg.get("infer_panels", True):
            for dist in (1000, 500):
                s_target = route.s[i] - dist
                if s_target < 0:
                    continue
                k = int(np.searchsorted(route.s, s_target))
                off = R[k] - 1.5
                px, py = route.x[k] + route.N[k, 0] * off, route.y[k] + route.N[k, 1] * off
                z = float(ground.surface(np.array([px]), np.array([py]))[0])
                tk = route.T[k]
                yk = _yaw_facing(-tk[0], -tk[1])
                adv = [f"{lang_exit} {ref}  {dist} m".strip()] + lines[1:]
                w, h = _panel(scene, pf, px, py, z + 2.2 + 0.9, yk, "exit", lines=adv, size=1.8)
                for dx in (-w / 3, w / 3):
                    _pole(scene, px + math.cos(yk) * dx, py + math.sin(yk) * dx, z, 2.2, r=0.08)
                n += 1
    log(f"  {n} exit panels")


# ================================================================ rail, water

def build_rail(scene, osm, elev, corridor, cfg, log):
    scene.material("M_Ballast", (0.45, 0.42, 0.38))
    scene.material("M_Rail", (0.40, 0.36, 0.33))
    models = []
    for wid, w in osm.ways.items():
        t = w["tags"]
        if t.get("railway") not in ("rail", "light_rail", "tram"):
            continue
        P = osm.way_xy(wid)
        if P is None or cumlen(P)[-1] < 5:
            continue
        Q, s = resample(P, STEP)
        m = RoadModel(f"rail{wid}", "rail", Q, s)
        m.bridge[:] = is_bridge(t)
        m.tunnel[:] = is_tunnel(t)
        m.wr = np.full((m.n, 1), 3.4)
        m.c0[:] = 1.7
        road_elevation(m, elev, 8.0)
        keep = shapely.contains_xy(corridor, m.x, m.y)
        for a, b in runs(keep):
            if b - a < 3:
                continue
            sub = m.subset(a, b)
            if sub.tunnel.all():
                continue
            models.append(sub)
            tram = t.get("railway") == "tram"
            for k in range(0, sub.n - 1, 40):
                sl = slice(k, min(sub.n, k + 41))
                if not tram:
                    A = sub.point(1.7, -0.05)[sl]
                    B = sub.point(-1.7, -0.05)[sl]
                    scene.add("RailTrack", "M_Ballast", *ml.ribbon(A, B))
                    scene.add("RailTrack", "M_Ballast", *ml.ribbon(A - [0, 0, 0.6], A))
                    scene.add("RailTrack", "M_Ballast", *ml.ribbon(B, B - [0, 0, 0.6]))
                for off in (0.7175, -0.7175):
                    P2 = sub.point(off)[sl]
                    scene.add("RailTrack", "M_Rail", *ml.wall_along(P2[:, :2], P2[:, 2] - 0.02, 0.17, 0.07))
            if t.get("electrified") == "contact_line":
                for k in range(10, sub.n, int(50 / STEP)):
                    p = sub.point(2.6)[k]
                    _pole(scene, p[0], p[1], p[2], 7.0, r=0.12)
    log(f"  {len(models)} railway pieces")
    return models


def build_water(scene, osm, ground, corridor, log):
    scene.material("M_Water", (0.18, 0.30, 0.38))
    polys = []
    for _, tags, poly in osm.areas(lambda t: t.get("natural") == "water"
                                   or t.get("landuse") in ("reservoir", "basin")):
        area = poly.intersection(corridor)
        if area.is_empty:
            continue
        for part in getattr(area, "geoms", [area]):
            if part.geom_type != "Polygon" or part.area < 10:
                continue
            ring = np.array(part.exterior.coords)
            z = float(ground.height(ring[:, 0], ring[:, 1]).min()) - 0.15
            scene.add("Water", "M_Water", *ml.polygon_cap(part, z))
            polys.append(part)
    if polys:
        log(f"  {len(polys)} water surfaces")
    return unary_union(polys) if polys else None
