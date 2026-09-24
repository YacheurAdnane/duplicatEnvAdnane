# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Free elevation sources.

* IGN RGE ALTI (France, 1 m lidar based terrain model) through the open
  Geoplateforme WMS-Raster service. No key needed.
* AWS Terrain Tiles (Terrarium PNG, worldwide, ~10-30 m). No key needed.

`Elevation.sample_xy(x, y)` returns ground height in metres for points in the
local frame.
"""
import io
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box

from . import progress as prog
from .net import http_get

IGN_WMS = "https://data.geopf.fr/wms-r/wms"
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
FRANCE_BBOX = (-5.3, 41.3, 9.7, 51.2)


class TerrariumDEM:
    def __init__(self, lonlat_poly, zoom=14, log=print, task="dem"):
        self.z = zoom
        self.n = 2 ** zoom
        self.tiles = {}
        minx, miny, maxx, maxy = lonlat_poly.bounds
        x0, y1 = self._tile(minx, miny)
        x1, y0 = self._tile(maxx, maxy)
        need = []
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                if lonlat_poly.intersects(self._tile_box(tx, ty)):
                    need.append((tx, ty))
        log(f"  Terrarium DEM: {len(need)} tiles at zoom {zoom}")
        c = prog.Counter(task, len(need), "tile")
        with ThreadPoolExecutor(12) as ex:
            for key, arr in zip(need, ex.map(self._fetch, need)):
                self.tiles[key] = arr
                c.tick()

    def _tile(self, lon, lat):
        px, py = self._pix(np.array([lon]), np.array([lat]))
        return int(px[0] // 256), int(py[0] // 256)

    def _tile_box(self, tx, ty):
        def lon(x):
            return x / self.n * 360.0 - 180.0

        def lat(y):
            return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / self.n))))

        return box(lon(tx), lat(ty + 1), lon(tx + 1), lat(ty))

    def _pix(self, lon, lat):
        lat = np.clip(lat, -85, 85)
        px = (lon + 180.0) / 360.0 * self.n * 256
        lr = np.radians(lat)
        py = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * self.n * 256
        return px, py

    def _fetch(self, key):
        tx, ty = key
        try:
            raw = http_get(TERRARIUM.format(z=self.z, x=tx, y=ty), timeout=60)
            im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float64)
            return im[:, :, 0] * 256 + im[:, :, 1] + im[:, :, 2] / 256 - 32768
        except Exception:
            return None

    def _get(self, ix, iy):
        out = np.zeros(len(ix))
        tx, ty = ix // 256, iy // 256
        keys = tx * 10_000_000 + ty
        for k in np.unique(keys):
            m = keys == k
            t = self.tiles.get((int(k // 10_000_000), int(k % 10_000_000)))
            if t is None:
                out[m] = np.nan
            else:
                out[m] = t[iy[m] % 256, ix[m] % 256]
        return out

    def sample(self, lon, lat):
        px, py = self._pix(np.asarray(lon, float), np.asarray(lat, float))
        px -= 0.5
        py -= 0.5
        ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        fx, fy = px - ix, py - iy
        a = self._get(ix, iy)
        b = self._get(ix + 1, iy)
        c = self._get(ix, iy + 1)
        d = self._get(ix + 1, iy + 1)
        v = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
        return v


class IGNDEM:
    """IGN float rasters (RGE ALTI terrain, LIDAR HD canopy height...) requested
    on a global pixel grid so tiles join seamlessly."""

    N = 500
    LAYER = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"

    def __init__(self, lonlat_poly, res=2.0, log=print, layer=None, label="IGN RGE ALTI", task="dem"):
        self.layer = layer or self.LAYER
        minx, miny, maxx, maxy = lonlat_poly.bounds
        lat_c = 0.5 * (miny + maxy)
        self.dlat = res / 111320.0
        self.dlon = res / (111320.0 * math.cos(math.radians(lat_c)))
        self.lon_o = minx - 2 * self.dlon
        self.lat_o = miny - 2 * self.dlat
        span = self.N
        ni = int((maxx - self.lon_o) / self.dlon / span) + 1
        nj = int((maxy - self.lat_o) / self.dlat / span) + 1
        need = []
        for i in range(ni):
            for j in range(nj):
                if lonlat_poly.intersects(self._tile_box(i, j)):
                    need.append((i, j))
        log(f"  {label}: {len(need)} tiles of {self.N * res:.0f} m at {res} m/px")
        self.tiles = {}
        c = prog.Counter(task, len(need), "tile")
        with ThreadPoolExecutor(10) as ex:
            for key, arr in zip(need, ex.map(self._fetch, need)):
                self.tiles[key] = arr
                c.tick()
        bad = sum(1 for a in self.tiles.values() if a is None)
        if bad:
            log(f"  {bad} {label} tiles failed")

    def _tile_box(self, i, j):
        N = self.N
        return box(
            self.lon_o + i * N * self.dlon - self.dlon / 2,
            self.lat_o + j * N * self.dlat - self.dlat / 2,
            self.lon_o + (i + 1) * N * self.dlon + self.dlon / 2,
            self.lat_o + (j + 1) * N * self.dlat + self.dlat / 2,
        )

    def _fetch(self, key):
        i, j = key
        b = self._tile_box(i, j).bounds
        W = self.N + 1
        url = (
            f"{IGN_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&LAYERS={self.layer}&STYLES=&CRS=EPSG:4326"
            f"&BBOX={b[1]:.9f},{b[0]:.9f},{b[3]:.9f},{b[2]:.9f}&WIDTH={W}&HEIGHT={W}"
            "&FORMAT=image/x-bil;bits=32"
        )
        try:
            raw = http_get(url, timeout=120, retries=6)
            a = np.frombuffer(raw, dtype="<f4")
            if a.size != W * W:
                return None
            a = a.reshape(W, W)[::-1].astype(np.float32)  # row 0 = south
            a[a < -1000] = np.nan
            return a
        except Exception:
            return None

    def sample(self, lon, lat):
        u = (np.asarray(lon, float) - self.lon_o) / self.dlon
        v = (np.asarray(lat, float) - self.lat_o) / self.dlat
        N = self.N
        ti = np.clip(np.floor(u / N).astype(np.int64), 0, None)
        tj = np.clip(np.floor(v / N).astype(np.int64), 0, None)
        out = np.full(len(u), np.nan)
        keys = ti * 100000 + tj
        for k in np.unique(keys):
            m = keys == k
            t = self.tiles.get((int(k // 100000), int(k % 100000)))
            if t is None:
                continue
            lu = np.clip(u[m] - (k // 100000) * N, 0, N - 1e-9)
            lv = np.clip(v[m] - (k % 100000) * N, 0, N - 1e-9)
            x0, y0 = np.floor(lu).astype(int), np.floor(lv).astype(int)
            fx, fy = lu - x0, lv - y0
            out[m] = (
                t[y0, x0] * (1 - fx) * (1 - fy)
                + t[y0, x0 + 1] * fx * (1 - fy)
                + t[y0 + 1, x0] * (1 - fx) * fy
                + t[y0 + 1, x0 + 1] * fx * fy
            )
        return out


def in_france(frame, corridor_xy_poly):
    ext = corridor_xy_poly.buffer(60).exterior
    lon, lat = frame.to_lonlat(*np.array(ext.coords).T)
    return box(*FRANCE_BBOX).contains(Polygon(np.stack([lon, lat], 1)).buffer(0))


class Elevation:
    def __init__(self, frame, corridor_xy_poly, source="auto", ign_res=2.0, log=print):
        self.frame = frame
        ext = corridor_xy_poly.buffer(60).exterior
        lon, lat = frame.to_lonlat(*np.array(ext.coords).T)
        ll_poly = Polygon(np.stack([lon, lat], 1)).buffer(0)
        in_france = box(*FRANCE_BBOX).contains(ll_poly)
        if source == "auto":
            source = "ign" if in_france else "terrarium"
        self.source = source
        self.ign = IGNDEM(ll_poly, res=ign_res, log=log) if source == "ign" else None
        # Terrarium is always loaded: primary source outside France, gap filler inside
        self.terr = TerrariumDEM(ll_poly, zoom=14 if source == "terrarium" else 12, log=log,
                                 task="dem" if source == "terrarium" else "dem_fill")
        log(f"  elevation source: {source}")

    def sample_lonlat(self, lon, lat):
        lon = np.atleast_1d(np.asarray(lon, float))
        lat = np.atleast_1d(np.asarray(lat, float))
        v = self.ign.sample(lon, lat) if self.ign is not None else np.full(len(lon), np.nan)
        bad = np.isnan(v)
        if bad.any():
            v[bad] = self.terr.sample(lon[bad], lat[bad])
        v[np.isnan(v)] = np.nanmean(v) if np.isfinite(v).any() else 0.0
        return v

    def sample_xy(self, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        out = np.empty(len(x))
        for a in range(0, len(x), 200_000):
            lon, lat = self.frame.to_lonlat(x[a : a + 200_000], y[a : a + 200_000])
            out[a : a + 200_000] = self.sample_lonlat(lon, lat)
        return out
