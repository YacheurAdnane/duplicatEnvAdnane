# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Land cover from free French open data (and OpenStreetMap everywhere).

* IGN LIDAR HD canopy height model (MNH, 1 m): real position and height of
  every tree, found as local maxima of the canopy height.
* IGN infrared aerial photo (ORTHOIMAGERY.ORTHOPHOTOS.IRC): vegetation index
  (NDVI) for grass / crops / bare soil / pavement, and a hint for conifers.
* IGN BD TOPO (WFS): vegetation zones with leaf type (broadleaf, conifer,
  mixed), hedges, and buildings with measured heights.

Outside France (or where a layer has no data) everything falls back to
OpenStreetMap: woods, tree rows, single trees and landuse polygons.
"""
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union

from .elevation import IGN_WMS, IGNDEM
from . import progress as prog
from .net import http_get

WFS = "https://data.geopf.fr/wfs/ows"
MNH_LAYER = "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.WGS84G"
IRC_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS.IRC"
RGB_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"

# ground classes -> terrain material
GRASS, MEADOW, CROP, FIELD, FOREST, PAVED, BARE = range(7)
CLASS_MATERIAL = {GRASS: "M_Grass", MEADOW: "M_Meadow", CROP: "M_Crop", FIELD: "M_Field",
                  FOREST: "M_ForestFloor", PAVED: "M_Paving", BARE: "M_Dirt"}

URBAN_LANDUSE = {"residential", "industrial", "commercial", "retail", "railway", "construction",
                 "garages", "military", "brownfield"}
GRASS_LANDUSE = {"grass", "village_green", "recreation_ground", "cemetery", "allotments"}
GRASS_LEISURE = {"park", "garden", "pitch", "playground", "golf_course", "sports_centre"}
MEADOW_LANDUSE = {"meadow", "greenfield"}
FARM_LANDUSE = {"farmland", "farmyard", "vineyard", "orchard", "plant_nursery"}


def _wfs(layer, bbox_ll, log, max_features=20000):
    """All features of a BD TOPO layer inside a lon/lat bbox (paged)."""
    x0, y0, x1, y1 = bbox_ll
    feats, start = [], 0
    while True:
        url = (f"{WFS}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES={layer}"
               f"&OUTPUTFORMAT=application/json&COUNT=5000&STARTINDEX={start}"
               f"&BBOX={y0:.6f},{x0:.6f},{y1:.6f},{x1:.6f},urn:ogc:def:crs:EPSG::4326")
        try:
            js = json.loads(http_get(url, timeout=120, retries=5))
        except Exception as e:
            log(f"  BD TOPO {layer.split(':')[-1]} unavailable: {str(e)[:80]}")
            break
        f = js.get("features", [])
        feats += f
        if len(f) < 5000 or len(feats) >= max_features:
            break
        start += 5000
    return feats


class LandCover:
    def __init__(self, frame, corridor, osm, source_is_ign, cfg, log, tiles=None):
        self.frame = frame
        self.corridor = corridor
        self.log = log
        self.france = source_is_ign
        self.chm = None
        self.irc = {}
        self.irc_px = 2.0
        self.bd_veg = []        # (polygon xy, kind)
        self.bd_buildings = []  # (polygon xy, height, usage)
        self.bd_hedges = []     # polygons xy
        self._veg_index = None
        ext = corridor.buffer(30).exterior
        lon, lat = frame.to_lonlat(*np.array(ext.coords).T)
        self.ll_poly = Polygon(np.stack([lon, lat], 1)).buffer(0)
        self.landuse = []
        self._lu_index = None
        # the three French sources download at the same time
        jobs = []
        if self.france and cfg.get("lidar_trees", True):
            jobs.append(("lidar", "LIDAR HD canopy (trees)", self._load_chm))
        if self.france and cfg.get("bdtopo", True):
            jobs.append(("bdtopo", "BD TOPO (vegetation, hedges, buildings)", self._load_bdtopo))
        if self.france and tiles is not None:
            jobs.append(("photo", "infrared aerial photo", lambda: self.prefetch_irc(tiles)))
        threads = []
        for key, label, fn in jobs:
            th = threading.Thread(target=self._run_job, args=(key, label, fn), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        if osm is not None:
            self._load_osm_landuse(osm)

    def _run_job(self, key, label, fn):
        prog.begin(key, label, parent="download")
        try:
            fn()
            prog.end(key)
        except Exception as e:
            self.log(f"  {label} failed: {str(e)[:100]}")
            prog.end(key, "failed", str(e)[:60])

    def _load_chm(self):
        self.chm = IGNDEM(self.ll_poly, res=1.0, log=self.log, layer=MNH_LAYER,
                          label="IGN LIDAR HD canopy height", task="lidar")
        if all(t is None or np.isnan(t).all() for t in self.chm.tiles.values()):
            self.log("  no LIDAR HD coverage here yet, trees come from OpenStreetMap")
            self.chm = None

    def add_osm(self, osm):
        self._load_osm_landuse(osm)

    # ------------------------------------------------------------ sources
    def _to_xy_poly(self, geom):
        def conv(poly):
            ex = np.array(poly.exterior.coords)
            x, y = self.frame.to_xy(ex[:, 0], ex[:, 1])
            holes = []
            for r in poly.interiors:
                h = np.array(r.coords)
                hx, hy = self.frame.to_xy(h[:, 0], h[:, 1])
                holes.append(np.column_stack([hx, hy]))
            return Polygon(np.column_stack([x, y]), holes).buffer(0)
        g = shape(geom)
        if g.geom_type == "Polygon":
            return conv(g)
        if g.geom_type == "MultiPolygon":
            return unary_union([conv(p) for p in g.geoms])
        return None

    def _load_bdtopo(self):
        bb = self.ll_poly.bounds
        # split long corridors into ~3 km boxes so each WFS request stays small
        nx = max(1, int((bb[2] - bb[0]) / 0.04) + 1)
        ny = max(1, int((bb[3] - bb[1]) / 0.03) + 1)
        boxes = [box(bb[0] + (bb[2] - bb[0]) * i / nx, bb[1] + (bb[3] - bb[1]) * j / ny,
                     bb[0] + (bb[2] - bb[0]) * (i + 1) / nx, bb[1] + (bb[3] - bb[1]) * (j + 1) / ny)
                 for i in range(nx) for j in range(ny)]
        boxes = [b for b in boxes if b.intersects(self.ll_poly)]
        layers = ("BDTOPO_V3:zone_de_vegetation", "BDTOPO_V3:haie", "BDTOPO_V3:batiment")
        # every (layer, box) request in parallel, then parsed in order
        c = prog.Counter("bdtopo", len(layers) * len(boxes), "request")

        def get(lb):
            r = _wfs(lb[0], lb[1].bounds, self.log)
            c.tick()
            return r
        with ThreadPoolExecutor(8) as ex:
            got = dict(zip([(ly, i) for ly in layers for i in range(len(boxes))],
                           ex.map(get, [(ly, b) for ly in layers for b in boxes])))
        seen = set()
        for layer in layers:
            n = 0
            for bi in range(len(boxes)):
                for f in got[(layer, bi)]:
                    fid = f.get("id")
                    if fid in seen or not f.get("geometry"):
                        continue
                    seen.add(fid)
                    try:
                        poly = self._to_xy_poly(f["geometry"])
                    except Exception:
                        continue
                    if poly is None or poly.is_empty or not poly.intersects(self.corridor.buffer(30)):
                        continue
                    p = f["properties"]
                    if layer.endswith("zone_de_vegetation"):
                        self.bd_veg.append((poly, self._veg_kind(p.get("nature", ""))))
                    elif layer.endswith("haie"):
                        self.bd_hedges.append(poly)
                    else:
                        self.bd_buildings.append((poly, p.get("hauteur"), p.get("usage_1") or ""))
                    n += 1
            self.log(f"  BD TOPO {layer.split(':')[-1]}: {n}")
        self._veg_index = shapely.STRtree([p for p, _ in self.bd_veg]) if self.bd_veg else None

    @staticmethod
    def _veg_kind(nature):
        n = nature.lower()
        if "conif" in n:
            return "conifer"
        if "mixte" in n:
            return "mixed"
        if "haie" in n:
            return "hedge"
        if "lande" in n:
            return "scrub"
        if "vigne" in n or "verger" in n:
            return "orchard"
        if "for" in n or "bois" in n or "feuill" in n or "peupl" in n:
            return "broadleaf"
        return "other"

    def _load_osm_landuse(self, osm):
        polys, kinds = [], []
        for _, tags, poly in osm.areas(lambda t: "landuse" in t or "leisure" in t
                                       or t.get("natural") in ("grassland", "heath", "scrub", "wood")):
            lu = tags.get("landuse", "")
            le = tags.get("leisure", "")
            nat = tags.get("natural", "")
            if lu in URBAN_LANDUSE:
                k = "urban"
            elif lu in GRASS_LANDUSE or le in GRASS_LEISURE:
                k = "grass"
            elif lu in MEADOW_LANDUSE or nat in ("grassland", "heath"):
                k = "meadow"
            elif lu in FARM_LANDUSE:
                k = "farm"
            elif lu == "forest" or nat == "wood":
                k = "forest"
            elif nat == "scrub":
                k = "meadow"
            else:
                continue
            if poly.intersects(self.corridor):
                polys.append(poly)
                kinds.append(k)
        self.landuse = list(zip(polys, kinds))
        self._lu_index = shapely.STRtree(polys) if polys else None

    # ------------------------------------------------------------ rasters
    def chm_xy(self, x, y):
        if self.chm is None:
            return np.zeros(len(x))
        lon, lat = self.frame.to_lonlat(x, y)
        v = self.chm.sample(lon, lat)
        return np.nan_to_num(v, nan=0.0)

    def _irc_tile(self, key):
        if key in self.irc:
            return self.irc[key]
        i, j = key
        t = 500.0
        xs = np.array([i * t - 5, (i + 1) * t + 5])
        ys = np.array([j * t - 5, (j + 1) * t + 5])
        lon, lat = self.frame.to_lonlat(np.array([xs[0], xs[1], xs[0], xs[1]]),
                                        np.array([ys[0], ys[0], ys[1], ys[1]]))
        bb = (lon.min(), lat.min(), lon.max(), lat.max())
        px = int(510 / self.irc_px)
        url = (f"{IGN_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={IRC_LAYER}&STYLES="
               f"&CRS=EPSG:4326&BBOX={bb[1]:.9f},{bb[0]:.9f},{bb[3]:.9f},{bb[2]:.9f}"
               f"&WIDTH={px}&HEIGHT={px}&FORMAT=image/jpeg")
        try:
            im = np.asarray(Image.open(io.BytesIO(http_get(url, timeout=120, retries=6))).convert("RGB"), np.float32)
            nir, red = im[:, :, 0], im[:, :, 1]
            ndvi = (nir - red) / np.maximum(nir + red, 1.0)
        except Exception:
            self.irc[key] = None
            return None
        # the colour photo of the same square: the infrared one is often from
        # another season, when verges and gardens are dry
        exg = sat = lum = None
        try:
            rgb = np.asarray(Image.open(io.BytesIO(http_get(url.replace(f"LAYERS={IRC_LAYER}", f"LAYERS={RGB_LAYER}"),
                                                             timeout=120, retries=6))).convert("RGB"), np.float32)
            if rgb.shape == im.shape:
                s = np.maximum(rgb.sum(axis=2), 1.0)
                exg = ((2 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]) / s).astype(np.float32)  # excess green
                sat = ((rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 1.0)).astype(np.float32)
                lum = (rgb.mean(axis=2) / 255.0).astype(np.float32)
        except Exception:
            pass
        self.irc[key] = (bb, ndvi.astype(np.float32), nir.astype(np.float32), exg, sat, lum)
        return self.irc[key]

    def prefetch_irc(self, tiles):
        if not self.france:
            return
        c = prog.Counter("photo", len(tiles), "tile")

        def one(k):
            self._irc_tile(k)
            c.tick()
        with ThreadPoolExecutor(10) as ex:
            list(ex.map(one, tiles))

    def photo_xy(self, x, y):
        """Excess green, saturation and brightness of the colour photo at points (NaN where unknown)."""
        exg = np.full(len(x), np.nan, np.float32)
        sat = np.full(len(x), np.nan, np.float32)
        lum = np.full(len(x), np.nan, np.float32)
        if not self.france:
            return exg, sat, lum
        ti = np.floor(np.asarray(x) / 500.0).astype(int)
        tj = np.floor(np.asarray(y) / 500.0).astype(int)
        lon, lat = self.frame.to_lonlat(x, y)
        for key in set(zip(ti.tolist(), tj.tolist())):
            t = self._irc_tile(key)
            if t is None or t[3] is None:
                continue
            bb, e, s, lu_ = t[0], t[3], t[4], t[5]
            m = (ti == key[0]) & (tj == key[1])
            h, w = e.shape
            c = np.clip(((lon[m] - bb[0]) / (bb[2] - bb[0]) * w).astype(int), 0, w - 1)
            r = np.clip(((bb[3] - lat[m]) / (bb[3] - bb[1]) * h).astype(int), 0, h - 1)
            exg[m], sat[m], lum[m] = e[r, c], s[r, c], lu_[r, c]
        return exg, sat, lum

    def ndvi_xy(self, x, y, want_nir=False):
        """NDVI (and infrared) at points; NaN where unknown."""
        out = np.full(len(x), np.nan, np.float32)
        nir_out = np.full(len(x), np.nan, np.float32)
        if not self.france:
            return (out, nir_out) if want_nir else out
        ti = np.floor(np.asarray(x) / 500.0).astype(int)
        tj = np.floor(np.asarray(y) / 500.0).astype(int)
        lon, lat = self.frame.to_lonlat(x, y)
        for key in set(zip(ti.tolist(), tj.tolist())):
            t = self._irc_tile(key)
            if t is None:
                continue
            bb, ndvi, nir = t[:3]
            m = (ti == key[0]) & (tj == key[1])
            h, w = ndvi.shape
            c = np.clip(((lon[m] - bb[0]) / (bb[2] - bb[0]) * w).astype(int), 0, w - 1)
            r = np.clip(((bb[3] - lat[m]) / (bb[3] - bb[1]) * h).astype(int), 0, h - 1)
            out[m] = ndvi[r, c]
            nir_out[m] = nir[r, c]
        return (out, nir_out) if want_nir else out

    # ------------------------------------------------------------ queries
    def veg_kind_at(self, x, y):
        kinds = np.array(["other"] * len(x), dtype=object)
        if self._veg_index is None:
            return kinds
        pts = shapely.points(x, y)
        a, b = self._veg_index.query(pts, predicate="within")
        kinds[a] = [self.bd_veg[k][1] for k in b]
        return kinds

    def landuse_at(self, x, y):
        out = np.array([""] * len(x), dtype=object)
        if self._lu_index is None:
            return out
        a, b = self._lu_index.query(shapely.points(x, y), predicate="within")
        out[a] = [self.landuse[k][1] for k in b]
        return out

    def ground_class(self, x, y, near_building=None):
        """Ground class for terrain faces at (x, y)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = len(x)
        cls = np.full(n, MEADOW, np.int8)
        lu = self.landuse_at(x, y)
        cls[lu == "grass"] = GRASS
        cls[lu == "farm"] = CROP
        cls[lu == "urban"] = GRASS
        cls[lu == "forest"] = FOREST
        vk = self.veg_kind_at(x, y)
        cls[np.isin(vk, ["broadleaf", "conifer", "mixed"])] = FOREST
        ndvi = self.ndvi_xy(x, y)
        known = np.isfinite(ndvi)
        if known.any():
            urban = (lu == "urban") | (near_building if near_building is not None else False)
            low = known & (ndvi < 0.08)
            mid = known & (ndvi >= 0.08) & (ndvi < 0.25)
            high = known & (ndvi >= 0.25)
            cls[low & urban] = PAVED
            cls[low & ~urban & (lu == "farm")] = FIELD
            # dry summer grass has a low index too: only really bare ground becomes dirt
            cls[low & ~urban & (lu != "farm")] = MEADOW
            cls[known & (ndvi < -0.02) & ~urban & (lu != "farm")] = BARE
            cls[mid & (lu == "farm")] = FIELD
            cls[mid & (lu != "farm") & ~urban & (cls != FOREST)] = MEADOW
            cls[high & (lu == "farm")] = CROP
            cls[high & urban & (cls != FOREST)] = GRASS
        # colour photo: green there means grass whatever the infrared says;
        # grey (unsaturated, not green) in town means paved
        exg, sat, lum = self.photo_xy(x, y)
        seen = np.isfinite(exg)
        if seen.any():
            urban = (lu == "urban") | (near_building if near_building is not None else False)
            green = seen & (exg > 0.05)
            lush = seen & (exg > 0.09)
            grey = seen & (sat < 0.12) & (exg < 0.02)
            free = (cls != FOREST) & (cls != CROP) & (cls != FIELD)
            cls[green & free & (cls != GRASS)] = MEADOW
            cls[lush & free] = GRASS
            cls[green & urban & (cls == PAVED)] = GRASS
            cls[grey & urban & free & ~green] = PAVED
            cls[(cls == PAVED) & ~urban & seen & ~grey] = BARE
            # light, not green, outside town: harvested field or dry stubble
            stubble = seen & ~urban & ~green & (lum > 0.50) & (cls != FOREST) & (cls != PAVED)
            cls[stubble] = CROP
        if self.chm is not None:
            h = self.chm_xy(x, y)
            cls[(h > 4) & (~known | (ndvi > 0.15))] = FOREST
        return cls

    # ------------------------------------------------------------ trees
    def detect_trees(self, tiles, ground, blocked, rng, log):
        """Trees from the canopy height model: list of (x, y, height, width, kind)."""
        if self.chm is None:
            return None
        bmask = blocked.buffer(1.5) if blocked is not None else None
        bd_b = unary_union([p for p, _, _ in self.bd_buildings]).buffer(1.5) if self.bd_buildings else None
        out = []
        stats = np.zeros(6, int)
        from .geo import thread_geom

        def one(ij):
            from .progress import wait_for_ram
            wait_for_ram()
            i, j = ij
            corridor_t, bmask_t, bd_t = thread_geom(self.corridor), thread_geom(bmask), thread_geom(bd_b)
            st = np.zeros(6, int)
            xs = np.arange(i * 500.0, (i + 1) * 500.0, 1.0) + 0.5
            ys = np.arange(j * 500.0, (j + 1) * 500.0, 1.0) + 0.5
            X, Y = np.meshgrid(xs, ys)
            inside = shapely.contains_xy(corridor_t, X, Y)
            if not inside.any():
                return None, st
            H = np.zeros(X.shape, np.float32)
            H[inside] = self.chm_xy(X[inside], Y[inside])
            S = ndimage.gaussian_filter(H, 1.0)
            peak = (S == ndimage.maximum_filter(S, size=5)) & (S > 3.0) & inside
            r, c = np.nonzero(peak)
            if len(r) == 0:
                return None, st
            px, py, ph = X[r, c], Y[r, c], H[r, c]
            st[0] += len(px)
            ok = ph > 3.0
            if bmask is not None:
                ok &= ~shapely.contains_xy(bmask_t, px, py)
            if bd_b is not None:
                ok &= ~shapely.contains_xy(bd_t, px, py)
            st[1] += ok.sum()
            ok &= ~ground.near_road(px, py, 1.5)
            st[2] += ok.sum()
            ndvi = self.ndvi_xy(px, py)
            ok &= ~(np.isfinite(ndvi) & (ndvi < 0.1))  # tall but not green: lamp, truck, roof
            st[3] += ok.sum()
            r, c, px, py, ph = r[ok], c[ok], px[ok], py[ok], ph[ok]
            if len(px) == 0:
                return None, st
            # non-maximum suppression, tallest first, radius grows with height
            order = np.argsort(-ph)
            tree = cKDTree(np.column_stack([px, py]))
            taken = np.zeros(len(px), bool)
            keep = []
            for k in order:
                if taken[k]:
                    continue
                keep.append(k)
                for q in tree.query_ball_point([px[k], py[k]], 0.22 * ph[k] + 1.5):
                    taken[q] = True
            keep = np.array(keep)
            st[4] += len(keep)
            r, c, px, py, ph = r[keep], c[keep], px[keep], py[keep], ph[keep]
            # crown radius: distance where the smoothed canopy drops under half the height
            rad = np.zeros(len(px))
            dirs = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
            for dx, dy in dirs:
                found = np.full(len(px), 8.0)
                for d in range(1, 9):
                    rr = np.clip((r + dy * d).astype(int), 0, S.shape[0] - 1)
                    cc = np.clip((c + dx * d).astype(int), 0, S.shape[1] - 1)
                    below = (S[rr, cc] < 0.5 * ph) & (found == 8.0)
                    found[below] = d
                rad += found
            width = np.clip(2 * rad / len(dirs), 2.0, 16.0)
            return np.column_stack([px, py, ph, width]), st

        from .progress import Counter, workers
        cnt = Counter("vegetation", len(tiles), "LIDAR tile")
        with ThreadPoolExecutor(workers("cpu")) as ex:
            for arr, st in ex.map(one, tiles):
                stats += st
                cnt.tick()
                if arr is not None:
                    out.append(arr)
        log(f"  canopy peaks {stats[0]}, not on buildings {stats[1]}, off roads {stats[2]}, "
            f"green {stats[3]}, after spacing {stats[4]}")
        if not out:
            return np.zeros((0, 4)), np.array([], dtype=object)
        T = np.concatenate(out)
        kind = self.veg_kind_at(T[:, 0], T[:, 1])
        # infrared hint where BD TOPO does not say: conifers are darker in near infrared
        _, nir = self.ndvi_xy(T[:, 0], T[:, 1], want_nir=True)
        med = np.nanmedian(nir) if np.isfinite(nir).any() else np.nan
        unknown = ~np.isin(kind, ["broadleaf", "conifer"])
        mixed = kind == "mixed"
        conif = np.isfinite(nir) & (nir < 0.72 * med) if np.isfinite(med) else np.zeros(len(T), bool)
        final = np.where(kind == "conifer", "conifer", "broadleaf").astype(object)
        final[unknown & conif] = "conifer"
        final[mixed & ~conif] = "broadleaf"
        log(f"  LIDAR HD: {len(T)} trees found "
            f"({int((final == 'conifer').sum())} conifers, heights {np.percentile(T[:, 2], 10):.0f}-"
            f"{np.percentile(T[:, 2], 90):.0f} m)")
        return T, final
