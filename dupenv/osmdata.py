# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Download OpenStreetMap features along the route with the Overpass API."""
import json
import time
from xml.sax.saxutils import quoteattr

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from .geo import cumlen, dp_simplify
from .net import http_get

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

QUERY = """[out:json][timeout:300];
way{A}->.aw;
node{A}->.an;
rel{A}->.ar;
(
  way.aw["highway"];
  way.aw["building"];
  way.aw["building:part"];
  rel.ar["building"];
  way.aw["natural"="tree_row"];
  way.aw["barrier"];
  way.aw["wall"="noise_barrier"];
  way.aw["man_made"~"^(gantry|bridge)$"];
  way.aw["railway"~"^(rail|light_rail|tram)$"];
  node.an["natural"="tree"];
  node.an["barrier"~"^(toll_booth|bollard|block|gate|lift_gate)$"];
  node.an["traffic_sign"];
  node.an["highway"~"^(traffic_signals|stop|give_way|street_lamp|milestone|motorway_junction|speed_camera|emergency_access_point)$"];
  node.an["man_made"~"^(street_cabinet|mast|flagpole|surveillance)$"];
  node.an["power"~"^(pole|tower|portal)$"];
  node.an["advertising"];
  node.an["amenity"="emergency_phone"];
  way["natural"~"^(wood|scrub|water)$"]({B});
  way["landuse"~"^(forest|orchard|reservoir|basin)$"]({B});
  rel["natural"~"^(wood|scrub|water)$"]({B});
  rel["landuse"~"^(forest|orchard)$"]({B});
  way["landuse"]({B});
  rel["landuse"]["type"="multipolygon"]({B});
  way["leisure"~"^(park|garden|pitch|playground|golf_course|sports_centre)$"]({B});
  way["natural"~"^(grassland|heath)$"]({B});
);
out body;
>;
out skel qt;
"""


class OSMData:
    def __init__(self):
        self.nodes = {}      # id -> [lon, lat, tags]
        self.ways = {}       # id -> {"nodes": [...], "tags": {}}
        self.relations = {}  # id -> {"members": [(type, ref, role)], "tags": {}}
        self.node_xy = {}

    # ------------------------------------------------------------ loading
    def merge_json(self, js):
        for el in js.get("elements", []):
            t = el["type"]
            tags = el.get("tags", {})
            if t == "node":
                old = self.nodes.get(el["id"])
                if old is None or (tags and not old[2]):
                    self.nodes[el["id"]] = [el["lon"], el["lat"], tags]
            elif t == "way":
                old = self.ways.get(el["id"])
                if old is None or (tags and not old["tags"]):
                    self.ways[el["id"]] = {"nodes": el.get("nodes", []), "tags": tags}
            elif t == "relation":
                old = self.relations.get(el["id"])
                if old is None or (tags and not old["tags"]):
                    self.relations[el["id"]] = {
                        "members": [(m["type"], m["ref"], m.get("role", "")) for m in el.get("members", [])],
                        "tags": tags,
                    }

    def project(self, frame):
        ids = np.fromiter(self.nodes.keys(), dtype=np.int64, count=len(self.nodes))
        if len(ids) == 0:
            return
        lon = np.array([self.nodes[i][0] for i in ids])
        lat = np.array([self.nodes[i][1] for i in ids])
        x, y = frame.to_xy(lon, lat)
        self.node_xy = {int(i): (float(a), float(b)) for i, a, b in zip(ids, x, y)}

    # ------------------------------------------------------------ geometry
    def way_xy(self, wid):
        w = self.ways.get(wid)
        if w is None:
            return None
        pts = [self.node_xy[n] for n in w["nodes"] if n in self.node_xy]
        if len(pts) < 2:
            return None
        return np.array(pts)

    def way_polygon(self, wid):
        P = self.way_xy(wid)
        w = self.ways[wid]
        if P is None or len(P) < 4 or w["nodes"][0] != w["nodes"][-1]:
            return None
        poly = Polygon(P)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None

    def relation_polygon(self, rid):
        r = self.relations[rid]
        outer, inner = [], []
        for typ, ref, role in r["members"]:
            if typ != "way":
                continue
            P = self.way_xy(ref)
            if P is None:
                continue
            (inner if role == "inner" else outer).append(LineString(P))
        if not outer:
            return None
        o = unary_union(list(polygonize(unary_union(outer))))
        if o.is_empty:
            return None
        if inner:
            i = unary_union(list(polygonize(unary_union(inner))))
            if not i.is_empty:
                o = o.difference(i)
        return o if not o.is_empty else None

    def areas(self, predicate):
        """Yield (tags, polygon) for closed ways and multipolygon relations."""
        for wid, w in self.ways.items():
            if w["tags"] and predicate(w["tags"]):
                poly = self.way_polygon(wid)
                if poly is not None:
                    yield wid, w["tags"], poly
        for rid, r in self.relations.items():
            if r["tags"].get("type") in ("multipolygon", "building") and predicate(r["tags"]):
                poly = self.relation_polygon(rid)
                if poly is not None:
                    yield -rid, r["tags"], poly

    # ------------------------------------------------------------ export
    def write_osm_xml(self, path, bounds=None):
        """Write the corridor data as a plain OSM XML file (JOSM/CARLA/netconvert)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<osm version="0.6" generator="duplicat_env" upload="never">\n')
            if bounds:
                f.write(
                    f'  <bounds minlat="{bounds[1]:.7f}" minlon="{bounds[0]:.7f}" '
                    f'maxlat="{bounds[3]:.7f}" maxlon="{bounds[2]:.7f}"/>\n'
                )
            for nid in sorted(self.nodes):
                lon, lat, tags = self.nodes[nid]
                if tags:
                    f.write(f'  <node id="{nid}" version="1" visible="true" lat="{lat:.7f}" lon="{lon:.7f}">\n')
                    for k, v in tags.items():
                        f.write(f"    <tag k={quoteattr(k)} v={quoteattr(str(v))}/>\n")
                    f.write("  </node>\n")
                else:
                    f.write(f'  <node id="{nid}" version="1" visible="true" lat="{lat:.7f}" lon="{lon:.7f}"/>\n')
            for wid in sorted(self.ways):
                w = self.ways[wid]
                nds = [n for n in w["nodes"] if n in self.nodes]
                if len(nds) < 2:
                    continue
                f.write(f'  <way id="{wid}" version="1" visible="true">\n')
                for n in nds:
                    f.write(f'    <nd ref="{n}"/>\n')
                for k, v in w["tags"].items():
                    f.write(f"    <tag k={quoteattr(k)} v={quoteattr(str(v))}/>\n")
                f.write("  </way>\n")
            for rid in sorted(self.relations):
                r = self.relations[rid]
                f.write(f'  <relation id="{rid}" version="1" visible="true">\n')
                for typ, ref, role in r["members"]:
                    if (typ == "way" and ref in self.ways) or (typ == "node" and ref in self.nodes):
                        f.write(f'    <member type="{typ}" ref="{ref}" role={quoteattr(role)}/>\n')
                for k, v in r["tags"].items():
                    f.write(f"    <tag k={quoteattr(k)} v={quoteattr(str(v))}/>\n")
                f.write("  </relation>\n")
            f.write("</osm>\n")


def _pieces(route_xy, route_ll, piece_len=3000.0, overlap=100.0):
    s = cumlen(route_xy)
    L = s[-1]
    out = []
    start = 0.0
    while True:
        end = min(L, start + piece_len)
        m = (s >= start - overlap) & (s <= end + overlap)
        idx = np.nonzero(m)[0]
        if len(idx) >= 2:
            out.append(idx)
        if end >= L:
            break
        start = end
    return out


def _piece_query(route_xy, route_ll, idx, radius):
    sub_xy = route_xy[idx]
    tol = 2.0
    keep = dp_simplify(sub_xy, tol)
    while len(keep) > 120:
        tol *= 1.6
        keep = dp_simplify(sub_xy, tol)
    ll = route_ll[idx][keep]
    coords = ",".join(f"{lat:.6f},{lon:.6f}" for lon, lat in ll)
    around = f"(around:{int(radius)},{coords})"
    sub_ll = route_ll[idx]
    pad = radius / 111000.0 * 1.6
    bbox = (f"{sub_ll[:, 1].min() - pad:.6f},{sub_ll[:, 0].min() - pad * 1.5:.6f},"
            f"{sub_ll[:, 1].max() + pad:.6f},{sub_ll[:, 0].max() + pad * 1.5:.6f}")
    return QUERY.replace("{A}", around).replace("{B}", bbox)


def area_queries(frame, area_xy, piece=3000.0):
    """Overpass queries covering a polygon (area mode): one per ~3 km cell,
    each restricted to the part of the polygon inside the cell."""
    from shapely.geometry import box
    minx, miny, maxx, maxy = area_xy.bounds
    out = []
    for x0 in np.arange(minx, maxx, piece):
        for y0 in np.arange(miny, maxy, piece):
            part = area_xy.intersection(box(x0, y0, x0 + piece, y0 + piece))
            if part.is_empty or part.area < 1.0:
                continue
            for g in getattr(part, "geoms", [part]):
                if g.geom_type != "Polygon":
                    continue
                g = g.buffer(20).simplify(5.0)  # a margin, and few enough points for the URL
                ring = np.array(g.exterior.coords)
                lon, lat = frame.to_lonlat(ring[:, 0], ring[:, 1])
                poly = " ".join(f"{la:.6f} {lo:.6f}" for lo, la in zip(lon, lat))
                pad = 0.002
                bbox = f"{lat.min() - pad:.6f},{lon.min() - pad:.6f},{lat.max() + pad:.6f},{lon.max() + pad:.6f}"
                out.append(QUERY.replace("{A}", f'(poly:"{poly}")').replace("{B}", bbox))
    return out


def fetch_osm(frame, route_xy, route_ll, radius, log, progress=None, cancel=None, task="osm", queries=None):
    """Corridor data from Overpass, in ~3 km pieces fetched in parallel.

    Piece i starts on server i mod 4 and moves to the next server straight
    away when one is busy (429/504), so a slow instance never holds the job.
    `cancel` (threading.Event) stops the remaining pieces, used when the
    Geofabrik extract wins the race."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from . import progress as prog
    from .net import cached
    if queries is None:
        pieces = _pieces(route_xy, route_ll)
        queries = [_piece_query(route_xy, route_ll, idx, radius) for idx in pieces]
    n_srv = len(OVERPASS_URLS)
    counter = prog.Counter(task, len(queries), "piece")
    first_err = []
    lock = threading.Lock()

    def one(pi):
        q = queries[pi]
        # a piece already in the disk cache is read from it whatever server cached it
        order = sorted(range(n_srv), key=lambda k: (not cached(OVERPASS_URLS[k], {"data": q}), (k - pi) % n_srv))
        for rnd in range(4):
            for k in order:
                if cancel is not None and cancel.is_set():
                    return None
                url = OVERPASS_URLS[k]
                try:
                    body = http_get(url, data={"data": q}, timeout=320, retries=1)
                    js = json.loads(body)
                    if "remark" in js and "error" in js["remark"].lower():
                        raise RuntimeError(js["remark"])
                    counter.tick()
                    if progress:
                        progress(counter.n / counter.total)
                    return js
                except Exception as e:
                    with lock:
                        if not first_err:
                            first_err.append(e)
                    log(f"  Overpass piece {pi + 1}: {url.split('/')[2]} failed ({str(e)[:60]}), trying the next server")
            if cancel is not None and cancel.wait(10 * (rnd + 1)):
                return None
            if cancel is None:
                time.sleep(10 * (rnd + 1))
        raise RuntimeError(f"all Overpass servers failed for piece {pi + 1}: {first_err[:1]}")

    log(f"  {len(queries)} Overpass pieces, up to {2 * n_srv} in parallel on {n_srv} servers")
    with ThreadPoolExecutor(2 * n_srv) as ex:
        results = list(ex.map(one, range(len(queries))))
    if cancel is not None and cancel.is_set():
        return None
    data = OSMData()
    for js in results:
        data.merge_json(js)
    data.project(frame)
    log(f"  OSM: {len(data.nodes)} nodes, {len(data.ways)} ways, {len(data.relations)} relations")
    return data


def corridor_polygon(route_xy, radius):
    return LineString(route_xy).buffer(radius, quad_segs=8)


def contains_xy(poly, x, y):
    return shapely.contains_xy(poly, x, y)
