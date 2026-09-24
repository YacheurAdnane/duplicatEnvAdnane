# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Offline fallback for OSM data: Geofabrik regional extract (.osm.pbf) read
with pyosmium. Used when the public Overpass servers are overloaded.

The smallest Geofabrik region containing the route is downloaded once into
cache/ (for Ile-de-France about 300 MB) and filtered locally.
"""
import json
import os
import urllib.request

from shapely.geometry import box, shape

from . import progress as prog
from .net import CACHE_DIR, UA, http_get
from .osmdata import OSMData


class Cancelled(Exception):
    pass

INDEX_URL = "https://download.geofabrik.de/index-v1.json"

NODE_KEYS = ("natural", "highway", "traffic_sign", "barrier", "man_made", "power", "advertising", "amenity")
WAY_KEYS = ("highway", "building", "building:part", "natural", "landuse", "barrier", "wall", "man_made",
            "railway", "leisure")
REL_KEYS = ("building", "natural", "landuse")

NODE_OK = {
    "natural": {"tree"},
    "highway": {"traffic_signals", "stop", "give_way", "street_lamp", "milestone", "motorway_junction",
                "speed_camera", "emergency_access_point"},
    "barrier": {"toll_booth", "bollard", "block", "gate", "lift_gate"},
    "man_made": {"street_cabinet", "mast", "flagpole", "surveillance"},
    "power": {"pole", "tower", "portal"},
    "amenity": {"emergency_phone"},
}


def _node_wanted(tags):
    if "traffic_sign" in tags or "advertising" in tags:
        return True
    return any(tags.get(k) in v for k, v in NODE_OK.items())


def _way_wanted(tags):
    if "highway" in tags or "building" in tags or "building:part" in tags or "barrier" in tags:
        return True
    if tags.get("natural") in ("tree_row", "wood", "scrub", "water"):
        return True
    if tags.get("landuse") or tags.get("natural") in ("grassland", "heath"):
        return True
    if tags.get("leisure") in ("park", "garden", "pitch", "playground", "golf_course", "sports_centre"):
        return True
    if tags.get("wall") == "noise_barrier" or tags.get("man_made") in ("gantry", "bridge"):
        return True
    return tags.get("railway") in ("rail", "light_rail", "tram")


def pick_region(lonlat_bbox, log):
    raw = http_get(INDEX_URL, timeout=120)
    idx = json.loads(raw)
    target = box(*lonlat_bbox)
    best = None
    for feat in idx["features"]:
        pbf = feat["properties"].get("urls", {}).get("pbf")
        if not pbf or not feat.get("geometry"):
            continue
        g = shape(feat["geometry"])
        if g.contains(target) and (best is None or g.area < best[0]):
            best = (g.area, feat["properties"]["id"], pbf)
    if best is None:
        raise RuntimeError("no Geofabrik extract covers this route")
    log(f"  Geofabrik region: {best[1]}")
    return best[1], best[2]


def download(url, log, cancel=None, task="osm_pbf"):
    os.makedirs(os.path.join(CACHE_DIR, "pbf"), exist_ok=True)
    dest = os.path.join(CACHE_DIR, "pbf", os.path.basename(url))
    if os.path.exists(dest):
        return dest
    log(f"  downloading {url} (one time)...")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got, last = 0, 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                prog.update(task, frac=0.5 * got / total, detail=f"download {got / 1e6:.0f}/{total / 1e6:.0f} MB (one time)")
            if total and got - last > total / 10:
                last = got
                log(f"    {got / 1e6:.0f}/{total / 1e6:.0f} MB")
            if cancel is not None and cancel.is_set():
                raise Cancelled()
    os.replace(tmp, dest)
    return dest


def _mask(lonlat_bbox, area_ll, cell=0.0004):
    """inside(lon, lat) for the corridor: a boolean grid of ~100 m cells
    (anything touching the corridor counts), so the per-node test is cheap."""
    import numpy as np
    import shapely
    x0, y0, x1, y1 = lonlat_bbox
    nx, ny = int((x1 - x0) / cell) + 2, int((y1 - y0) / cell) + 2
    if area_ll is None:
        return lambda lon, lat: x0 <= lon <= x1 and y0 <= lat <= y1
    grown = area_ll.buffer(cell * 0.75)  # a cell counts when its centre is within ~half a cell
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * cell, y0 + (np.arange(ny) + 0.5) * cell)
    m = shapely.contains_xy(grown, gx, gy)

    def inside(lon, lat):
        i, j = int((lon - x0) / cell), int((lat - y0) / cell)
        return 0 <= i < nx and 0 <= j < ny and bool(m[j, i])
    return inside


def extract(pbf, lonlat_bbox, log, area_ll=None, cancel=None, task="osm_pbf"):
    import osmium
    from osmium.filter import IdFilter, KeyFilter

    x0, y0, x1, y1 = lonlat_bbox
    data = OSMData()
    inside = _mask(lonlat_bbox, area_ll)
    area_prep = None
    if area_ll is not None:
        from shapely.prepared import prep
        area_prep = prep(area_ll.buffer(0.004))
    seen = [0]

    def check(what, base):
        seen[0] += 1
        if seen[0] % 20000 == 0:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            prog.update(task, frac=base, detail=f"{what}: {seen[0] // 1000}k read")

    # one read for tagged nodes and ways (a PBF lists all nodes before the ways)
    log("  pass 1/2: tagged nodes and ways (with node locations)")
    fp = osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations()
    fp = fp.with_filter(KeyFilter(*sorted(set(NODE_KEYS) | set(WAY_KEYS))))

    def ways():
        for o in fp:
            if o.is_node():
                check("pass 1/2 nodes", 0.55)
                loc = o.location
                if o.id in data.nodes or not loc.valid() or not inside(loc.lon, loc.lat):
                    continue
                tags = dict(o.tags)
                if _node_wanted(tags):
                    data.nodes[o.id] = [loc.lon, loc.lat, tags]
            elif o.is_way():
                yield o
    for w in ways():
        check("pass 1/2 ways", 0.65)
        tags = dict(w.tags)
        if not _way_wanted(tags):
            continue
        coords = []
        hit = False
        for nd in w.nodes:
            loc = nd.location
            if not loc.valid():
                continue
            coords.append((nd.ref, loc.lon, loc.lat))
            if not hit and inside(loc.lon, loc.lat):
                hit = True
        if not hit and tags.get("natural") not in ("wood", "water") and not tags.get("landuse"):
            continue
        if not hit:
            # large areas (woods, fields) around the corridor: keep when their bbox touches it
            lons = [c[1] for c in coords]
            lats = [c[2] for c in coords]
            if not lons or max(lons) < x0 or min(lons) > x1 or max(lats) < y0 or min(lats) > y1:
                continue
            if area_prep is not None and not area_prep.intersects(box(min(lons), min(lats), max(lons), max(lats))):
                continue
        for ref, lon, lat in coords:
            if ref not in data.nodes:
                data.nodes[ref] = [lon, lat, {}]
        data.ways[w.id] = {"nodes": [c[0] for c in coords], "tags": tags}

    log("  pass 2/2: multipolygon relations")
    rels = {}
    seen[0] = 0
    for r in osmium.FileProcessor(pbf, osmium.osm.RELATION).with_filter(KeyFilter(*REL_KEYS)):
        check("pass 2/2 relations", 0.85)
        tags = dict(r.tags)
        if tags.get("type") not in ("multipolygon", "building"):
            continue
        if not ("building" in tags or tags.get("natural") in ("wood", "scrub", "water")
                or tags.get("landuse")):
            continue
        rels[r.id] = {"members": [(m.type == "w" and "way" or m.type == "n" and "node" or "relation", m.ref, m.role)
                                  for m in r.members], "tags": tags}
    # member ways of all candidate relations; keep a relation only if one of
    # its members has a node inside the bbox
    need = {ref for r in rels.values() for t, ref, _ in r["members"] if t == "way" and ref not in data.ways}
    tmp, hit_ways = {}, set()
    if need:
        fp = osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations()
        fp = fp.with_filter(osmium.filter.EntityFilter(osmium.osm.WAY)).with_filter(IdFilter(need))
        for w in fp:
            coords = [(nd.ref, nd.location.lon, nd.location.lat) for nd in w.nodes if nd.location.valid()]
            tmp[w.id] = (coords, dict(w.tags))
            if any(inside(lon, lat) for _, lon, lat in coords):
                hit_ways.add(w.id)
    for rid, r in rels.items():
        members = [m[1] for m in r["members"] if m[0] == "way"]
        if not any(m in hit_ways or m in data.ways for m in members):
            continue
        for wid in members:
            if wid in tmp and wid not in data.ways:
                coords, tags = tmp[wid]
                for ref, lon, lat in coords:
                    if ref not in data.nodes:
                        data.nodes[ref] = [lon, lat, {}]
                data.ways[wid] = {"nodes": [c[0] for c in coords], "tags": tags}
        data.relations[rid] = r
    return data


def fetch_osm_pbf(frame, route_ll, radius, log, cancel=None, task="osm_pbf", area_ll=None):
    """Corridor data from the Geofabrik extract. Returns None when cancelled
    (Overpass won the race). area_ll (lon/lat polygon) replaces the corridor
    around the route in area mode."""
    from shapely.geometry import LineString
    pad = radius / 111000.0 * 1.6
    bbox = (route_ll[:, 0].min() - pad * 1.5, route_ll[:, 1].min() - pad,
            route_ll[:, 0].max() + pad * 1.5, route_ll[:, 1].max() + pad)
    if area_ll is None:
        # the same corridor Overpass is asked for ("around" the route), in degrees
        import math
        k = math.cos(math.radians(float(route_ll[:, 1].mean())))
        line = LineString(list(zip(route_ll[:, 0] * k, route_ll[:, 1])))
        area = line.buffer(radius / 111000.0)
        from shapely import affinity
        area_ll = affinity.scale(area, xfact=1 / k, yfact=1.0, origin=(0, 0))
    try:
        _, url = pick_region(bbox, log)
        pbf = download(url, log, cancel, task)
        data = extract(pbf, bbox, log, area_ll, cancel, task)
    except Cancelled:
        return None
    data.project(frame)
    log(f"  OSM (Geofabrik): {len(data.nodes)} nodes, {len(data.ways)} ways, {len(data.relations)} relations")
    return data
