# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Road routing between the clicked waypoints, with free public routers.

Every leg (waypoint i -> i+1) is routed with several variants: with and
without a heading hint at each end, and on up to three servers (OSRM demo,
OSRM FOSSGIS, Valhalla FOSSGIS). Each candidate is scored on length plus a
heavy penalty for U-turns and for backtracking (driving down one carriageway
and coming back on the other). The best candidate wins, and the heading at
the end of a leg is kept for the start of the next one, so the car never
reverses at a waypoint. There is no straight-line fallback: if no router
answers, an error is raised.
"""
import json
import math

import numpy as np

from .net import http_get

OSRM_SERVERS = [
    "https://router.project-osrm.org/route/v1/driving/",
    "https://routing.openstreetmap.de/routed-car/route/v1/driving/",
]
VALHALLA = "https://valhalla1.openstreetmap.de/route"


def _bearing(a, b):
    """Compass bearing (deg) from a to b, both [lon, lat]."""
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    dlon = math.radians(b[0] - a[0])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _local_xy(coords, lat0):
    c = np.asarray(coords, float)
    k = 111320.0
    return np.column_stack([(c[:, 0]) * k * math.cos(math.radians(lat0)), c[:, 1] * k])


def analyse(coords):
    """(length m, number of U-turns, backtracked metres) of a lon/lat polyline."""
    if len(coords) < 2:
        return 0.0, 0, 0.0
    P = _local_xy(coords, np.mean(np.asarray(coords)[:, 1]))
    seg = np.hypot(*np.diff(P, axis=0).T)
    L = float(seg.sum())
    if L < 1:
        return L, 0, 0.0
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(L / 5) + 1)
    ss = np.linspace(0, L, n)
    Q = np.column_stack([np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])])
    d = np.diff(Q, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    uturns = 0
    i = 0
    while i < len(h) - 1:
        j = min(len(h), i + 9)  # within ~40 m
        dh = np.abs((h[i + 1:j] - h[i] + np.pi) % (2 * np.pi) - np.pi)
        if len(dh) and dh.max() > math.radians(150):
            uturns += 1
            i = j
        else:
            i += 1
    # backtracking: a point passed again later in the opposite direction
    back = 0.0
    if len(Q) > 20:
        from scipy.spatial import cKDTree
        tree = cKDTree(Q[:-1])
        pairs = tree.query_pairs(12.0, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            far = np.abs(b - a) > 30  # at least 150 m apart along the route
            dh = np.abs((h[np.minimum(a, len(h) - 1)] - h[np.minimum(b, len(h) - 1)] + np.pi) % (2 * np.pi) - np.pi)
            opposite = far & (dh > math.radians(135))
            back = float(len(np.unique(np.concatenate([a[opposite], b[opposite]])))) * 5.0 / 2
    return L, uturns, back


def score(coords):
    L, u, back = analyse(coords)
    return L + 20000.0 * u + 10.0 * back


def _osrm(server, pts, bearings):
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in pts)
    url = server + coords + "?overview=full&geometries=geojson&steps=false&continue_straight=true"
    if bearings:
        url += "&bearings=" + ";".join("" if b is None else f"{int(b) % 360},60" for b in bearings)
    js = json.loads(http_get(url, timeout=60, retries=2))
    if js.get("code") != "Ok" or not js.get("routes"):
        raise RuntimeError(js.get("code"))
    return js["routes"][0]["geometry"]["coordinates"]


def _decode6(s):
    out, idx, lat, lon = [], 0, 0, 0
    while idx < len(s):
        for which in (0, 1):
            shift = result = 0
            while True:
                b = ord(s[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            v = ~(result >> 1) if result & 1 else result >> 1
            if which == 0:
                lat += v
            else:
                lon += v
        out.append([lon / 1e6, lat / 1e6])
    return out


def _valhalla(pts, bearings):
    locs = []
    for i, (lon, lat) in enumerate(pts):
        loc = {"lat": lat, "lon": lon, "type": "break"}
        if bearings and bearings[i] is not None:
            loc["heading"] = int(bearings[i]) % 360
            loc["heading_tolerance"] = 60
        locs.append(loc)
    body = json.dumps({"locations": locs, "costing": "auto", "directions_type": "none"})
    js = json.loads(_post_json(VALHALLA, body))
    coords = []
    for leg in js["trip"]["legs"]:
        c = _decode6(leg["shape"])
        coords += c if not coords else c[1:]
    return coords


def _post_json(url, body):
    import urllib.request
    from .net import UA
    req = urllib.request.Request(url, data=body.encode(), headers={"User-Agent": UA,
                                                                   "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _candidates(a, b, start_heading):
    """Routes for one leg, as (label, coords)."""
    hb = _bearing(a, b)
    variants = [("hint both", [start_heading if start_heading is not None else hb, hb]),
                ("hint start", [start_heading if start_heading is not None else hb, None]),
                ("no hint", None)]
    out, errors = [], []
    for server in OSRM_SERVERS:
        for label, br in variants:
            try:
                out.append((f"osrm {label}", _osrm(server, [a, b], br)))
            except Exception as e:
                errors.append(str(e))
        if out:
            break
    try:
        out.append(("valhalla", _valhalla([a, b], None)))
    except Exception as e:
        errors.append(str(e))
    return out, errors


def route(waypoints_lonlat, log=None):
    """Return (coords, distance m, info dict). Raises if no router can route a leg."""
    w = [list(map(float, p)) for p in waypoints_lonlat]
    if len(w) < 2:
        raise ValueError("need at least 2 waypoints")
    full, heading, legs = [], None, []
    for i in range(len(w) - 1):
        cands, errors = _candidates(w[i], w[i + 1], heading)
        if not cands:
            raise RuntimeError(f"no router could route leg {i + 1}: {errors[:2]}")
        scored = sorted(((score(c), lbl, c) for lbl, c in cands), key=lambda t: t[0])
        best_score, label, coords = scored[0]
        L, u, back = analyse(coords)
        legs.append({"leg": i + 1, "method": label, "length_m": round(L), "uturns": u,
                     "backtrack_m": round(back)})
        if len(coords) >= 2:
            heading = _bearing(coords[-2], coords[-1])
        full += coords if not full else coords[1:]
    L, u, back = analyse(full)
    straight = sum(math.dist(_local_xy([w[k]], w[k][1])[0], _local_xy([w[k + 1]], w[k][1])[0])
                   for k in range(len(w) - 1))
    info = {"legs": legs, "uturns": u, "backtrack_m": round(back),
            "detour_ratio": round(L / max(straight, 1.0), 2)}
    return full, L, info


def clean_route(coords):
    """Remove spikes where a polyline goes out and comes straight back."""
    c = [list(p) for p in coords]
    changed = True
    while changed and len(c) > 3:
        changed = False
        for i in range(1, len(c) - 1):
            a, b, d = np.array(c[i - 1]), np.array(c[i]), np.array(c[i + 1])
            v1, v2 = b - a, d - b
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0 and np.dot(v1, v2) / (n1 * n2) < -0.95:
                del c[i]
                changed = True
                break
    return c
