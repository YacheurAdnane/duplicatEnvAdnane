# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Road network with real junctions, built by CARLA's own OSM converter.

`carla.Osm2Odr` (SUMO netconvert inside) turns the OSM corridor into an
OpenDRIVE network with a junction at every intersection, exit, slip road and
roundabout entry. This module then

1. splits every road longer than `max_road_len` (1 km by default) at a
   geometry boundary and rewires all links and junction references,
2. adds real heights: the converter output is flat, so each road gets an
   elevation profile from the terrain/road height model, with the heights
   matched at every junction so there is no step between roads,
3. samples the final network lane by lane (CARLA's own map code), which feeds
   the road mesh, the terrain, the guard rails and the Lanelet2 map. Every
   mesh therefore matches the OpenDRIVE exactly, and junctions are one merged
   surface instead of overlapping road pieces.
"""
import copy
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from .geo import dp_simplify, gaussian_smooth

WAY_TYPES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
             "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
             "residential", "living_street", "service", "road", "busway"]


def carla_available():
    try:
        import carla  # noqa: F401
        return hasattr(carla, "Osm2Odr")
    except Exception:
        return False


# ---------------------------------------------------------------- conversion

def convert(osm_xml, proj_string, traffic_lights=True, log=print):
    """Run CARLA's converter in a child process: its traffic-light generator
    segfaults on large networks (seen on a 56 km route), which would otherwise
    kill the whole job. On a crash it retries without traffic lights."""
    import os
    import subprocess
    import sys
    import tempfile
    tmp = tempfile.mkdtemp(prefix="dupenv_o2o_")
    src, dst = os.path.join(tmp, "in.osm"), os.path.join(tmp, "out.xodr")
    with open(src, "w", encoding="utf-8") as f:
        f.write(osm_xml)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        for tl in ([True, False] if traffic_lights else [False]):
            code = (f"import sys; sys.path.insert(0, {here!r}); from dupenv.network import _convert_inproc; "
                    f"open({dst!r}, 'w').write(_convert_inproc(open({src!r}).read(), {proj_string!r}, {tl}))")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
            if r.returncode == 0 and os.path.exists(dst):
                with open(dst, encoding="utf-8") as f:
                    return f.read()
            log(f"  CARLA converter crashed (exit {r.returncode})"
                + (", retrying without traffic lights" if tl else ""))
        raise RuntimeError("CARLA OSM converter failed")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _convert_inproc(osm_xml, proj_string, traffic_lights):
    import carla
    s = carla.Osm2OdrSettings()
    s.proj_string = proj_string
    s.center_map = False
    s.use_offsets = False
    s.generate_traffic_lights = traffic_lights
    s.all_junctions_with_traffic_lights = False
    s.set_osm_way_types(WAY_TYPES)
    return carla.Osm2Odr.convert(osm_xml, s)


# ---------------------------------------------------------------- polynomial records

def _shift_poly(a, b, c, d, t):
    """Coefficients of p(t + u) as a polynomial in u."""
    return (a + b * t + c * t * t + d * t ** 3, b + 2 * c * t + 3 * d * t * t, c + 3 * d * t, d)


def _cut_records(elems, key, start, end, poly=True):
    """Re-base a list of s-indexed records (elevation, width...) to [start, end)."""
    recs = sorted(elems, key=lambda e: float(e.get(key)))
    out = []
    first = None
    for e in recs:
        s0 = float(e.get(key))
        if s0 <= start + 1e-9:
            first = e
    if first is not None:
        e = copy.deepcopy(first)
        t = start - float(first.get(key))
        if poly and t > 0:
            a, b, c, d = (float(first.get(k, 0)) for k in "abcd")
            a, b, c, d = _shift_poly(a, b, c, d, t)
            for k, v in zip("abcd", (a, b, c, d)):
                e.set(k, f"{v:.10g}")
        e.set(key, "0")
        out.append(e)
    for e in recs:
        s0 = float(e.get(key))
        if start + 1e-9 < s0 < end - 1e-9:
            e2 = copy.deepcopy(e)
            e2.set(key, f"{s0 - start:.10g}")
            out.append(e2)
    return out


# ---------------------------------------------------------------- road splitting

def split_long_roads(xodr, max_len=1000.0):
    root = ET.fromstring(xodr)
    roads = root.findall("road")
    next_id = max(int(r.get("id")) for r in roads) + 1
    renamed_end = {}  # old road id -> id of the piece that now holds its end
    internal = set()  # links between pieces of one split road: never redirected below
    n_split = 0
    for road in list(roads):
        if road.get("junction", "-1") != "-1":
            continue
        L = float(road.get("length"))
        if L <= max_len * 1.05:
            continue
        pv = road.find("planView")
        geoms = pv.findall("geometry")
        starts = [float(g.get("s")) for g in geoms]
        # greedy: furthest geometry start within max_len of the previous cut
        cuts, last = [], 0.0
        while L - last > max_len * 1.05:
            cand = [c for c in starts if last + 50 < c <= last + max_len]
            if not cand:
                cand = [c for c in starts if c > last + 50][:1]
            if not cand or L - cand[-1] < 1.0:
                break
            cuts.append(cand[-1])
            last = cand[-1]
        cuts = sorted(set(c for c in cuts if 1.0 < c < L - 1.0))
        if not cuts:
            continue
        bounds = [0.0] + cuts + [L]
        rid = road.get("id")
        pieces = []
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            new = copy.deepcopy(road)
            if k > 0:
                new.set("id", str(next_id))
                next_id += 1
            new.set("length", f"{b - a:.8f}")
            # planView
            npv = new.find("planView")
            for g in list(npv):
                npv.remove(g)
            for g in geoms:
                s0 = float(g.get("s"))
                if a - 1e-6 <= s0 < b - 1e-6:
                    g2 = copy.deepcopy(g)
                    g2.set("s", f"{s0 - a:.8f}")
                    npv.append(g2)
            # s-indexed records on the road
            for parent_tag, tag in (("elevationProfile", "elevation"), ("lateralProfile", "superelevation"),
                                    ("lanes", "laneOffset")):
                par = new.find(parent_tag)
                if par is None:
                    continue
                recs = [e for e in par.findall(tag)]
                for e in recs:
                    par.remove(e)
                for e in _cut_records(recs, "s", a, b):
                    par.insert(0 if tag == "laneOffset" else len(par), e)
            for tp in new.findall("type"):
                new.remove(tp)
            types = _cut_records(road.findall("type"), "s", a, b, poly=False)
            for i, tp in enumerate(types):
                new.insert(1 + i, tp)
            # lane sections: keep those overlapping [a, b), re-based
            lanes = new.find("lanes")
            secs = road.find("lanes").findall("laneSection")
            for sec in lanes.findall("laneSection"):
                lanes.remove(sec)
            sec_s = [float(sc.get("s")) for sc in secs]
            for i, sc in enumerate(secs):
                s_start = sec_s[i]
                s_end = sec_s[i + 1] if i + 1 < len(secs) else L
                if s_end <= a + 1e-6 or s_start >= b - 1e-6:
                    continue
                sc2 = copy.deepcopy(sc)
                off = max(0.0, a - s_start)
                sc2.set("s", f"{max(s_start, a) - a:.8f}")
                for ln in sc2.iter("lane"):
                    for tag in ("width", "border"):
                        recs = ln.findall(tag)
                        for e in recs:
                            ln.remove(e)
                        for j, e in enumerate(_cut_records(recs, "sOffset", off, off + (b - a))):
                            ln.insert(1 + j, e)
                    for tag in ("roadMark", "speed", "material", "height", "access"):
                        recs = ln.findall(tag)
                        if not recs:
                            continue
                        for e in recs:
                            ln.remove(e)
                        for e in _cut_records(recs, "sOffset", off, off + (b - a), poly=False):
                            ln.append(e)
                lanes.append(sc2)
            # signals / objects inside the piece
            for grp, tag in (("signals", "signal"), ("objects", "object")):
                par = new.find(grp)
                if par is None:
                    continue
                for e in list(par.findall(tag)):
                    s0 = float(e.get("s", 0))
                    if a - 1e-6 <= s0 < b - 1e-6:
                        e.set("s", f"{s0 - a:.8f}")
                    else:
                        par.remove(e)
            pieces.append(new)
        # links between the pieces
        for k, p in enumerate(pieces):
            link = p.find("link")
            if link is None:
                link = ET.SubElement(p, "link")
            if k > 0:
                for e in link.findall("predecessor"):
                    link.remove(e)
                internal.add(ET.SubElement(link, "predecessor", elementType="road",
                                           elementId=pieces[k - 1].get("id"), contactPoint="end"))
            if k < len(pieces) - 1:
                for e in link.findall("successor"):
                    link.remove(e)
                ET.SubElement(link, "successor", elementType="road",
                              elementId=pieces[k + 1].get("id"), contactPoint="start")
            # lane links across the cut: same lane ids on both sides
            for sec_i, sc in enumerate(p.find("lanes").findall("laneSection")):
                last_sec = sec_i == len(p.find("lanes").findall("laneSection")) - 1
                first_sec = sec_i == 0
                for ln in sc.iter("lane"):
                    lid = ln.get("id")
                    if lid == "0":
                        continue
                    lk = ln.find("link")
                    if lk is None:
                        lk = ET.Element("link")
                        ln.insert(0, lk)
                    if k > 0 and first_sec:
                        for e in lk.findall("predecessor"):
                            lk.remove(e)
                        ET.SubElement(lk, "predecessor", id=lid)
                    if k < len(pieces) - 1 and last_sec:
                        for e in lk.findall("successor"):
                            lk.remove(e)
                        ET.SubElement(lk, "successor", id=lid)
        idx = list(root).index(road)
        root.remove(road)
        for k, p in enumerate(pieces):
            root.insert(idx + k, p)
        renamed_end[rid] = pieces[-1].get("id")
        n_split += 1
    if renamed_end:
        # whatever pointed at the END of a split road now points at its last piece
        for road in root.findall("road"):
            link = road.find("link")
            if link is None:
                continue
            for e in list(link):
                if e in internal:
                    continue
                if e.get("elementType") == "road" and e.get("elementId") in renamed_end \
                        and e.get("contactPoint") == "end":
                    e.set("elementId", renamed_end[e.get("elementId")])
        for junc in root.findall("junction"):
            for con in junc.findall("connection"):
                inc = con.get("incomingRoad")
                if inc in renamed_end:
                    conn = root.find(f"road[@id='{con.get('connectingRoad')}']")
                    at_end = False
                    if conn is not None:
                        lk = conn.find("link")
                        for e in (lk if lk is not None else []):
                            if e.get("elementId") in (inc, renamed_end[inc]) and e.get("elementType") == "road":
                                at_end = e.get("elementId") == renamed_end[inc]
                    if at_end:
                        con.set("incomingRoad", renamed_end[inc])
    return ET.tostring(root, encoding="unicode"), n_split


# ---------------------------------------------------------------- elevation

def _sample_roads(cmap, step=2.0, xodr=None):
    """{road_id: (s, x, y, heading)} along the innermost lane of every road.

    Every road is sampled, whatever its lanes: CARLA's generate_waypoints()
    only returns driving lanes, and roads with only service ("restricted") or
    sidewalk lanes kept height 0 (they showed as roads under the ground in
    RoadRunner and CARLA)."""
    res = {}
    root = ET.fromstring(xodr) if xodr else None
    if root is None:
        return res
    for road in root.findall("road"):
        rid = int(road.get("id"))
        L = float(road.get("length"))
        secs = road.find("lanes").findall("laneSection")
        ss = [float(sc.get("s")) for sc in secs] + [L]
        pts = []
        for k, sc in enumerate(secs):
            ids = [int(ln.get("id")) for ln in sc.iter("lane") if ln.get("id") != "0"]
            if not ids:
                continue
            lid = min(ids, key=lambda v: (abs(v), v > 0))
            s0, s1 = ss[k], ss[k + 1]
            n = max(2, int(math.ceil((s1 - s0) / step)) + 1)
            for s in np.linspace(s0 + 0.01, s1 - 0.01, n) if s1 - s0 > 0.03 else [0.5 * (s0 + s1)]:
                try:
                    w = cmap.get_waypoint_xodr(rid, lid, float(s))
                except Exception:
                    w = None
                if w is None:
                    continue
                loc = w.transform.location
                yaw = -math.radians(w.transform.rotation.yaw)
                if lid > 0:
                    yaw += math.pi
                pts.append((float(s), loc.x, -loc.y, yaw))
        if pts:
            a = np.array(sorted(pts))
            _, u = np.unique(np.round(a[:, 0], 3), return_index=True)
            res[rid] = a[u]
    return res


def add_elevation(xodr, models, ground, log, sigma=5.0):
    """Fill every road's elevationProfile from the road models' heights."""
    import carla
    cmap = carla.Map("net", xodr)
    samples = _sample_roads(cmap, xodr=xodr)
    # height field: our OSM road models (they carry bridge / tunnel heights)
    P, Z, D = [], [], []
    for m in models:
        if m.kind not in ("route", "road", "track"):
            continue
        P.append(np.column_stack([m.x, m.y]))
        Z.append(m.z)
        D.append(m.T)
    P, Z, D = np.concatenate(P), np.concatenate(Z), np.concatenate(D)
    tree = cKDTree(P)
    zs = {}
    for rid, a in samples.items():
        s, x, y, h = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        t = np.column_stack([np.cos(h), np.sin(h)])
        dist, idx = tree.query(np.column_stack([x, y]), k=12, distance_upper_bound=14.0)
        z = np.full(len(s), np.nan)
        prev = None
        for i in range(len(s)):
            best, bz = None, None
            for d, j in zip(dist[i], idx[i]):
                if not np.isfinite(d):
                    break
                align = abs(D[j] @ t[i])
                cost = d + (0 if align > 0.6 else 25.0)
                if prev is not None:
                    cost += 2.0 * abs(Z[j] - prev)  # stay on the same level (bridges)
                if best is None or cost < best:
                    best, bz = cost, Z[j]
            if bz is None:
                bz = float(ground.height(np.array([x[i]]), np.array([y[i]]))[0])
            z[i] = bz
            prev = bz
        zs[rid] = [s, x, y, gaussian_smooth(z, sigma / 2.0) if len(z) > 3 else z]
    # match heights where roads meet: the ends the xodr links together (road
    # to road, and junction connectors to their roads) get their mean height
    ends = []
    end_of = {}
    for rid, (s, x, y, z) in zs.items():
        end_of[(rid, 0)] = len(ends)
        ends.append((rid, 0, x[0], y[0], z[0]))
        end_of[(rid, 1)] = len(ends)
        ends.append((rid, 1, x[-1], y[-1], z[-1]))
    parent = list(range(len(ends)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for road in ET.fromstring(xodr).findall("road"):
        rid = int(road.get("id"))
        lk = road.find("link")
        if lk is None or (rid, 0) not in end_of:
            continue
        for tag, which in (("predecessor", 0), ("successor", 1)):
            e = lk.find(tag)
            if e is None or e.get("elementType") != "road":
                continue
            oid = int(e.get("elementId"))
            ow = 0 if e.get("contactPoint", "start") == "start" else 1
            if (oid, ow) in end_of:
                i, j = end_of[(rid, which)], end_of[(oid, ow)]
                if abs(ends[i][4] - ends[j][4]) < 3.0:  # never glue a bridge to the road under it
                    parent[find(i)] = find(j)
    groups = {}
    for i in range(len(ends)):
        groups.setdefault(find(i), []).append(i)
    target = np.array([e[4] for e in ends])
    for members in groups.values():
        target[members] = np.mean([ends[i][4] for i in members])
    for i, (rid, which, _, _, z0) in enumerate(ends):
        s, x, y, z = zs[rid]
        L = max(s[-1] - s[0], 1e-6)
        blend = min(15.0, L / 2)
        dz = target[i] - z0
        if which == 0:
            w = np.clip(1 - (s - s[0]) / blend, 0, 1)
        else:
            w = np.clip(1 - (s[-1] - s) / blend, 0, 1)
        zs[rid][3] = z + dz * w
    # write the profiles
    root = ET.fromstring(xodr)
    for road in root.findall("road"):
        rid = int(road.get("id"))
        if rid not in zs:
            continue
        s, x, y, z = zs[rid]
        prof = road.find("elevationProfile")
        if prof is None:
            prof = ET.Element("elevationProfile")
            road.insert(list(road).index(road.find("planView")) + 1, prof)
        for e in list(prof):
            prof.remove(e)
        L = float(road.get("length"))
        s = np.clip(s, 0, L)
        # samples fall up to one step short of the road ends; pin the ends to the
        # matched heights so the profile doesn't extrapolate a slope past them
        # (that left steps of up to 20 cm where a long road was split, and
        # RoadRunner refuses to export "not aligned" roads)
        if s[0] > 1e-3:
            s, z = np.concatenate([[0.0], s]), np.concatenate([[z[0]], z])
        if s[-1] < L - 1e-3:
            s, z = np.concatenate([s, [L]]), np.concatenate([z, [z[-1]]])
        keep = dp_simplify(np.column_stack([s, z]), 0.01) if len(s) > 2 else np.arange(len(s))
        for i, j in zip(keep[:-1], keep[1:]):
            ds = s[j] - s[i]
            b = (z[j] - z[i]) / ds if ds > 1e-6 else 0.0
            ET.SubElement(prof, "elevation", s=f"{s[i]:.4f}", a=f"{z[i]:.4f}", b=f"{b:.7f}", c="0", d="0")
        if len(keep) < 2:
            ET.SubElement(prof, "elevation", s="0", a=f"{z[0]:.4f}", b="0", c="0", d="0")
    # signals carry an absolute position too (z = 0 from the converter), and
    # RoadRunner uses it: traffic lights were placed at height 0
    n_sig = 0
    for road in root.findall("road"):
        rid = int(road.get("id"))
        sg = road.find("signals")
        if sg is None or rid not in zs:
            continue
        s_, x_, y_, z_ = zs[rid]
        for sig in sg.findall("signal"):
            pi = sig.find("positionInertial")
            if pi is None:
                continue
            k = int(np.argmin((x_ - float(pi.get("x"))) ** 2 + (y_ - float(pi.get("y"))) ** 2))
            pi.set("z", f"{float(z_[k]) + float(sig.get('zOffset', 0) or 0):.3f}")
            n_sig += 1
    log(f"  elevation added to {len(zs)} roads, heights matched at {len(ends)} road ends"
        + (f", {n_sig} signals raised to their road" if n_sig else ""))
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------- lane sampling

class Lane:
    __slots__ = ("road", "section", "lane", "junction", "type", "pts", "width", "left", "right",
                 "speed", "marks_r", "marks_l", "travel", "key", "succ", "s")


def _sections(xodr):
    """{road id: [(s_start, s_end), ...]} for every lane section."""
    root = ET.fromstring(xodr)
    out = {}
    for road in root.findall("road"):
        L = float(road.get("length"))
        ss = [float(sc.get("s")) for sc in road.find("lanes").findall("laneSection")]
        out[int(road.get("id"))] = [(ss[i], ss[i + 1] if i + 1 < len(ss) else L) for i in range(len(ss))]
    return out


def sample_lanes(cmap, xodr, step=1.0):
    """Every lane of the network as an ordered list of samples (travel direction).

    Uses generate_waypoints() plus exact section start/end points from
    get_waypoint_xodr(); CARLA's previous_until_lane_start() crashes on some
    junction lanes, so it is not used. Each Lane has pts (n,3) in the local
    ENU frame, width (n,), left/right edges, lane type and markings."""
    import carla
    sections = _sections(xodr)
    groups = {}
    for w in cmap.generate_waypoints(step):
        groups.setdefault((w.road_id, w.section_id, w.lane_id), []).append(w)

    def complete(key, wps):
        road, sec, lane = key
        spans = sections.get(road)
        if spans and sec < len(spans):
            s0, s1 = spans[sec]
            for sv in (s0 + 0.01, s1 - 0.01):
                try:
                    e = cmap.get_waypoint_xodr(road, lane, sv)
                except Exception:
                    e = None
                if e is not None and (e.road_id, e.section_id, e.lane_id) == key:
                    wps.append(e)
        wps.sort(key=lambda q: q.s)
        out = [wps[0]]
        for q in wps[1:]:
            if q.s - out[-1].s > 0.05:
                out.append(q)
        if lane > 0:  # positive lanes drive against s
            out.reverse()
        return out

    def make(seq):
        ln = Lane()
        w0 = seq[0]
        ln.road, ln.section, ln.lane = w0.road_id, w0.section_id, w0.lane_id
        ln.junction = w0.junction_id if w0.is_junction else -1
        ln.type = str(w0.lane_type)
        P = np.array([[q.transform.location.x, -q.transform.location.y, q.transform.location.z] for q in seq])
        W = np.array([q.lane_width for q in seq])
        yaw = -np.radians([q.transform.rotation.yaw for q in seq])
        n = np.column_stack([-np.sin(yaw), np.cos(yaw)])
        ln.pts, ln.width = P, W
        ln.left = np.column_stack([P[:, :2] + n * W[:, None] / 2, P[:, 2]])
        ln.right = np.column_stack([P[:, :2] - n * W[:, None] / 2, P[:, 2]])
        ln.marks_r = [str(q.right_lane_marking.type) for q in seq]
        ln.marks_l = [str(q.left_lane_marking.type) for q in seq]
        ln.travel = True
        ln.key = (ln.road, ln.section, ln.lane)
        ln.succ = []
        ln.s = np.array([q.s for q in seq])
        if w0.lane_type == carla.LaneType.Driving:
            try:  # lanes that follow this one, straight from the xodr links
                ln.succ = list({(q.road_id, q.section_id, q.lane_id)
                                for q in seq[-1].next(0.5) if q.lane_type == carla.LaneType.Driving}
                               - {ln.key})
            except Exception:
                pass
        return ln

    lanes = []
    seen = set()
    for key, wps in groups.items():
        seq = complete(key, list(wps))
        if len(seq) < 2:
            continue
        seen.add(key)
        lanes.append(make(seq))
        # non-driving lanes outside (shoulders, sidewalks, borders), sampled at the same s
        for getter in ("get_right_lane", "get_left_lane"):
            cur = seq
            for _ in range(4):
                nxt = [getattr(q, getter)() for q in cur]
                if any(q is None for q in nxt):
                    break
                if nxt[0].lane_type == carla.LaneType.Driving or nxt[0].lane_id * seq[0].lane_id < 0:
                    break
                k2 = (nxt[0].road_id, nxt[0].section_id, nxt[0].lane_id)
                if k2 not in seen and all((q.road_id, q.section_id, q.lane_id) == k2 for q in nxt):
                    seen.add(k2)
                    lanes.append(make(nxt))
                cur = nxt
    # roads without a driving lane (service roads the converter marks
    # "restricted", sidewalk-only pieces): generate_waypoints() skips them, so
    # they are walked lane by lane here, or they would have no surface at all
    root = ET.fromstring(xodr)
    for road in root.findall("road"):
        rid = int(road.get("id"))
        for sec_i, (s0, s1) in enumerate(sections.get(rid, [])):
            secs = road.find("lanes").findall("laneSection")
            if sec_i >= len(secs):
                continue
            for ln in secs[sec_i].iter("lane"):
                lid = int(ln.get("id"))
                if lid == 0 or (rid, sec_i, lid) in seen:
                    continue
                n = max(2, int(math.ceil((s1 - s0) / step)) + 1)
                seq = []
                for s in np.linspace(s0 + 0.01, s1 - 0.01, n):
                    try:
                        w = cmap.get_waypoint_xodr(rid, lid, float(s))
                    except Exception:
                        w = None
                    if w is not None and (w.road_id, w.section_id, w.lane_id) == (rid, sec_i, lid):
                        if not seq or w.s - seq[-1].s > 0.05:
                            seq.append(w)
                if len(seq) < 2:
                    continue
                if lid > 0:
                    seq.reverse()
                seen.add((rid, sec_i, lid))
                lanes.append(make(seq))
    return lanes


# ---------------------------------------------------------------- lanelet2

def _no_foldback(E, d):
    """Indices of the points of a lane edge E to keep, dropping those that step
    backwards along the unit travel directions d. The inner edge of a tight
    junction curve folds over itself (so can a corner snapped to the next
    lane), and Lanelet2 then flips the whole bound and cuts the lanelet from
    the routing graph. The two end points are always kept."""
    if len(E) <= 2:
        return np.arange(len(E))
    keep = [0]
    for j in range(1, len(E) - 1):
        if np.dot(E[j, :2] - E[keep[-1], :2], d[j]) > 0.05 and np.dot(E[-1, :2] - E[j, :2], d[j]) > 0.05:
            keep.append(j)
    keep.append(len(E) - 1)
    return np.array(keep)


def write_lanelet2(path, lanes, frame, speed_of=None, seg_len=50.0, merge_tol=0.15, link_tol=4.0, min_len=3.0):
    """Lanelets for every driving lane (junction lanes included).

    Neighbouring lanes of one road section that drive the same way share their
    boundary line string, cut at common s positions, so Lanelet2 sees them as
    neighbours and allows lane changes over dashed lines. Lane ends that meet
    (within merge_tol) share one node. A lane end is also joined to the start
    of every lane the xodr links it to (up to link_tol apart), because the
    converter's road geometry can kink at road ends and leave the corners a
    metre or more apart."""
    drive = [ln for ln in lanes if "Driving" in ln.type and len(ln.pts) >= 2]
    # stub lanes under min_len (tiny junction connectors, sliver lane sections)
    # often point the wrong way; skip them and link around them
    by_key = {getattr(ln, "key", None): ln for ln in drive}
    stub = {k for k, ln in by_key.items()
            if k is not None and np.hypot(*np.diff(ln.pts[:, :2], axis=0).T).sum() < min_len}

    def through(k, depth=0):
        if k not in stub or depth > 4:
            return [k]
        return [q for s2 in (by_key[k].succ or ()) for q in through(s2, depth + 1)]
    succ_of = {k: list(dict.fromkeys(q for s2 in (ln.succ or ()) for q in through(s2) if q not in stub))
               for k, ln in by_key.items() if k is not None}
    drive = [ln for ln in drive if getattr(ln, "key", None) not in stub]
    groups = {}
    for ln in drive:
        groups.setdefault((ln.road, ln.section), []).append(ln)
    ways = []       # [points (n,3), marking type, junction]
    way_key = {}    # shared boundary key -> way index
    chunks = []     # (left way, right way, lane)
    first_end, last_end = {}, {}
    for (road, sec), grp in groups.items():
        s_lo = min(float(ln.s.min()) for ln in grp if ln.s is not None)
        s_hi = max(float(ln.s.max()) for ln in grp if ln.s is not None)
        n_cut = max(1, int(round((s_hi - s_lo) / seg_len)))
        cut_s = np.linspace(s_lo, s_hi, n_cut + 1)
        # inner lanes first so a shared boundary takes the inner lane's right edge
        for ln in sorted(grp, key=lambda q: abs(q.lane)):
            s = ln.s
            order = np.argsort(s)  # sample indices in increasing s
            ci = sorted({int(order[np.argmin(np.abs(s[order] - c))]) for c in cut_s[1:-1]} | {0, len(s) - 1})
            # s-interval ids of each chunk, independent of travel direction
            cid = [int(np.argmin(np.abs(cut_s - s[i]))) for i in ci]
            sign = 1 if ln.lane > 0 else -1
            k = abs(ln.lane)
            n0 = len(chunks)
            for (a, b), (ca, cb) in zip(zip(ci[:-1], ci[1:]), zip(cid[:-1], cid[1:])):
                if b <= a:
                    continue
                span = (min(ca, cb), max(ca, cb))
                bw = []
                for side, P, mk in ((0, ln.left, ln.marks_l), (1, ln.right, ln.marks_r)):
                    B = sign * (k - 1 + side)  # boundary index across the road, 0 = centre line
                    key = (road, sec, B, span) if B != 0 else None
                    if key is not None and key in way_key:
                        bw.append(way_key[key])
                        continue
                    d = np.gradient(ln.pts[a:b + 1, :2], axis=0) if b - a > 1 else np.diff(ln.pts[a:b + 1, :2], axis=0)
                    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                    ways.append([P[a:b + 1], mk[(a + b) // 2], ln.junction, d])
                    if key is not None:
                        way_key[key] = len(ways) - 1
                    bw.append(len(ways) - 1)
                chunks.append((bw[0], bw[1], ln))
            if len(chunks) > n0:
                first_end[ln.key], last_end[ln.key] = n0, len(chunks) - 1
    # way end points: 2w = start, 2w+1 = end. Ends closer than merge_tol become one node
    ends = np.array([w[0][i] for w in ways for i in (0, -1)]) if ways else np.zeros((0, 3))
    parent = np.arange(len(ends))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if len(ends):
        for i, j in cKDTree(ends[:, :2]).query_pairs(merge_tol):
            if abs(ends[i, 2] - ends[j, 2]) < 0.5:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
        for key, c in last_end.items():
            lw, rw, ln = chunks[c]
            for sk in succ_of.get(key, ()):
                d = first_end.get(sk)
                if d is None:
                    continue
                for a, b in ((2 * lw + 1, 2 * chunks[d][0]), (2 * rw + 1, 2 * chunks[d][1])):
                    if np.linalg.norm(ends[a, :2] - ends[b, :2]) < link_tol and abs(ends[a, 2] - ends[b, 2]) < 1.0:
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[ra] = rb  # the successor's corner wins
    ids = [0]
    nodes = []
    root_node = {}

    def nid():
        ids[0] += 1
        return ids[0]

    def end_node(e):
        r = find(e)
        if r not in root_node:
            i = nid()
            root_node[r] = i
            nodes.append((i, *map(float, ends[r])))
        return root_node[r]

    way_ids, out_ways = [], []
    for w, (P, mk, junction, d) in enumerate(ways):
        # corners sit where the merged nodes are; drop points that now fold back
        P = P.copy()
        P[0], P[-1] = ends[find(2 * w)], ends[find(2 * w + 1)]
        sel = _no_foldback(P, d)
        P = P[sel]
        keep = dp_simplify(P[:, :2], 0.02)
        nds = []
        for j in keep:
            if j == 0:
                nds.append(end_node(2 * w))
            elif j == len(P) - 1:
                nds.append(end_node(2 * w + 1))
            else:
                i = nid()
                nodes.append((i, float(P[j, 0]), float(P[j, 1]), float(P[j, 2])))
                nds.append(i)
        if junction >= 0:
            tags = {"type": "virtual", "subtype": "virtual", "lane_change": "no"}
        else:
            sub = "dashed" if "Broken" in mk else "solid"
            tags = {"type": "line_thin", "subtype": sub, "lane_change": "yes" if sub == "dashed" else "no"}
        wid = nid()
        way_ids.append(wid)
        out_ways.append((wid, nds, tags))
    rels = []
    for lw, rw, ln in chunks:
        spd = speed_of(ln) if speed_of else 50
        rels.append((nid(), way_ids[lw], way_ids[rw], {"type": "lanelet", "subtype": "road", "location": "urban",
                                                       "one_way": "yes", "participant:vehicle": "yes",
                                                       "speed_limit": str(int(round(spd)))}))
    arr = np.array([(x, y) for _, x, y, _ in nodes]) if nodes else np.zeros((0, 2))
    lon, lat = frame.to_lonlat(arr[:, 0], arr[:, 1]) if len(arr) else ([], [])
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="duplicat_env">\n')
        f.write('  <MetaInfo format_version="1" map_version="1"/>\n')
        for (i, x, y, z), lo, la in zip(nodes, lon, lat):
            f.write(f'  <node id="{i}" lat="{la:.11f}" lon="{lo:.11f}">\n'
                    f'    <tag k="local_x" v="{x:.4f}"/>\n    <tag k="local_y" v="{y:.4f}"/>\n'
                    f'    <tag k="ele" v="{z:.4f}"/>\n  </node>\n')
        for i, nds, tags in out_ways:
            f.write(f'  <way id="{i}">\n' + "".join(f'    <nd ref="{n}"/>\n' for n in nds))
            f.write("".join(f'    <tag k="{k2}" v="{v}"/>\n' for k2, v in tags.items()) + "  </way>\n")
        for i, lw, rw, tags in rels:
            f.write(f'  <relation id="{i}">\n    <member type="way" role="left" ref="{lw}"/>\n'
                    f'    <member type="way" role="right" ref="{rw}"/>\n')
            f.write("".join(f'    <tag k="{k2}" v="{v}"/>\n' for k2, v in tags.items()) + "  </relation>\n")
        f.write("</osm>\n")
    return len(rels)


# ---------------------------------------------------------------- meshes from the network

ROAD_TYPES = ("Driving", "Shoulder", "Border", "Parking", "Biking", "Bidirectional", "Restricted",
              "Stop", "Entry", "Exit", "OnRamp", "OffRamp")


def road_speeds(xodr):
    """{road id: max lane speed km/h} from the OpenDRIVE lane <speed> records."""
    root = ET.fromstring(xodr)
    out = {}
    for road in root.findall("road"):
        vals = [float(s.get("max")) for s in road.iter("speed") if s.get("max")]
        unit = "m/s"
        for s in road.iter("speed"):
            unit = s.get("unit", "m/s")
            break
        if vals:
            v = max(vals)
            out[int(road.get("id"))] = v * 3.6 if unit in ("m/s", "") else v
    return out


def _dash_intervals(L, dash, gap):
    starts = np.arange(gap / 2, max(L - dash, 0), dash + gap)
    return [(a, a + dash) for a in starts]


def network_meshes(scene, lanes, elev, cfg, log):
    from . import meshlib as ml

    dash, gap = cfg.get("dash", 3.0), cfg.get("gap", 10.0)
    n_strips = 0
    walk_quads = []
    road_quads, road_high = [], []
    for ln in lanes:
        if len(ln.pts) < 2:
            continue
        is_walk = "Sidewalk" in ln.type
        if not is_walk and not any(t in ln.type for t in ROAD_TYPES):
            continue
        if is_walk:
            # pavements are merged into one surface later (they overlap at bends)
            for i in range(len(ln.pts) - 1):
                walk_quads.append([ln.left[i], ln.left[i + 1], ln.right[i + 1], ln.right[i]])
            continue
        if ln.junction >= 0:
            for i in range(len(ln.pts) - 1):
                road_quads.append([ln.left[i], ln.left[i + 1], ln.right[i + 1], ln.right[i]])
                road_high.append(False)
            continue
        dz = 0.12 if is_walk else 0.0
        A = ln.left + [0, 0, dz]
        B = ln.right + [0, 0, dz]
        # bridges: underside and piers where the lane runs well above the ground
        ground_z = elev.sample_xy(ln.pts[:, 0], ln.pts[:, 1])
        high = (ln.pts[:, 2] - ground_z) > 3.0
        from .geo import runs as _runs
        for a0, b0 in _runs(high):
            if b0 - a0 < 2:
                continue
            Au, Bu = A[a0:b0] - [0, 0, 1.2], B[a0:b0] - [0, 0, 1.2]
            V, F, UV = ml.ribbon(Bu, Au)
            scene.add("Bridge", "M_Concrete", V, F, UV)
            if abs(ln.lane) == 1:
                seg = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ln.pts[a0:b0, :2], axis=0).T))])
                for sv in np.arange(15.0, seg[-1] - 5, 30.0):
                    i = a0 + int(np.searchsorted(seg, sv))
                    h = ln.pts[i, 2] - 1.2 - ground_z[i]
                    if h > 1.5:
                        V, F, UV = ml.box(1.2, 1.2, h + 1.0)
                        scene.add("Bridge", "M_Concrete", V + [ln.pts[i, 0], ln.pts[i, 1], ground_z[i] - 1.0], F, UV)
        # asphalt is merged per level after the loop (lanes of different roads touch at bends)
        for i in range(len(A) - 1):
            road_quads.append([A[i], A[i + 1], B[i + 1], B[i]])
            road_high.append(bool(high[i] and high[i + 1]))
        n_strips += 1
        # lane markings along the lane edges (types come from the OpenDRIVE roadMarks)
        if "Driving" in ln.type and cfg.get("markings", True):
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ln.pts[:, :2], axis=0).T))])
            edges = [(ln.right, ln.marks_r)]
            if abs(ln.lane) == 1:
                edges.append((ln.left, ln.marks_l))
            for E, marks in edges:
                kinds = np.array(marks)
                solid = np.array(["Solid" in k for k in kinds])
                broken = np.array(["Broken" in k for k in kinds])
                d = np.gradient(E[:, :2], axis=0)
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                nrm = np.column_stack([-d[:, 1], d[:, 0]])
                w = 0.15

                def strip(idx):
                    P = E[idx]
                    Nn = nrm[idx]
                    L2 = np.column_stack([P[:, :2] + Nn * w / 2, P[:, 2] + 0.02])
                    R2 = np.column_stack([P[:, :2] - Nn * w / 2, P[:, 2] + 0.02])
                    if len(P) >= 2:
                        scene.add("Road_Marking", "M_Marking_White", *ml.ribbon(L2, R2))

                from .geo import runs
                for a, b in runs(solid):
                    if b - a >= 2:
                        strip(np.arange(a, b))
                for a, b in runs(broken):
                    if b - a < 2:
                        continue
                    for s0, s1 in _dash_intervals(s[b - 1] - s[a], dash, gap):
                        i0 = a + int(np.searchsorted(s[a:b] - s[a], s0))
                        i1 = a + int(np.searchsorted(s[a:b] - s[a], s1))
                        if i1 > i0:
                            strip(np.arange(i0, min(i1 + 1, b)))
    # all asphalt (lanes and junctions) merged per level: one layer, nothing overlaps
    RQ, RH = np.array(road_quads), np.array(road_high, bool)
    nj, road_union = 0, {}
    if len(RQ):
        n1, road_union = merged_surface(scene, RQ[~RH], "Road_Road", "M_Asphalt", curb=0.3)
        n2, _ = merged_surface(scene, RQ[RH], "Road_Road", "M_Asphalt", curb=0.3)
        nj = n1 + n2
    log(f"  road mesh from OpenDRIVE: {n_strips} lanes merged into {nj} asphalt surfaces "
        f"(junctions included, no overlapping pieces)")
    return walk_quads, road_union


def surface_points(lanes, elev):
    """Points on every paved surface of the network (for terrain fitting).

    Returns (points (k,3), direction (k,2), is_asphalt (k,), bridge (k,), tunnel (k,))."""
    P, D, A = [], [], []
    for ln in lanes:
        if len(ln.pts) < 2:
            continue
        d = np.gradient(ln.pts[:, :2], axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            P.append(ln.left * (1 - t) + ln.right * t)
            D.append(d)
            A.append(np.full(len(ln.pts), "Sidewalk" not in ln.type))
    P, D, A = np.concatenate(P), np.concatenate(D), np.concatenate(A)
    dem = elev.sample_xy(P[:, 0], P[:, 1])
    return P, D, A, (P[:, 2] - dem) > 2.5, (P[:, 2] - dem) < -3.0


def guardrails_from_network(scene, lanes, speeds, ground, barrier_pts, log, min_speed=90.0):
    """Guard rails along the outer edges of fast roads, stopping at junctions."""
    from .features import guardrail_mesh
    btree = cKDTree(barrier_pts) if len(barrier_pts) else None
    by_side = {}
    for ln in lanes:
        if ln.junction >= 0 or speeds.get(ln.road, 0) < min_speed or len(ln.pts) < 10:
            continue
        key = (ln.road, ln.section, np.sign(ln.lane))
        by_side.setdefault(key, []).append(ln)
    total = 0.0
    for (road, sec, sign), group in by_side.items():
        outer = max(group, key=lambda q: abs(q.lane))
        opposite = any(k[0] == road and k[1] == sec and k[2] == -sign for k in by_side)
        edges = [(outer.right, -1)]  # right edge, rail offset to the right
        if not opposite:  # one-way carriageway: rail on the median side too
            inner = min(group, key=lambda q: abs(q.lane))
            edges.append((inner.left, 1))
        for E, side in edges:
            d = np.gradient(E[:, :2], axis=0)
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            n = np.column_stack([-d[:, 1], d[:, 0]])
            P = np.column_stack([E[:, :2] + n * 0.5 * side, E[:, 2]])
            ok = np.ones(len(P), bool)
            ok[:3] = ok[-3:] = False
            if btree is not None:
                dd, _ = btree.query(P[:, :2], distance_upper_bound=3.0)
                ok &= ~np.isfinite(dd)
            from .geo import runs
            for a, b in runs(ok):
                if b - a < 5:
                    continue
                seg = P[a:b]
                scene.record("barrier", type="guard_rail", points=seg.tolist(), road_left=bool(side < 0))
                for k in range(0, len(seg) - 1, 40):
                    p = seg[k:k + 41]
                    if len(p) >= 2:
                        beam, posts = guardrail_mesh(p[:, :2], p[:, 2])
                        scene.add("GuardRail", "M_Metal", *beam)
                        scene.add("GuardRail", "M_Metal", *posts)
                total += float(np.hypot(*np.diff(seg[:, :2], axis=0).T).sum())
    log(f"  guard rails along fast roads: {total / 1000:.1f} km (they stop at every junction and exit)")


def _keep_car_lanes(tags):
    """Tags where bus lanes leave at least one car lane per direction.

    OSM often counts `lanes` without the bus lane (busway:right=lane with
    lanes=1); the converter then gives the only lane to buses, types it
    "restricted", and cars can't drive the road in CARLA or Autoware."""
    def n_bus(side):
        v = tags.get(f"busway:{side}") or tags.get("busway:both") or tags.get("busway")
        return 1 if v in ("lane", "opposite_lane") else 0
    try:
        lanes = int(str(tags.get("lanes", "")).split(";")[0])
    except ValueError:
        return tags
    oneway = tags.get("oneway") in ("yes", "1", "-1") or tags.get("junction") == "roundabout"
    bus = n_bus("right") + (0 if oneway else n_bus("left"))
    for k in ("lanes:psv", "lanes:bus"):
        try:
            bus = max(bus, int(tags.get(k, 0)))
        except ValueError:
            pass
    need = bus + (1 if oneway else 2)
    if bus and lanes < need:
        out = dict(tags)
        out["lanes"] = str(need)
        return out
    return tags


def osm_for_network(osm, corridor, frame, margin=40.0, route_xy=None, log=print):
    """OSM XML with the drivable roads cut to the corridor (plus a margin).

    Ways are cut on their geometry, not on their nodes: where a road leaves
    the corridor a new node is created on the border, so long straight
    segments (motorway nodes can be 600 m apart) are kept up to the edge.

    The converter's SUMO type map closes highway=service to cars (its lanes
    come out as Restricted), so service roads the selected route drives on
    are passed as highway=unclassified, 20 km/h unless tagged otherwise."""
    from xml.sax.saxutils import quoteattr
    from shapely.geometry import LineString
    area = corridor.buffer(margin)
    n_bus = 0
    on_route = LineString(route_xy).buffer(4.0) if route_xy is not None and len(route_xy) > 1 else None
    from shapely.prepared import prep
    on_route_p = prep(on_route) if on_route is not None else None
    n_service = 0
    ways_out = []
    extra_nodes = {}  # new id -> (lon, lat)
    used = set()
    new_way = 9 * 10 ** 11
    new_node = [8 * 10 ** 11]
    for wid, w in osm.ways.items():
        if w["tags"].get("highway") not in WAY_TYPES or w["tags"].get("area") == "yes":
            continue
        nds = [n for n in w["nodes"] if n in osm.node_xy]
        if len(nds) < 2:
            continue
        P = np.array([osm.node_xy[n] for n in nds])
        line = LineString(P)
        if not line.intersects(area):
            continue
        if w["tags"].get("highway") == "service" and on_route is not None and on_route_p.intersects(line):
            driven = line.intersection(on_route).length
            if driven > 15.0 or driven > 0.6 * line.length:
                tags = {k: v for k, v in w["tags"].items() if k not in ("service", "access")}
                tags["highway"] = "unclassified"
                tags.setdefault("maxspeed", "20")
                w = dict(w, tags=tags)
                n_service += 1
        tags2 = _keep_car_lanes(w["tags"])
        if tags2 is not w["tags"]:
            w = dict(w, tags=tags2)
            n_bus += 1
        if area.contains(line):
            ways_out.append((wid, nds, w["tags"]))
            used.update(nds)
            continue
        by_xy = {(round(x, 3), round(y, 3)): n for n, (x, y) in zip(nds, P)}
        inter = line.intersection(area)
        parts = [g for g in getattr(inter, "geoms", [inter]) if g.geom_type == "LineString" and g.length > 1.0]
        for k, part in enumerate(parts):
            ids = []
            for x, y in part.coords:
                n = by_xy.get((round(x, 3), round(y, 3)))
                if n is None:
                    new_node[0] += 1
                    n = new_node[0]
                    lon, lat = frame.to_lonlat(np.array([x]), np.array([y]))
                    extra_nodes[n] = (float(lon[0]), float(lat[0]))
                if not ids or ids[-1] != n:
                    ids.append(n)
            if len(ids) < 2:
                continue
            pid = wid if k == 0 else new_way
            if k:
                new_way += 1
            ways_out.append((pid, ids, w["tags"]))
            used.update(i for i in ids if i in osm.nodes)
    if n_service:
        log(f"  {n_service} service roads on the route opened to cars (the converter closes service roads)")
    if n_bus:
        log(f"  {n_bus} roads with a bus lane keep a car lane (the converter gave their only lane to buses)")
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="duplicat_env">']
    for nid in sorted(used):
        lon, lat, tags = osm.nodes[nid]
        keep = {k: v for k, v in tags.items() if k in ("highway", "traffic_signals", "crossing")}
        if keep:
            lines.append(f'<node id="{nid}" version="1" lat="{lat:.8f}" lon="{lon:.8f}">')
            lines += [f"<tag k={quoteattr(k)} v={quoteattr(str(v))}/>" for k, v in keep.items()]
            lines.append("</node>")
        else:
            lines.append(f'<node id="{nid}" version="1" lat="{lat:.8f}" lon="{lon:.8f}"/>')
    for nid, (lon, lat) in extra_nodes.items():
        lines.append(f'<node id="{nid}" version="1" lat="{lat:.8f}" lon="{lon:.8f}"/>')
    for wid, nds, tags in ways_out:
        lines.append(f'<way id="{wid}" version="1">')
        lines += [f'<nd ref="{n}"/>' for n in nds]
        lines += [f"<tag k={quoteattr(k)} v={quoteattr(str(v))}/>" for k, v in tags.items()]
        lines.append("</way>")
    lines.append("</osm>")
    return "\n".join(lines)


def merged_surface(scene, quads, cat, mat, chunk=250.0, raise_z=0.0, curb=0.3, subtract=None):
    """Union of many quads (x,y,z corners) triangulated as one surface per
    chunk: overlapping pieces become a single layer, so nothing flickers.

    quads: array (k, 4, 3). Heights come from the nearest quad corner."""
    import shapely
    from scipy.spatial import Delaunay
    from . import meshlib as ml
    if len(quads) == 0:
        return 0, {}
    Q = np.asarray(quads, float)
    qmin, qmax = Q[:, :, :2].min(axis=1), Q[:, :, :2].max(axis=1)
    i0 = np.floor(qmin / chunk).astype(int)
    i1 = np.floor(qmax / chunk).astype(int)
    keys = set()
    for a, b in zip(i0, i1):
        for cx in range(a[0], b[0] + 1):
            for cy in range(a[1], b[1] + 1):
                keys.add((cx, cy))
    n_surf = 0
    unions = {}
    all_polys = shapely.make_valid(shapely.polygons(Q[:, :, :2]))
    for cx, cy in sorted(keys):
        # every quad touching this square, clipped exactly to the square, so
        # neighbouring squares meet edge to edge without overlapping
        sel = (i0[:, 0] <= cx) & (i1[:, 0] >= cx) & (i0[:, 1] <= cy) & (i1[:, 1] >= cy)
        q = Q[sel]
        sq = shapely.box(cx * chunk, cy * chunk, (cx + 1) * chunk, (cy + 1) * chunk)
        area = shapely.union_all(all_polys[sel]).buffer(0.03).buffer(-0.03).intersection(sq)
        if subtract is not None and (cx, cy) in subtract:
            area = area.difference(subtract[(cx, cy)])
        if area.is_empty:
            continue
        unions[(cx, cy)] = area
        corners = q.reshape(-1, 3)
        ztree = cKDTree(corners[:, :2])
        for part in getattr(area, "geoms", [area]):
            if part.geom_type != "Polygon" or part.area < 0.3:
                continue
            seg = shapely.segmentize(part, 1.0)
            bpts = shapely.get_coordinates(seg)
            inner = corners[shapely.contains_xy(part.buffer(-0.3), corners[:, 0], corners[:, 1]), :2]
            pts = np.unique(np.round(np.concatenate([bpts, inner]), 3), axis=0)
            if len(pts) < 3:
                continue
            tri = Delaunay(pts).simplices
            tc = pts[tri].mean(axis=1)
            keep = shapely.contains_xy(part.buffer(0.01), tc[:, 0], tc[:, 1])
            tri = tri[keep]
            if len(tri) == 0:
                continue
            kq = min(6, len(corners))
            d, j = ztree.query(pts, k=kq)
            j = np.reshape(j, (len(pts), kq))
            wgt = 1.0 / (np.reshape(d, (len(pts), kq)) + 0.3)
            V = np.column_stack([pts, (corners[j, 2] * wgt).sum(1) / wgt.sum(1) + raise_z])
            F = tri.copy()
            flip = ml.face_normals(V, F)[:, 2] < 0
            F[flip] = F[flip][:, ::-1]
            scene.add(cat, mat, V, F, pts / 4.0)
            for ring in [part.exterior] + list(part.interiors):
                R = np.array(shapely.segmentize(ring, 1.0).coords)
                _, j = ztree.query(R)
                E = np.column_stack([R, corners[j, 2] + raise_z])
                scene.add(cat, mat, *ml.ribbon(E, E - [0, 0, curb + 0.9]))
            n_surf += 1
    return n_surf, unions
