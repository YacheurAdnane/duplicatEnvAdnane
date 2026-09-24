#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Load lanelet2_map.osm with the real Lanelet2 library (ROS 2 Humble) and
check routing along the route. Run inside a sourced ROS 2 shell:

    source /opt/ros/humble/setup.bash
    python3 tools/validate_lanelet2.py output/<name>/autoware/<name>/lanelet2_map.osm
"""
import sys

import lanelet2
from lanelet2.core import GPSPoint
from lanelet2.io import Origin
from lanelet2.projection import LocalCartesianProjector


def main(path):
    # coordinates come from local_x/local_y tags, so any projector works for loading
    proj = LocalCartesianProjector(Origin(GPSPoint(0.0, 0.0, 0.0)))
    m, errors = lanelet2.io.loadRobust(path, proj)
    print(f"loaded: {len(m.laneletLayer)} lanelets, {len(m.lineStringLayer)} line strings, "
          f"{len(m.pointLayer)} points, {len(errors)} load warnings")
    for e in errors[:5]:
        print("  ", e)
    # restore local coordinates from the tags (what Autoware's Local projector does);
    # the loader already turned the ele tag into z
    for p in m.pointLayer:
        p.x = float(p.attributes["local_x"])
        p.y = float(p.attributes["local_y"])
    rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                          lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(m, rules)
    errs = graph.checkValidity()
    print(f"routing graph validity issues: {len(errs)}")
    lls = list(m.laneletLayer)
    dead = sum(1 for ll in lls if not graph.following(ll))
    print(f"lanelets without a successor: {dead} (map ends at the corridor edge)")
    import json
    import os
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))),
                             "metadata.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        sp_, gl = meta["spawn"], meta["goal"]

        import math

        def heading(ll):
            c = ll.centerline
            return math.atan2(c[-1].y - c[0].y, c[-1].x - c[0].x)

        def nearest(x, y, yaw):
            """closest lanelet driving roughly in direction yaw"""
            best = None
            for ll in lls:
                if math.cos(heading(ll) - yaw) < 0.5:
                    continue
                for p in ll.centerline:
                    d = (p.x - x) ** 2 + (p.y - y) ** 2
                    if best is None or d < best[0]:
                        best = (d, ll)
            return best[1]
        yaw0 = math.radians(sp_["yaw_deg_enu"])
        route_file = os.path.join(os.path.dirname(meta_path), "route.geojson")
        yaw1 = yaw0
        if os.path.exists(route_file):
            from pyproj import Transformer
            c = json.load(open(route_file))["geometry"]["coordinates"]
            tr = Transformer.from_crs("EPSG:4326", meta["proj"], always_xy=True)
            (xa, xb), (ya, yb) = tr.transform([c[-2][0], c[-1][0]], [c[-2][1], c[-1][1]])
            yaw1 = math.atan2(yb - ya, xb - xa)
        start = nearest(sp_["x"], sp_["y"], yaw0)
        # lanelets near the goal, closest first; the goal counts as reached by the first reachable one
        cands = sorted(((min((p.x - gl["x"]) ** 2 + (p.y - gl["y"]) ** 2 for p in ll.centerline), ll)
                        for ll in lls if math.cos(heading(ll) - yaw1) > 0.3), key=lambda t: t[0])[:30]
        end = next((ll for d, ll in cands if d < 15 ** 2 and graph.getRoute(start, ll) is not None),
                   cands[0][1] if cands else start)
        print("routing from the start of the selected route to its end")
    else:
        start = min(lls, key=lambda ll: ll.id)
        end = max(lls, key=lambda ll: ll.id)
    route = graph.getRoute(start, end)
    if route is None:
        print("NO route between the start and the end lanelet")
        if os.path.exists(meta_path) and os.path.exists(route_file):
            follow_route(lls, graph, start, meta, route_file)
        return 1
    sp = route.shortestPath()
    length = sum(lanelet2.geometry.length2d(ll) for ll in sp)
    print(f"route found: {len(sp)} lanelets, {length:.0f} m, lane changes allowed where lines are dashed")
    if os.path.exists(meta_path) and os.path.exists(route_file):
        return follow_route(lls, graph, start, meta, route_file)
    return 0


def follow_route(lls, graph, start, meta, route_file, step=10.0):
    """Every `step` m of the selected route must lie on a lanelet driving the
    same way that is reachable from the start lanelet. A start-to-goal route
    alone proves little on a loop, where the goal is next to the start."""
    import json
    import math

    import numpy as np
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    c = np.array(json.load(open(route_file))["geometry"]["coordinates"])
    x, y = Transformer.from_crs("EPSG:4326", meta["proj"], always_xy=True).transform(c[:, 0], c[:, 1])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    ss = np.arange(0, s[-1], step)
    Q = np.column_stack([np.interp(ss, s, x), np.interp(ss, s, y)])
    H = np.arctan2(np.gradient(Q[:, 1]), np.gradient(Q[:, 0]))
    pts, owner, hd = [], [], []
    for ll in lls:
        cl = list(ll.centerline)
        for a, b in zip(cl, cl[1:]):
            n = max(1, int(math.hypot(b.x - a.x, b.y - a.y) / 2))
            for f in (np.arange(n) + 0.5) / n:
                pts.append((a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f))
                owner.append(ll.id)
                hd.append(math.atan2(b.y - a.y, b.x - a.x))
    tree, hd = cKDTree(pts), np.array(hd)
    def on_at(k):
        return {owner[j] for j in tree.query_ball_point(Q[k], 8.0) if math.cos(hd[j] - H[k]) > 0.7}
    # start from the route's first point (the spawn lies a few metres into it)
    by_id = {ll.id: ll for ll in lls}
    reach = {ll.id for ll in graph.reachableSet(start, 1e9)}
    for lid in on_at(0):
        reach |= {ll.id for ll in graph.reachableSet(by_id[lid], 1e9)}
    missing, cut = [], []
    for k in range(len(Q)):
        on = on_at(k)
        if not on:
            missing.append(ss[k])
        elif not on & reach:
            cut.append(ss[k])
    print(f"along the route ({len(Q)} points, every {step:.0f} m): {len(missing)} without a lane, "
          f"{len(cut)} not reachable from the start")
    if cut:
        print(f"  first unreachable point at {cut[0]:.0f} m")
    return 1 if missing or cut else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
