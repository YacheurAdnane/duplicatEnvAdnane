#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Check a generated .xodr with CARLA's own parser (no simulator needed).

    python3 tools/validate_xodr.py output/<name>/carla/<name>/<name>.xodr

Loads the map, generates waypoints, follows every lane of the route (road 1)
from start to end and prints where lanes stop, so you know the map is
drivable before importing it into Unreal.
"""
import sys

import carla


def main(path):
    m = carla.Map("check", open(path).read())
    wps = m.generate_waypoints(5.0)
    roads = {w.road_id for w in wps}
    print(f"OK: parsed, {len(wps)} waypoints on {len(roads)} roads, {len(m.get_spawn_points())} spawn points")
    import json
    import os
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))),
                             "metadata.json")
    if os.path.exists(meta_path):
        # network map: drive from the spawn point through roads and junctions
        sp = json.load(open(meta_path))["spawn"]["carla"]
        w = m.get_waypoint(carla.Location(sp["x"], sp["y"], sp["z"]))
        dist, roads, juncs, last = 0.0, {w.road_id}, 0, None
        lengths = [float(v) for v in __import__("re").findall(
            r'<road [^>]*length="([\d.]+)"[^>]*junction="-1"', open(path).read())]
        while dist < 100000:
            nx = w.next(5.0)
            if not nx:
                break
            yaw = w.transform.rotation.yaw
            w = min(nx, key=lambda q: abs((q.transform.rotation.yaw - yaw + 180) % 360 - 180))
            dist += 5.0
            roads.add(w.road_id)
            if w.is_junction and w.junction_id != last:
                juncs += 1
                last = w.junction_id
            elif not w.is_junction:
                last = None
        print(f"  longest road {max(lengths):.0f} m; driving from the spawn point: {dist:.0f} m "
              f"through {len(roads)} roads and {juncs} junctions (straightest branch) before the map ends")
        return 0
    route = [w for w in wps if w.road_id == 1]
    if not route:
        print("no route road (id 1)")
        return 1
    starts = {}
    for w in route:
        first = (w.s < starts[w.lane_id].s) if w.lane_id in starts else True
        if w.lane_id > 0 and w.lane_id in starts:
            first = w.s > starts[w.lane_id].s
        if first:
            starts[w.lane_id] = w
    length = max(w.s for w in route)
    for lane_id, w in sorted(starts.items(), reverse=True):
        n, last = 0, w
        step = last.previous if lane_id > 0 else last.next  # left lanes run against s
        while n < 100000:
            nx = [q for q in step(2.0) if q.road_id == 1]
            if not nx:
                break
            last = nx[0]
            step = last.previous if lane_id > 0 else last.next
            n += 1
        print(f"  lane {lane_id:3d}: starts s={w.s:8.1f}  ends s={last.s:8.1f} "
              f"(lane {last.lane_id}), width at start {w.lane_width:.2f} m, z {w.transform.location.z:.2f}")
    print(f"  route length {length:.0f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
