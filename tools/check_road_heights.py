#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Check the heights of every road in an OpenDRIVE file.

    python3 tools/check_road_heights.py output/<name>/carla/<name>/<name>.xodr

Reports roads left near height 0 while the terrain is higher (they show as
roads under the ground in RoadRunner and CARLA), roads far above or below
their neighbours, and steps where linked roads meet. Exit code 1 if any.
"""
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree


def main(path):
    import carla
    x = open(path).read()
    cmap = carla.Map("check", x)
    root = ET.fromstring(x)
    samples, ends = {}, {}
    for road in root.findall("road"):
        rid = int(road.get("id"))
        L = float(road.get("length"))
        sec = road.find("lanes").find("laneSection")
        ids = [int(ln.get("id")) for ln in sec.iter("lane") if ln.get("id") != "0"]
        if not ids:
            continue
        lid = min(ids, key=abs)
        pts = []
        for s in np.linspace(0.01, max(L - 0.01, 0.02), max(2, int(L / 5) + 1)):
            try:
                w = cmap.get_waypoint_xodr(rid, lid, float(s))
            except Exception:
                w = None
            if w is not None:
                q = w.transform.location
                pts.append((q.x, -q.y, q.z))
        if pts:
            samples[rid] = np.array(pts)
    allp = np.concatenate(list(samples.values()))
    med = float(np.median(allp[:, 2]))
    tree = cKDTree(allp[:, :2])
    low, odd = [], []
    for rid, P in samples.items():
        z = float(P[:, 2].mean())
        if med > 5 and abs(z) < 1.0:
            low.append(rid)
            continue
        idx = tree.query_ball_point(P[len(P) // 2, :2], 60.0)
        zn = float(np.median(allp[idx, 2]))
        if abs(z - zn) > 12:  # bridges and tunnels differ by ~6-10 m
            odd.append((rid, round(z - zn, 1)))
    # steps where road links meet (road-to-road and junction connectors)
    steps = []
    for road in root.findall("road"):
        rid = int(road.get("id"))
        lk = road.find("link")
        if lk is None or rid not in samples:
            continue
        for tag, mine in (("predecessor", 0), ("successor", -1)):
            e = lk.find(tag)
            if e is None or e.get("elementType") != "road":
                continue
            oid = int(e.get("elementId"))
            if oid not in samples:
                continue
            other = samples[oid][0 if e.get("contactPoint", "start") == "start" else -1]
            me = samples[rid][mine]
            dxy = math.hypot(me[0] - other[0], me[1] - other[1])
            if dxy < 6 and abs(me[2] - other[2]) > 0.3:
                steps.append((rid, oid, round(abs(me[2] - other[2]), 2)))
    sig_low = [s.get("id") for s in root.iter("signal")
               if s.find("positionInertial") is not None and med > 5
               and abs(float(s.find("positionInertial").get("z", 0))) < 1.0]
    print(f"{len(samples)} roads, median height {med:.1f} m")
    print(f"  signals (traffic lights) at height ~0: {len(sig_low)}")
    print(f"  roads at height ~0: {len(low)}" + (f" (e.g. {low[:8]})" if low else ""))
    print(f"  roads > 12 m off their neighbours: {len(odd)}" + (f" (e.g. {odd[:6]})" if odd else ""))
    print(f"  height steps > 0.3 m at road links: {len(steps)}" + (f" (e.g. {steps[:6]})" if steps else ""))
    return 1 if low or steps or sig_low else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
