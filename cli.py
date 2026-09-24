#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Command line version of the generator.

    python3 cli.py --waypoints "48.7117,2.2851;48.6167,2.1260" --name A10_Massy_Briis
    python3 cli.py --area zone.geojson --name Massy_zone      # everything inside a polygon
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dupenv import pipeline, routing  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waypoints", help='"lat,lon;lat,lon;..." routed along OSM roads with OSRM')
    ap.add_argument("--geojson", help="LineString GeoJSON file used as the exact route instead")
    ap.add_argument("--area", help="Polygon GeoJSON file: build everything inside it (area mode)")
    ap.add_argument("--name", default="A10_twin")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    ap.add_argument("--set", action="append", default=[], help="option=value, e.g. corridor=200 ortho=false")
    ap.add_argument("--coords-json", help=argparse.SUPPRESS)   # used by server.py
    ap.add_argument("--options-json", help=argparse.SUPPRESS)  # used by server.py
    ap.add_argument("--events", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.coords_json:
        coords = json.load(open(a.coords_json))
        cfg = json.load(open(a.options_json)) if a.options_json else {}

        def log(m):
            print("@@LOG " + m.replace("\n", " "), flush=True)

        def prog(st, fr):
            print(f"@@PROG {st} {fr:.4f}", flush=True)

        def tasks(snap):
            print("@@TASKS " + json.dumps(snap, separators=(",", ":")), flush=True)

        out = pipeline.run(a.out, coords, cfg, log=log, progress_cb=prog, tasks_cb=tasks)
        print("@@OUT " + out, flush=True)
        return
    area = False
    if a.area:
        g = json.load(open(a.area))
        g = g.get("geometry", g)
        coords = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"]
        area = True
    elif a.geojson:
        g = json.load(open(a.geojson))
        g = g.get("geometry", g)
        coords = g["coordinates"]
    elif a.waypoints:
        wps = [[float(p.split(",")[1]), float(p.split(",")[0])] for p in a.waypoints.split(";")]
        coords, dist, info = routing.route(wps)
        print(f"route: {dist / 1000:.2f} km, detour ratio {info['detour_ratio']}, U-turns {info['uturns']}")
    else:
        ap.error("give --waypoints, --geojson or --area")
    cfg = {"name": a.name, "area": area}
    for kv in a.set:
        k, v = kv.split("=", 1)
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = float(v)
            except ValueError:
                pass
        cfg[k] = v
    last = [None, None, 0.0]

    def show(snap):
        # one line whenever the running tasks change, so a terminal run shows movement too
        parts = []
        for x in snap["tasks"]:
            if x["state"] != "running" or x["key"] == "download":
                continue
            pct = "" if x["frac"] is None else f" {100 * x['frac']:.0f}%"
            parts.append(f"{x['label'].split(' (')[0]}{pct} {x['detail']}".strip())
        if not parts:
            return
        ram = snap.get("ram") or {}
        line = f"[{snap['overall'] * 100:5.1f}%] " + " | ".join(parts)
        keys = [x["key"] for x in snap["tasks"] if x["state"] == "running"]
        if line != last[0] and (keys != last[1] or time.time() - last[2] > 3):
            last[0], last[1], last[2] = line, keys, time.time()
            print(f"{line}  (RAM {100 * ram.get('used', 0):.0f}%)", flush=True)

    pipeline.run(a.out, coords, cfg, progress_cb=lambda s, f: None, tasks_cb=show)


if __name__ == "__main__":
    main()
