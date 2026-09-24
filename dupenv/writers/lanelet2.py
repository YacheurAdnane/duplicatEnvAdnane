# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Autoware Lanelet2 map (.osm) for the selected route.

Nodes carry local_x / local_y / ele tags in the same frame as the FBX and the
OpenDRIVE file, and the projector is written as `Local`, so Autoware uses the
exact coordinates CARLA uses.
"""
import numpy as np

from ..geo import dp_simplify
from ..roads import EPS_W
from .xodr import lane_sections


def write_lanelet2(path, route, frame, chunk_len=50.0, extra_models=()):
    ids = [0]

    def nid():
        ids[0] += 1
        return ids[0]

    nodes = []      # (id, x, y, z)
    ways = []       # (id, [node ids], tags)
    rels = []       # (id, left way, right way, tags)
    models = [route] + [m for m in extra_models if m.kind == "road"]

    for m in models:
        ranges, nR, _ = lane_sections(m)
        # chunk boundaries: section starts plus every chunk_len metres
        cuts = set()
        for a, b in ranges:
            cuts.add(a)
            k = a
            while m.s[b] - m.s[k] > 1.5 * chunk_len:
                k = int(np.searchsorted(m.s, m.s[k] + chunk_len))
                cuts.add(k)
        cuts.add(m.n - 1)
        cuts = sorted(cuts)
        end_nodes = {}  # (sample index, boundary k) -> node id

        def boundary_nodes(a, b, k, off):
            P = m.point(off)[a:b + 1]
            keep = dp_simplify(P[:, :2], 0.02)
            out = []
            for j in keep:
                i = a + j
                if j in (0, b - a):
                    key = (i, k)
                    if key not in end_nodes:
                        end_nodes[key] = nid()
                        nodes.append((end_nodes[key], *P[j]))
                    out.append(end_nodes[key])
                else:
                    q = nid()
                    nodes.append((q, *P[j]))
                    out.append(q)
            return out

        cum = np.concatenate([np.zeros((m.n, 1)), np.cumsum(m.wr, axis=1)], axis=1)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            r = int(nR[a:b + 1].max())
            if r == 0:
                continue
            spd = int(round(np.median(m.speed[a:b + 1])))
            bways = []
            for k in range(r + 1):
                off = m.c0 - cum[:, k]
                nds = boundary_nodes(a, b, k, off)
                if k == 0 or k == r:
                    tags = {"type": "line_thin", "subtype": "solid", "lane_change": "no"}
                else:
                    tags = {"type": "line_thin", "subtype": "dashed", "lane_change": "yes"}
                wid = nid()
                ways.append((wid, nds, tags))
                bways.append(wid)
            for k in range(1, r + 1):
                if m.wr[a:b + 1, k - 1].max() <= EPS_W:
                    continue
                rels.append((nid(), bways[k - 1], bways[k], {
                    "type": "lanelet", "subtype": "road", "location": "urban",
                    "one_way": "yes", "participant:vehicle": "yes",
                    "speed_limit": str(spd),
                }))

    arr = np.array([(x, y) for _, x, y, _ in nodes]) if nodes else np.zeros((0, 2))
    lon, lat = frame.to_lonlat(arr[:, 0], arr[:, 1]) if len(arr) else ([], [])
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="duplicat_env">\n')
        f.write('  <MetaInfo format_version="1" map_version="1"/>\n')
        for (i, x, y, z), lo, la in zip(nodes, lon, lat):
            f.write(f'  <node id="{i}" lat="{la:.11f}" lon="{lo:.11f}">\n'
                    f'    <tag k="local_x" v="{x:.4f}"/>\n    <tag k="local_y" v="{y:.4f}"/>\n'
                    f'    <tag k="ele" v="{z:.4f}"/>\n  </node>\n')
        for i, nds, tags in ways:
            f.write(f'  <way id="{i}">\n')
            for n in nds:
                f.write(f'    <nd ref="{n}"/>\n')
            for k, v in tags.items():
                f.write(f'    <tag k="{k}" v="{v}"/>\n')
            f.write("  </way>\n")
        for i, lw, rw, tags in rels:
            f.write(f'  <relation id="{i}">\n    <member type="way" role="left" ref="{lw}"/>\n'
                    f'    <member type="way" role="right" ref="{rw}"/>\n')
            for k, v in tags.items():
                f.write(f'    <tag k="{k}" v="{v}"/>\n')
            f.write("  </relation>\n")
        f.write("</osm>\n")
    return len(rels)


def write_projector_info(path, frame):
    with open(path, "w") as f:
        f.write("# local_x / local_y tags of lanelet2_map.osm are used as-is (same frame as CARLA)\n")
        f.write("projector_type: Local\n")
        f.write(f"# frame origin (for reference): lat {frame.lat0:.9f}, lon {frame.lon0:.9f}\n")
