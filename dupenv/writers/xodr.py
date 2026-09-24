# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""OpenDRIVE 1.4 writer (lines + linear elevation + lane width polynomials)."""
import numpy as np
from xml.sax.saxutils import escape

from ..geo import dp_simplify
from ..roads import EPS_W


def _linear_records(s, v, tol):
    """Piecewise-linear records (s0, a, b) approximating v(s)."""
    keep = dp_simplify(np.column_stack([s, v]), tol)
    out = []
    for i, j in zip(keep[:-1], keep[1:]):
        ds = s[j] - s[i]
        b = (v[j] - v[i]) / ds if ds > 1e-9 else 0.0
        out.append((s[i], v[i], b))
    if not out:
        out = [(s[0], v[0], 0.0)]
    return out


def lane_sections(m):
    """Sample index ranges with constant lane topology and speed."""
    nR = (m.wr > EPS_W).sum(1) if m.wr.shape[1] else np.zeros(m.n, int)
    nL = (m.wl > EPS_W).sum(1) if m.wl.shape[1] else np.zeros(m.n, int)
    spd = np.round(m.speed).astype(int)
    key = nR * 1000000 + nL * 1000 + spd
    brk = [0] + list(np.nonzero(np.diff(key) != 0)[0] + 1)
    # drop sections shorter than 3 samples (merge into the previous one)
    clean = [brk[0]]
    for b in brk[1:]:
        if b - clean[-1] >= 3 and m.n - b >= 3:
            clean.append(b)
    ranges = []
    for a, b in zip(clean, clean[1:] + [m.n - 1]):
        ranges.append((a, b))
    return ranges, nR, nL


def _geometry_s(m):
    """Chord-length s along DP-simplified planView, for every sample."""
    P = np.column_stack([m.x, m.y])
    keep = dp_simplify(P, 0.03)
    seg = np.hypot(np.diff(m.x[keep]), np.diff(m.y[keep]))
    sk = np.concatenate([[0.0], np.cumsum(seg)])
    s_all = np.interp(np.arange(m.n), keep, sk)
    return keep, s_all


def _road_xml(m, rid, name, f, signals_from=None):
    keep, s = _geometry_s(m)
    L = s[-1]
    f.write(f'  <road name="{escape(name)}" length="{L:.4f}" id="{rid}" junction="-1">\n')
    f.write("    <link/>\n")
    f.write(f'    <type s="0.0000" type="{m.xodr_type}"><speed max="{int(round(m.speed.max()))}" unit="km/h"/></type>\n')
    f.write("    <planView>\n")
    for i, j in zip(keep[:-1], keep[1:]):
        dx, dy = m.x[j] - m.x[i], m.y[j] - m.y[i]
        ln = s[j] - s[i]
        if ln < 1e-6:
            continue
        hdg = np.arctan2(dy, dx)
        f.write(f'      <geometry s="{s[i]:.4f}" x="{m.x[i]:.4f}" y="{m.y[i]:.4f}" hdg="{hdg:.8f}" length="{ln:.4f}"><line/></geometry>\n')
    f.write("    </planView>\n    <elevationProfile>\n")
    for s0, a, b in _linear_records(s, m.z, 0.01):
        f.write(f'      <elevation s="{s0:.4f}" a="{a:.4f}" b="{b:.7f}" c="0" d="0"/>\n')
    f.write("    </elevationProfile>\n    <lateralProfile/>\n    <lanes>\n")
    for s0, a, b in _linear_records(s, m.c0, 0.005):
        f.write(f'      <laneOffset s="{s0:.4f}" a="{a:.4f}" b="{b:.7f}" c="0" d="0"/>\n')

    ranges, nR, nL = lane_sections(m)
    prev = None
    for si, (a, b) in enumerate(ranges):
        r = int(nR[a:b + 1].max())
        l = int(nL[a:b + 1].max())
        ss = s[a:b + 1] - s[a]
        spd = int(round(np.median(m.speed[a:b + 1])))
        nxt = ranges[si + 1] if si + 1 < len(ranges) else None
        nr_next = int(nR[nxt[0]:nxt[1] + 1].max()) if nxt else 0
        nl_next = int(nL[nxt[0]:nxt[1] + 1].max()) if nxt else 0
        nxt_rsh = bool(nxt) and m.rsh[nxt[0]:nxt[1] + 1].max() > 0.05
        nxt_lsh = bool(nxt) and m.lsh[nxt[0]:nxt[1] + 1].max() > 0.05
        f.write(f'      <laneSection s="{s[a]:.4f}">\n')

        def lane(lid, typ, widths, mark, pred, succ, speed=True):
            f.write(f'          <lane id="{lid}" type="{typ}" level="false">\n            <link>')
            if pred:
                f.write(f'<predecessor id="{pred}"/>')
            if succ:
                f.write(f'<successor id="{succ}"/>')
            f.write("</link>\n")
            for s0, wa, wb in _linear_records(ss, widths, 0.005):
                f.write(f'            <width sOffset="{s0:.4f}" a="{max(wa, 0):.4f}" b="{wb:.7f}" c="0" d="0"/>\n')
            if mark:
                f.write(f'            <roadMark sOffset="0" type="{mark}" weight="standard" color="standard" width="0.15" laneChange="{"both" if mark == "broken" else "none"}"/>\n')
            if speed and typ == "driving":
                f.write(f'            <speed sOffset="0" max="{spd}" unit="km/h"/>\n')
            f.write("          </lane>\n")

        has_prev = prev is not None
        # left side (opposite direction lanes, or inner shoulder of a one-way)
        # lane links: driving lanes keep their id, shoulders link to the
        # neighbouring section's shoulder (whose id depends on its lane count)
        left = []
        for k in range(l, 0, -1):
            mark = "solid" if k == l else "broken"
            pred = k if has_prev and k <= prev[1] else None
            succ = k if nxt and k <= nl_next else None
            left.append((k, "driving", m.wl[a:b + 1, k - 1], mark, pred, succ))
        lsh = m.lsh[a:b + 1]
        has_lsh = lsh.max() > 0.05
        if has_lsh:
            pred = prev[1] + 1 if has_prev and prev[3] else None
            succ = nl_next + 1 if nxt and nxt_lsh else None
            left.insert(0, (l + 1, "shoulder", lsh, None, pred, succ))
        if left:
            f.write("        <left>\n")
            for args in left:
                lane(*args)
            f.write("        </left>\n")
        cmark = "solid" if (m.oneway or l == 0) else ("broken" if r + l <= 3 else "solid")
        f.write(f'        <center>\n          <lane id="0" type="none" level="false">\n'
                f'            <roadMark sOffset="0" type="{cmark}" weight="standard" color="standard" width="0.15"/>\n'
                f"          </lane>\n        </center>\n")
        f.write("        <right>\n")
        for k in range(1, r + 1):
            mark = "solid" if k == r else "broken"
            pred = -k if has_prev and k <= prev[0] else None
            succ = -k if nxt and k <= nr_next else None
            lane(-k, "driving", m.wr[a:b + 1, k - 1], mark, pred, succ)
        rsh = m.rsh[a:b + 1]
        has_rsh = rsh.max() > 0.05
        if has_rsh:
            pred = -(prev[0] + 1) if has_prev and prev[2] else None
            succ = -(nr_next + 1) if nxt and nxt_rsh else None
            lane(-(r + 1), "shoulder", rsh, None, pred, succ)
        f.write("        </right>\n      </laneSection>\n")
        prev = (r, l, has_rsh, has_lsh)
    f.write("    </lanes>\n    <objects/>\n    <signals>\n")
    if signals_from is not None:
        _, R = m.surface_edges()
        sid = rid * 1000
        last = None
        for i in range(0, m.n, 5):
            v = int(round(m.speed[i]))
            if v != last:
                t = float(R[i] - 1.0)
                f.write(f'      <signal s="{s[i]:.4f}" t="{t:.4f}" id="{sid}" name="speed_{v}" dynamic="no" '
                        f'orientation="+" zOffset="1.5" country="DE" type="274" subtype="{v}" value="{v}" '
                        f'unit="km/h" height="0.8" width="0.8"/>\n')
                sid += 1
                last = v
    f.write("    </signals>\n  </road>\n")


def write_xodr(path, name, frame, models, bounds_xy):
    minx, miny, maxx, maxy = bounds_xy
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" standalone="yes"?>\n<OpenDRIVE>\n')
        f.write(f'  <header revMajor="1" revMinor="4" name="{escape(name)}" version="1.00" '
                f'north="{maxy:.4f}" south="{miny:.4f}" east="{maxx:.4f}" west="{minx:.4f}" vendor="duplicat_env">\n')
        f.write(f"    <geoReference><![CDATA[{frame.proj_string}]]></geoReference>\n  </header>\n")
        rid = 1
        for m in models:
            if m.kind not in ("route", "road") or m.n < 3:
                continue
            if m.kind == "road" and m.hclass[0] == "service":
                continue
            _road_xml(m, rid, m.name, f, signals_from=True if m.kind == "route" else None)
            rid += 1
        f.write("</OpenDRIVE>\n")
    return path
