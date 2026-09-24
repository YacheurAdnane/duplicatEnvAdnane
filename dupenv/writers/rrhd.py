# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""RoadRunner HD Map (.rrhd) writer.

Lanes, lane boundaries, markings and speed limits come from the road models.
Trees, bushes, guard rails, fences and the signs that exist in the RoadRunner
library are placed as library assets. Everything else (terrain with aerial
photo, buildings, rail, lamps, panels with French text...) is written as FBX
props next to the .rrhd and placed as static objects at their exact size.
"""
import math
import os

import numpy as np
from google.protobuf.internal.encoder import _VarintBytes

from .. import rr_proto
from ..roads import EPS_W
from .fbx import write_fbx
from .xodr import lane_sections

TREE_ASSETS = {
    "broadleaf": ["Beech01", "Ash01", "Elm01", "Maple01", "Birch01", "Zelkova01"],
    "conifer": ["CoulPine_Med01", "CoulPine_Med02", "CoulPine_Med03", "CoulPine_Lg01", "Cypress_Med01"],
    "bush": ["Bush_Med01", "Bush_Med02", "Bush_Med03", "Bush_Med04", "Bush_Lg01"],
}
BARRIER_ASSETS = {
    "guard_rail": "Assets/Extrusions/GuardRail.rrext",
    "cable_barrier": "Assets/Extrusions/GuardRail02.rrext",
    "jersey_barrier": "Assets/Extrusions/JerseyBarrier.rrext",
    "fence": "Assets/Extrusions/Fence.rrext",
    "handrail": "Assets/Extrusions/MetalFencePost01.rrext",
}
SIGN_DIR = "Assets/Signs/Germany/"
EMBED_PREFIXES = ("Ground", "Terrain", "Building", "Water", "RailTrack", "Pole", "Static", "Wall",
                  "Vegetation_hedge", "Vegetation_grass", "Panel")


class AssetResolver:
    """Checks that an asset path exists in the project or as a built-in."""

    def __init__(self, project_dir, install_dir):
        self.roots = [project_dir]
        if install_dir:
            self.roots.append(os.path.join(install_dir, "AssetsInstall", "Base Project"))

    def exists(self, rel):
        for r in self.roots:
            p = os.path.join(r, rel)
            if os.path.exists(p) or os.path.exists(p + ".rrmeta"):
                return True
        return False


def rr_sign_asset(kind, value):
    """Library sign for a classified French/OSM sign, or None."""
    if kind == "speed" and value:
        return f"{SIGN_DIR}Regulatory Signs/MaxSpeed_{int(float(value))}_DE.svg_rrx"
    return {
        "stop": f"{SIGN_DIR}Regulatory Signs/Stop_DE.svg_rrx",
        "giveway": f"{SIGN_DIR}Regulatory Signs/Yield_DE.svg_rrx",
        "noentry": f"{SIGN_DIR}Regulatory Signs/DoNotEnter_DE.svg_rrx",
        "danger": f"{SIGN_DIR}Warning Signs/Warning_DE.svg_rrx",
    }.get(kind)


def _v3(msg, pts):
    for x, y, z in pts:
        p = msg.values.add()
        p.x, p.y, p.z = float(x), float(y), float(z)


def _obb(geom, center, half, heading):
    geom.center.x, geom.center.y, geom.center.z = (float(c) for c in center)
    geom.dimension.length, geom.dimension.width, geom.dimension.height = (float(h) for h in half)
    geom.geo_orientation.geo_angle.heading = float(heading)


def _thin(P, tol=0.02):
    from ..geo import dp_simplify
    return P[dp_simplify(P[:, :2], tol)]


def _add_lanes(hd, m, mdl_idx, speed_ids, chunk_len=200.0):
    """Lanes + boundaries for one road model, split at lane sections and every chunk_len."""
    ranges, nR, nL = lane_sections(m)
    pieces = []
    for a, b in ranges:
        k = a
        while True:
            e = int(np.searchsorted(m.s, m.s[k] + chunk_len))
            e = min(e, b)
            if b - e < 10:
                e = b
            pieces.append((k, e))
            if e >= b:
                break
            k = e
    cum_r = np.concatenate([np.zeros((m.n, 1)), np.cumsum(m.wr, axis=1)], axis=1)
    cum_l = np.concatenate([np.zeros((m.n, 1)), np.cumsum(m.wl, axis=1)], axis=1)
    kind_type = {"path": 7, "track": 1}  # LANE_TYPE_SIDEWALK, DRIVING
    prev_ids = {}
    by_id = {}
    for pi, (a, b) in enumerate(pieces):
        if b - a < 1:
            continue
        idx = slice(a, b + 1)
        r = int(nR[a:b + 1].max())
        l = int(nL[a:b + 1].max())
        base = f"{m.name}_{pi}"
        spd = int(round(np.median(m.speed[a:b + 1])))

        def boundary(bid, off, mark):
            bd = hd.lane_boundaries.add()
            bd.id = bid
            _v3(bd.geometry, _thin(m.point(off)[idx]))
            if mark:
                pa = bd.parametric_attributes.add()
                pa.span.span_start, pa.span.span_end = 0.0, 1.0
                pa.marking_reference.marking_id.id = mark
            return bid

        def lane(lid, left, right, center_off, ltype, travel, key):
            ln = hd.lanes.add()
            ln.id = lid
            _v3(ln.geometry, _thin(m.point(center_off)[idx]))
            ln.lane_type = ltype
            ln.travel_dir = travel
            ln.left_lane_boundary.reference.id = left
            ln.left_lane_boundary.alignment = 1
            ln.right_lane_boundary.reference.id = right
            ln.right_lane_boundary.alignment = 1
            if ltype == 1 and m.kind in ("route", "road"):
                pa = ln.parametric_attributes.add()
                pa.span.span_start, pa.span.span_end = 0.0, 1.0
                pa.speed_limit_reference.speed_limit_id.id = speed_ids(spd)
            if key in prev_ids:
                p = ln.predecessors.add()
                p.reference.id = prev_ids[key]
                p.alignment = 1
                s = by_id[prev_ids[key]].successors.add()
                s.reference.id = lid
                s.alignment = 1
            new_ids[key] = lid
            by_id[lid] = ln

        new_ids = {}
        road = m.kind in ("route", "road")
        edge = "solid" if (road and m.edge_lines) else None
        sep = "dashed" if road else None
        # right side: boundaries R0..Rr, lanes 1..r driving along the samples
        rb = []
        for k in range(r + 1):
            if k == 0:
                mark = ("solid" if m.oneway else ("dashed" if r + l <= 3 else "solid")) if road else None
            elif k == r:
                mark = edge
            else:
                mark = sep
            rb.append(boundary(f"{base}_R{k}", m.c0 - cum_r[:, k], mark))
        lt = kind_type.get(m.kind, 1)
        for k in range(1, r + 1):
            if m.wr[a:b + 1, k - 1].max() <= EPS_W:
                continue
            lane(f"{base}_r{k}", rb[k - 1], rb[k], m.c0 - (cum_r[:, k - 1] + cum_r[:, k]) / 2, lt,
                 2 if m.kind != "path" else 4, ("r", k))
        if m.rsh.max() > 0.05:
            rs = boundary(f"{base}_RS", m.c0 - cum_r[:, r] - m.rsh, None)
            lane(f"{base}_rs", rb[r], rs, m.c0 - cum_r[:, r] - m.rsh / 2, 2, 2, ("rs", 0))
        # left side: opposite lanes, digitised along the samples, travelling backwards
        lb = [rb[0]]
        for k in range(1, l + 1):
            mark = edge if k == l else sep
            lb.append(boundary(f"{base}_L{k}", m.c0 + cum_l[:, k], mark))
        for k in range(1, l + 1):
            if m.wl[a:b + 1, k - 1].max() <= EPS_W:
                continue
            lane(f"{base}_l{k}", lb[k], lb[k - 1], m.c0 + (cum_l[:, k - 1] + cum_l[:, k]) / 2, lt, 3, ("l", k))
        if m.lsh.max() > 0.05:
            outer = m.c0 + cum_l[:, l]
            ls = boundary(f"{base}_LS", outer + m.lsh, None)
            lane(f"{base}_ls", ls, lb[l], outer + m.lsh / 2, 2, 3 if l else 2, ("ls", 0))
        prev_ids = new_ids


def write_rrhd(path, name, frame, models, scene, objects, project_dir, install_dir, cfg, rng, log,
               road_cut=None, include_lanes=True):
    """Write <path> (.rrhd) and the FBX props it references (in the same folder)."""
    hd_pb, hdr_pb = rr_proto.load(install_dir)
    hd = hd_pb.HDMap()
    res = AssetResolver(project_dir, install_dir)
    folder = os.path.dirname(path)
    rel_folder = os.path.relpath(folder, project_dir)  # e.g. Assets/duplicat_env/<name>

    for mid, asset in (("solid", "Assets/Markings/SolidSingleWhite.rrlms"),
                       ("dashed", "Assets/Markings/DashedSingleWhite.rrlms")):
        mk = hd.lane_markings.add()
        mk.id = mid
        mk.asset_path.asset_path = asset

    speeds = {}

    def speed_id(v):
        if v not in speeds:
            sl = hd.speed_limits.add()
            sl.id = f"speed_{v}"
            sl.value = int(v)
            sl.speed_limit_unit = 2  # km/h
            speeds[v] = sl.id
        return speeds[v]

    if include_lanes:
        for i, m in enumerate(models):
            if m.kind in ("route", "road", "path", "track") and m.n >= 3:
                _add_lanes(hd, m, i, speed_id)
        log(f"  RoadRunner HD Map: {len(hd.lanes)} lanes, {len(hd.lane_boundaries)} boundaries")
    else:
        # roads come from the OpenDRIVE import; only footpaths/tracks are added here
        for i, m in enumerate(models):
            if m.kind in ("path", "track") and m.n >= 3:
                _add_lanes(hd, m, i, speed_id)
        log(f"  RoadRunner HD Map: props only (+{len(hd.lanes)} footpath lanes)")

    # --- barriers
    types = {}
    nb = 0
    for rec in scene.records:
        if rec["kind"] != "barrier":
            continue
        asset = BARRIER_ASSETS.get(rec["type"])
        if not asset or not res.exists(asset):
            continue
        if asset not in types:
            t = hd.barrier_types.add()
            t.id = f"barrier_{len(types)}"
            t.extrusion_path.asset_path = asset
            types[asset] = t.id
        b = hd.barriers.add()
        b.id = f"barrier{nb}"
        b.barrier_type_ref.id = types[asset]
        _v3(b.geometry, np.asarray(rec["points"]))
        b.flip_laterally = not rec.get("road_left", True)
        nb += 1

    # --- signs
    stypes = {}
    ns = 0
    for rec in scene.records:
        if rec["kind"] != "sign" or not rec.get("rr_asset"):
            continue
        asset = rec["rr_asset"]
        if asset not in stypes:
            t = hd.sign_types.add()
            t.id = f"sign_{len(stypes)}"
            t.asset_path.asset_path = asset
            stypes[asset] = t.id
        sg = hd.signs.add()
        sg.id = f"sign{ns}"
        sg.sign_type_ref.id = stypes[asset]
        fx, fy = rec["facing"]
        _obb(sg.geometry, rec["center"], (0.01, rec["size"] / 2, rec["size"] / 2), -math.atan2(fy, fx))
        ns += 1

    # --- trees (library assets)
    otypes = {}

    def obj_type(asset):
        if asset not in otypes:
            t = hd.static_object_types.add()
            t.id = f"obj_{len(otypes)}"
            t.asset_path.asset_path = asset
            otypes[asset] = t.id
        return otypes[asset]

    choices = {k: [f"Assets/Props/Trees/{a}.fbx_rrx" for a in v if res.exists(f"Assets/Props/Trees/{a}.fbx_rrx")]
               for k, v in TREE_ASSETS.items()}
    no = 0
    for rec in scene.records:
        if rec["kind"] != "tree" or not choices.get(rec["type"]):
            continue
        asset = choices[rec["type"]][rng.integers(len(choices[rec["type"]]))]
        o = hd.static_objects.add()
        o.id = f"tree{no}"
        o.object_type_ref.id = obj_type(asset)
        x, y, z = rec["pos"]
        H, W = rec["height"], rec["width"]
        _obb(o.geometry, (x, y, z + H / 2), (W / 2, W / 2, H / 2), -rec["yaw"])
        no += 1

    # --- grass clumps: RoadRunner's small bushes, scaled down, at a share of our grass tufts
    grass_assets = [f"Assets/Props/Trees/Bush_Sm0{i}.fbx_rrx" for i in range(1, 7)]
    grass_assets = [a if res.exists(a) else a.replace(".fbx_rrx", ".fbx") for a in grass_assets]
    grass_assets = [a for a in grass_assets if res.exists(a)]
    ng = 0
    if grass_assets:
        for rec in scene.records:
            if rec["kind"] != "grass":
                continue
            o = hd.static_objects.add()
            o.id = f"grass{ng}"
            o.object_type_ref.id = obj_type(grass_assets[ng % len(grass_assets)])
            x, y, z = rec["pos"]
            s = rec["size"]
            _obb(o.geometry, (x, y, z + s / 2), (s * 0.6, s * 0.6, s / 2), -rec["yaw"])
            ng += 1
    log(f"  {ng} grass clumps from the RoadRunner library")

    # --- our own meshes as FBX props; buildings one by one, so each can be
    # moved, turned or deleted in RoadRunner
    ne = 0
    per_building = bool(scene.items)
    items = []
    for nm, parts in sorted(scene.items.items()):
        by_mat = {}
        for mat, V, F, UV in parts:
            by_mat.setdefault(mat, []).append((V, F, UV))
        merged = []
        for mat, ps in by_mat.items():
            offs = np.cumsum([0] + [len(p[0]) for p in ps[:-1]])
            merged.append((mat, np.concatenate([p[0] for p in ps]),
                           np.concatenate([p[1] + o for p, o in zip(ps, offs)]), np.concatenate([p[2] for p in ps])))
        items.append({"name": nm, "category": "Building", "smooth": False, "parts": merged, "_item": True})
    nbld = 0
    for obj in list(objects) + items:
        nm = obj["name"]
        if not nm.startswith(EMBED_PREFIXES) or nm.startswith("Panel_rr"):
            continue
        if per_building and nm.startswith("Building") and obj["category"] == "Building" and not obj.get("_item"):
            continue  # the merged 250 m block; its buildings come one by one below
        if obj.get("_item"):
            nbld += 1
        parts = obj["parts"]
        if road_cut is not None and nm.startswith(("Ground", "Terrain")):
            # RoadRunner builds the roads itself: remove our ground under them so
            # neither the ground mesh nor the photo shows through the road
            cut = []
            for mat, V, F, UV in parts:
                # a cell goes only if its three corners and its centre are on asphalt
                cen = V[F].mean(axis=1)
                on_v = road_cut(V[:, 0], V[:, 1])
                inside = on_v[F].all(axis=1) & road_cut(cen[:, 0], cen[:, 1])
                F2 = F[~inside]
                if len(F2):
                    cut.append((mat, V, F2, UV))
            parts = cut
            if not parts:
                continue
            obj = dict(obj, parts=parts)
        Vall = np.concatenate([p[1] for p in obj["parts"]])
        lo, hi = Vall.min(axis=0), Vall.max(axis=0)
        c = (lo + hi) / 2
        half = np.maximum((hi - lo) / 2, 0.01)
        local = dict(obj, parts=[(mat, V - c, F, UV) for mat, V, F, UV in obj["parts"]])
        fbx_name = f"{nm}.fbx"
        write_fbx(os.path.join(folder, fbx_name), [local], {k: scene.materials[k] for k in
                                                          {p[0] for p in obj["parts"]}}, folder,
                  log=lambda *a: None)
        o = hd.static_objects.add()
        o.id = f"mesh_{nm}"
        o.object_type_ref.id = obj_type(f"{rel_folder}/{fbx_name}")
        _obb(o.geometry, c, half, 0.0)
        ne += 1
    log(f"  {nb} barriers, {ns} library signs, {no} library trees, {ne} embedded meshes "
        f"({nbld} buildings as separate props)")

    header = hdr_pb.Header()
    header.projection.projection = frame.proj_string
    hb = header.SerializeToString()
    with open(path, "wb") as f:
        f.write(_VarintBytes(len(hb)) + hb + hd.SerializeToString())
    return path
