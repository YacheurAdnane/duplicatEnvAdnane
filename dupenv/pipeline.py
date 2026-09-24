# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""End-to-end job: route -> OSM + elevation -> scene -> FBX / XODR / Lanelet2 / PCD."""
import json
import math
import os
import re
import shutil
import time

import numpy as np
from shapely.geometry import LineString

from . import features as ft
from . import groundtex, materials, network, roads, routing
from .geo import runs
from .elevation import Elevation, in_france
from .geo import LocalFrame
from .landcover import LandCover
from .meshlib import Scene
from . import progress as prog
from . import rr_proto
from .osm_pbf import fetch_osm_pbf
from .osmdata import corridor_polygon, fetch_osm
from .writers.fbx import write_fbx
from .writers.glb import select_for_preview, write_glb
from .writers.lanelet2 import write_lanelet2, write_projector_info
from .writers.pcd import sample_objects, voxel_filter, write_pcd
from .writers.xodr import write_xodr

DEFAULTS = {
    "name": "A10_twin",
    "corridor": 150.0,        # metres each side of the route
    "osm_source": "auto",     # auto (Overpass; over 8 km it races the Geofabrik extract) | overpass | geofabrik
    "terrain_res": 4.0,
    "dem": "auto",            # auto | ign | terrarium
    "ign_res": 2.0,
    "ortho": False,           # IGN orthophoto on the terrain instead of land-cover textures
    "ground_style": "landcover",  # landcover (grass/fields/forest floor... per area) | ortho
    "ground_quality": "high",  # low (blended colours) | medium (photo textures) | high (+ grass tufts, rocks)
    "ground_details": True,   # high: 3D grass tufts, flowers and small rocks near the road
    "detail_tri_budget": 2.5e6,  # triangles for those details
    "hq_tree_m": 60.0,        # detailed tree models within this distance of a road
    "lidar_trees": True,      # trees from the IGN LIDAR HD canopy height model (France)
    "bdtopo": True,           # IGN BD TOPO vegetation zones, hedges and building heights (France)
    "ortho_res": 0.5,
    "buildings": True,
    "vegetation": True,
    "tree_spacing": 7.0,
    "max_trees": 20000,
    "barriers": True,
    "infer_guardrails": True,
    "panels": True,
    "infer_panels": True,
    "rail": True,
    "water": True,
    "paths": True,
    "markings": True,
    "dash": 3.0,
    "gap": 10.0,
    "lane_taper": 80.0,
    "network": "auto",        # auto/carla: junctions by CARLA's converter | roadrunner: RoadRunner builds them | simple
    "max_road_len": 1000.0,   # OpenDRIVE roads longer than this are split
    "traffic_lights": True,   # traffic lights at OSM traffic_signals junctions
    "pcd": True,
    "pcd_density": 1.0,
    "pcd_terrain_density": 0.3,
    "pcd_voxel": 0.2,
    "glb": True,
    "hq_textures": True,
    "veg_tri_budget": 2.5e6,  # light tree models (far from the road)
    "hq_tri_budget": 3e6,     # detailed tree models, nearest the road first
    "preview_max_tris": 4e6,  # GLB preview and screenshots
    "rr_preview_max_tris": 3e6,  # ask RoadRunner for an OBJ preview only below this      # procedural asphalt/grass/bark/leaves/facade textures
    "roadrunner": "auto",     # auto (if installed) | true | false: build a RoadRunner project
    "rr_exports": True,       # RoadRunner's own CARLA FBX / Lanelet2 / OpenDRIVE (about 4 min on 50 km)
    "roadrunner_project": "",  # existing RoadRunner project to add the scene to (saves ~2 GB)
    "lanelet_other_roads": False,
    "country": "FR",
    "drive_left": False,
    "seed": 7,
    "ram_high": 0.80,         # above this share of the PC's RAM in use, new parallel work waits
}

# (key, label, weight). Weights follow the measured time of a 56 km cold run.
STAGES = [
    ("prepare", "Route", 1), ("download", "Downloads (OSM, terrain, LIDAR, photo, BD TOPO)", 30),
    ("roads", "Roads and junctions", 12), ("buildings", "Buildings", 3), ("terrain", "Terrain and ground types", 12),
    ("vegetation", "Trees", 5), ("objects", "Barriers, panels, poles", 4), ("fbx", "FBX for CARLA", 8),
    ("xodr", "OpenDRIVE", 1), ("lanelet2", "Lanelet2 for Autoware", 1), ("pcd", "Point cloud map", 5),
    ("preview", "3D preview (GLB)", 4), ("roadrunner", "RoadRunner project", 10), ("screenshots", "Screenshots", 3),
    ("package", "Package", 1),
]


def sanitize_name(name):
    n = re.sub(r"[^A-Za-z0-9_]", "_", name or "Twin").strip("_") or "Twin"
    if not n[0].isalpha():
        n = "M_" + n
    # CARLA drops meshes whose names contain these words
    n = re.sub("(?i)sign", "Sgn", n)
    n = re.sub("(?i)light", "Lght", n)
    return n[:48]


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


class Progress:
    """Stage-level progress on top of progress.Tracker (which also carries the
    parallel sub-tasks, the RAM gauge and the timings)."""

    def __init__(self, cb, log=None, tasks_cb=None):
        self.log = log
        self.cb = cb
        self.cur = None
        self.tasks_cb = tasks_cb
        self.tracker = prog.Tracker(self._emit, [(k, w) for k, _, w in STAGES])
        for k, label, _ in STAGES:
            self.tracker.tasks[k]["label"] = label
        prog.TRACKER = self.tracker

    def _emit(self, snap):
        if self.tasks_cb:
            self.tasks_cb(snap)
        self.cb(self.cur or "prepare", snap["overall"])

    def stage(self, name):
        if self.cur:
            t = self.tracker.tasks[self.cur]
            if self.log:
                self.log(f"  [memory] after {self.cur}: {rss_gb():.1f} GB, {time.time() - t['t0']:.0f} s")
            if t["state"] == "running":
                self.tracker.end(self.cur)
        self.cur = name
        self.tracker.begin(name)

    def sub(self, frac, detail=None):
        self.tracker.update(self.cur, frac=frac, detail=detail)

    def skip(self, name):
        self.tracker.end(name, "skipped")

    def finish(self):
        if self.cur and self.tracker.tasks[self.cur]["state"] == "running":
            self.tracker.end(self.cur)
            self.cur = None


def build_network(osm, corridor, frame, models, rail_models, elev, scene, cfg, log, name, route_xy=None):
    """OpenDRIVE network with junctions (CARLA Osm2Odr) and meshes built from it."""
    import carla
    log("Road network with junctions (CARLA OSM converter)...")
    prog.update("roads", frac=None, detail="CARLA OSM converter (no progress info from it, usually 1-3 min)")
    xodr = network.convert(network.osm_for_network(osm, corridor, frame, route_xy=route_xy, log=log), frame.proj_string,
                           traffic_lights=bool(cfg["traffic_lights"]), log=log)
    prog.update("roads", frac=0.5, detail="splitting long roads, adding elevation")
    xodr, n_split = network.split_long_roads(xodr, float(cfg["max_road_len"]))
    ground0 = ft.Ground(elev, models + rail_models, float(cfg["terrain_res"]))
    xodr = network.add_elevation(xodr, models, ground0, log)
    xodr = xodr.replace('name=""', f'name="{name}"', 1)
    cmap = carla.Map(name, xodr)
    lanes = network.sample_lanes(cmap, xodr)
    n_roads = xodr.count("<road ")
    n_junc = xodr.count("<junction ")
    log(f"  {n_roads} roads, {n_junc} junctions, {n_split} long roads split to <= "
        f"{float(cfg['max_road_len']):.0f} m, {len(lanes)} lanes")
    prog.update("roads", frac=0.7, detail="road, marking and pavement meshes")
    walk_quads, road_union = network.network_meshes(scene, lanes, elev, cfg, log)
    prog.update("roads", frac=0.9, detail="terrain fit under the roads")
    P, D, A, B, T = network.surface_points(lanes, elev)
    extra = [m for m in models if m.kind in ("path", "track")] + rail_models
    ground = ft.Ground.from_network(elev, P, D, A, B, T, float(cfg["terrain_res"]), extra=extra)
    from scipy.spatial import cKDTree
    return {"xodr": xodr, "map": cmap, "lanes": lanes, "ground": ground, "walk_quads": walk_quads, "road_union": road_union,
            "speeds": network.road_speeds(xodr), "surface": cKDTree(P[:, :2])}


def build_roadrunner(out, name, frame, models, scene, objects, tex_dir, cfg, rng, log, ground=None, xodr=None):
    from .roadrunner import RoadRunner, build_scene, export_all, new_project
    from .writers.rrhd import write_rrhd

    existing = (cfg.get("roadrunner_project") or "").strip()
    proj = os.path.abspath(os.path.expanduser(existing)) if existing else os.path.join(out, "roadrunner_project")
    rr = RoadRunner(log=log)
    try:
        if existing:
            if not os.path.isdir(os.path.join(proj, "Assets")):
                raise RuntimeError(f"{proj} is not a RoadRunner project")
            rr.cmd(f"LoadProject(folder_path='{proj}')")
        else:
            new_project(rr, proj)
        assets = os.path.join(proj, "Assets", "duplicat_env", name)
        os.makedirs(assets, exist_ok=True)
        shutil.copytree(tex_dir, os.path.join(assets, "textures"), dirs_exist_ok=True)
        rrhd = os.path.join(assets, f"{name}.rrhd")
        cut = None
        motor = [m for m in models if m.kind in ("route", "road", "track")]
        if xodr is not None and ground is not None:
            def cut(x, y):
                return ground.on_asphalt(np.asarray(x), np.asarray(y), 0.6)
        elif motor:
            from scipy.spatial import cKDTree
            fp, _, _ = ft.surface_samples(motor, "ground", spacing=1.0)
            if len(fp):
                tree = cKDTree(fp[:, :2])

                def cut(x, y, tree=tree):
                    d, _ = tree.query(np.column_stack([x, y]), distance_upper_bound=0.6)
                    return np.isfinite(d)
        write_rrhd(rrhd, name, frame, models, scene, objects, proj, rr.bin, cfg, rng, log, road_cut=cut,
                   include_lanes=xodr is None)
        xodr_rel = None
        if xodr is not None:
            with open(os.path.join(assets, f"{name}.xodr"), "w", encoding="utf-8") as f:
                f.write(xodr)
            xodr_rel = f"duplicat_env/{name}/{name}.xodr"
        build_scene(rr, proj, name, frame, f"duplicat_env/{name}/{name}.rrhd", xodr_rel)
        n_trees = sum(1 for r in scene.records if r["kind"] == "tree")
        # the OBJ export copies every library tree (~50 k triangles each), so it
        # is only worth it for small scenes
        small = cfg.get("_scene_tris", 0) <= float(cfg["rr_preview_max_tris"]) and n_trees <= 300
        if not small:
            log(f"  skipping RoadRunner's OBJ preview ({n_trees} library trees would make it several GB)")
        exports = export_all(rr, proj, name, log, preview=small) if cfg["rr_exports"] else {}
        if not cfg["rr_exports"]:
            log("  RoadRunner exports skipped (option off); export from RoadRunner after editing")
        rr.cmd("SaveProject()", quiet=True)
        log(f"  RoadRunner project: {proj} (scene Scenes/{name}.rrscene)")
        for k, v in exports.items():
            log(f"  RoadRunner export {k}: {os.path.relpath(v, out)}")
    finally:
        rr.close()


ROAD_RANK = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
             "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street", "service"]


def main_road_in_area(osm, area_xy, log):
    """Longest piece of the highest-class road inside the area, in its driving direction."""
    best = None
    for w in osm.ways.values():
        hw = w["tags"].get("highway")
        if hw not in ROAD_RANK or w["tags"].get("area") == "yes":
            continue
        nds = [n for n in w["nodes"] if n in osm.node_xy]
        if len(nds) < 2:
            continue
        line = LineString([osm.node_xy[n] for n in nds]).intersection(area_xy)
        for g in getattr(line, "geoms", [line]):
            if g.geom_type != "LineString" or g.length < 30:
                continue
            key = (-ROAD_RANK.index(hw), g.length)
            if best is None or key > best[0]:
                rev = w["tags"].get("oneway") == "-1"
                best = (key, np.array(g.coords)[::-1] if rev else np.array(g.coords), hw)
    if best is None:
        raise ValueError("no drivable road inside the area")
    log(f"  main road inside the area: {best[2]}, {LineString(best[1]).length:.0f} m (used for the spawn point and checks)")
    return best[1][:, :2]


def fetch_osm_any(cfg, frame, route_xy, ll, radius, log, area_xy=None):
    """OSM data for the corridor. On long routes Overpass (many pieces, public
    servers often busy) races the local Geofabrik extract; the first one to
    finish wins and the other is stopped."""
    import queue
    import threading
    from .osmdata import area_queries
    src = cfg["osm_source"]
    queries, area_ll = None, None
    if area_xy is not None:
        queries = area_queries(frame, area_xy)
        ring = np.array(area_xy.buffer(20).exterior.coords)
        lon, lat = frame.to_lonlat(ring[:, 0], ring[:, 1])
        from shapely.geometry import Polygon
        area_ll = Polygon(np.column_stack([lon, lat])).buffer(0)
    long_job = len(queries) > 2 if queries is not None else LineString(route_xy).length >= 8000
    if src == "geofabrik":
        prog.begin("osm", "OSM from the Geofabrik extract", parent="download")
        data = fetch_osm_pbf(frame, ll, radius, log, task="osm", area_ll=area_ll)
        prog.end("osm")
        return data
    prog.begin("osm", "OSM from Overpass", parent="download")
    if src == "overpass" or not long_job:
        try:
            data = fetch_osm(frame, route_xy, ll, radius, log, task="osm", queries=queries)
            prog.end("osm")
            return data
        except Exception as e:
            prog.end("osm", "failed", str(e)[:60])
            if src == "overpass":
                raise
            log(f"Overpass unavailable ({str(e)[:100]}), using a Geofabrik extract instead")
            prog.begin("osm_pbf", "OSM from the Geofabrik extract", parent="download")
            data = fetch_osm_pbf(frame, ll, radius, log, task="osm_pbf", area_ll=area_ll)
            prog.end("osm_pbf")
            return data
    log("  long route: Overpass and the Geofabrik extract race, the first one done is used")
    prog.begin("osm_pbf", "OSM from the Geofabrik extract", parent="download")
    cancel = threading.Event()
    q = queue.Queue()

    def worker(label, key, fn):
        try:
            q.put((label, key, fn(), None))
        except Exception as e:
            q.put((label, key, None, e))
    threading.Thread(target=worker, daemon=True, args=(
        "Overpass", "osm", lambda: fetch_osm(frame, route_xy, ll, radius, log, cancel=cancel, task="osm", queries=queries))).start()
    threading.Thread(target=worker, daemon=True, args=(
        "Geofabrik", "osm_pbf", lambda: fetch_osm_pbf(frame, ll, radius, log, cancel=cancel, task="osm_pbf", area_ll=area_ll))).start()
    errs = []
    for _ in range(2):
        label, key, data, err = q.get()
        if data is not None:
            cancel.set()
            other = "osm_pbf" if key == "osm" else "osm"
            prog.end(other, "skipped", "the other source was faster")
            prog.end("osm_pbf" if key == "osm_pbf" else "osm")
            log(f"  OSM data from {label}")
            return data
        errs.append(f"{label}: {err}")
        prog.end(key, "failed", str(err)[:60])
    raise RuntimeError("no OSM source worked: " + "; ".join(errs))


def run(out_root, route_lonlat, user_cfg, log=print, progress_cb=lambda s, f: None, tasks_cb=None):
    t0 = time.time()
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (user_cfg or {}).items() if v is not None})
    name = sanitize_name(cfg["name"])
    cfg["name"] = name
    # downloads log from several threads: one lock so lines never mix
    import threading
    out_lock = threading.Lock()

    def locked(fn):
        def f(*a):
            with out_lock:
                return fn(*a)
        return f
    log, progress_cb = locked(log), locked(progress_cb)
    tasks_cb = locked(tasks_cb) if tasks_cb else None
    prog.WATCHDOG = prog.Watchdog(high=float(cfg["ram_high"]), low=float(cfg["ram_high"]) - 0.08, log=log)
    prog.WATCHDOG.start()
    pr = Progress(progress_cb, log, tasks_cb)
    try:
        return _run(out_root, route_lonlat, cfg, name, pr, log, progress_cb, t0)
    finally:
        prog.WATCHDOG.stop()
        pr.tracker.flush(force=True)
        prog.TRACKER = None


def _run(out_root, route_lonlat, cfg, name, pr, log, progress_cb, t0):
    import threading
    rng = np.random.default_rng(int(cfg["seed"]))

    pr.stage("prepare")
    area_mode = bool(cfg.get("area"))
    if area_mode:
        # area mode: route_lonlat is the polygon the user drew, closed back to its start
        ll = np.asarray(route_lonlat, float)
        if len(ll) < 3:
            raise ValueError("an area needs at least 3 points")
        if not np.allclose(ll[0], ll[-1]):
            ll = np.vstack([ll, ll[:1]])
    else:
        ll = np.asarray(routing.clean_route(route_lonlat), float)  # drop out-and-back spikes
    if len(ll) < 2:
        raise ValueError("route needs at least 2 points")
    lat0 = 0.5 * (ll[:, 1].min() + ll[:, 1].max())
    lon0 = 0.5 * (ll[:, 0].min() + ll[:, 0].max())
    frame = LocalFrame(lat0, lon0)
    rx, ry = frame.to_xy(ll[:, 0], ll[:, 1])
    route_xy = np.column_stack([rx, ry])
    length = LineString(route_xy).length
    ext = max(np.ptp(rx), np.ptp(ry))
    area_xy = None
    if area_mode:
        from shapely.geometry import Polygon
        area_xy = Polygon(route_xy).buffer(0)
        if area_xy.geom_type == "MultiPolygon":  # an outline that crosses itself: keep the biggest part
            area_xy = max(area_xy.geoms, key=lambda g: g.area)
        if area_xy.is_empty or area_xy.area < 100:
            raise ValueError("the area is empty or crosses itself")
        log(f"Area: {area_xy.area / 1e6:.2f} km2, outline {length / 1000:.2f} km, extent {ext / 1000:.2f} km, "
            f"origin lat {lat0:.6f} lon {lon0:.6f}")
    else:
        log(f"Route: {length / 1000:.2f} km, extent {ext / 1000:.2f} km, origin lat {lat0:.6f} lon {lon0:.6f}")
    if ext > 30000:
        raise ValueError("route extent > 30 km: Unreal Engine 4 worlds are limited to about +-20 km around the origin")
    if ext > 16000:
        log("WARNING: extent > 16 km, far parts of the map may show float precision jitter in UE4")
    corridor = area_xy if area_mode else corridor_polygon(route_xy, float(cfg["corridor"]))

    out = os.path.join(out_root, name)
    carla_dir = os.path.join(out, "carla", name)
    tex_dir = os.path.join(carla_dir, "textures")
    aw_dir = os.path.join(out, "autoware", name)
    shutil.rmtree(tex_dir, ignore_errors=True)  # textures of an earlier run of this name
    for d in (carla_dir, tex_dir, aw_dir, os.path.join(out, "osm"), os.path.join(out, "preview")):
        os.makedirs(d, exist_ok=True)
    if area_mode:
        with open(os.path.join(out, "area.geojson"), "w") as f:
            json.dump({"type": "Feature", "properties": {"name": name},
                       "geometry": {"type": "Polygon", "coordinates": [ll.tolist()]}}, f)
    else:
        with open(os.path.join(out, "route.geojson"), "w") as f:
            json.dump({"type": "Feature", "properties": {"name": name},
                       "geometry": {"type": "LineString", "coordinates": ll.tolist()}}, f)

    pr.stage("download")
    log("Downloading OSM, elevation, LIDAR, aerial photo and BD TOPO in parallel...")
    radius = float(cfg["corridor"]) + 60
    tiles = ft.terrain_tiles(corridor, float(cfg["terrain_res"]))
    france = in_france(frame, corridor) if cfg["dem"] == "auto" else cfg["dem"] == "ign"
    want_lc = cfg["ground_style"] == "landcover" or cfg["lidar_trees"]
    results, errors = {}, {}

    def job(key, fn, label=None):
        if label:
            prog.begin(key, label, parent="download")
        try:
            results[key] = fn()
            if label:
                prog.end(key)
        except Exception as e:
            errors[key] = e
            prog.end(key, "failed", str(e)[:80])

    jobs = [("osm", lambda: fetch_osm_any(cfg, frame, route_xy, ll, radius, log, area_xy=area_xy), None),
            ("dem", lambda: Elevation(frame, corridor, source=cfg["dem"], ign_res=float(cfg["ign_res"]), log=log),
             "Terrain height (IGN RGE ALTI)" if france else "Terrain height (AWS Terrain Tiles)")]
    gq = cfg["ground_quality"] if cfg["ground_style"] == "landcover" and not cfg["ortho"] else "low"
    if groundtex.QUALITY.get(gq, {}).get("res"):
        jobs.append(("groundtex", lambda: groundtex.prefetch(gq, log), "Ground textures (ambientCG, CC0)"))
    if want_lc:
        log("Land cover (LIDAR HD trees, infrared photo, BD TOPO, OSM landuse)...")
        jobs.append(("landcover", lambda: LandCover(frame, corridor, None, france, cfg, log, tiles=tiles), None))
    threads = [threading.Thread(target=job, args=j, daemon=True) for j in jobs]
    for th in threads:
        th.start()
    kids = ("osm", "osm_pbf", "dem", "lidar", "bdtopo", "photo", "groundtex")
    while any(th.is_alive() for th in threads):
        # the download bar is the mean of its parallel parts
        rows = [pr.tracker.tasks[k] for k in kids if k in pr.tracker.tasks
                and pr.tracker.tasks[k]["state"] in ("running", "done")]
        if rows:
            pr.sub(sum(r["frac"] or 0.0 for r in rows) / len(rows),
                   ", ".join(f"{r['label'].split(' (')[0]} {100 * (r['frac'] or 0):.0f}%" for r in rows
                             if r["state"] == "running"))
        for th in threads:
            th.join(0.5)
    pr.sub(1.0, "")
    for key in ("osm", "dem"):
        if key in errors:
            raise errors[key]
    osm, elev = results["osm"], results["dem"]
    landcover = results.get("landcover")
    if "landcover" in errors:
        log(f"  land cover failed ({str(errors['landcover'])[:100]}), using OSM only")
    osm_path = os.path.join(out, "osm", f"{name}_corridor.osm")
    osm.write_osm_xml(osm_path, bounds=(ll[:, 0].min() - 0.01, ll[:, 1].min() - 0.01,
                                        ll[:, 0].max() + 0.01, ll[:, 1].max() + 0.01))
    if landcover is not None:
        landcover.add_osm(osm)
    if area_mode:
        # the rest of the pipeline wants one route (spawn point, route model,
        # checks): the main road inside the area
        route_xy = main_road_in_area(osm, area_xy, log)
        rlon, rlat = frame.to_lonlat(route_xy[:, 0], route_xy[:, 1])
        with open(os.path.join(out, "route.geojson"), "w") as f:
            json.dump({"type": "Feature", "properties": {"name": name, "note": "main road inside the area"},
                       "geometry": {"type": "LineString", "coordinates": np.column_stack([rlon, rlat]).tolist()}}, f)
    if cfg["ortho"]:
        cfg["ground_style"] = "ortho"
    if cfg["ground_style"] == "ortho" and elev.source != "ign":
        log("  orthophoto texture is only available in France (IGN), using land-cover textures")
        cfg["ortho"] = False
        cfg["ground_style"] = "landcover"

    pr.stage("roads")
    log("Building road models...")
    route = roads.build_route_model(route_xy, osm, elev, cfg, log)
    others = roads.build_way_models(osm, elev, route, corridor, cfg, log)
    models = [route] + others
    roads.harmonize_levels(models, log=log)
    scene = Scene(chunk=250.0)
    materials.register(scene, tex_dir, textured=bool(cfg["hq_textures"]))
    scene.material("M_Asphalt", (0.22, 0.22, 0.23))
    scene.material("M_Marking_White", (0.95, 0.95, 0.95))
    scene.material("M_Sidewalk", (0.60, 0.58, 0.55))
    scene.material("M_Gravel", (0.52, 0.47, 0.40))
    ft.register_common_materials(scene)
    rail_models = ft.build_rail(scene, osm, elev, corridor, cfg, log) if cfg["rail"] else []
    res = float(cfg["terrain_res"])
    net = None
    if cfg["network"] not in ("simple", "roadrunner") and network.carla_available():
        try:
            net = build_network(osm, corridor, frame, models, rail_models, elev, scene, cfg, log, name, route_xy=route_xy)
        except Exception as e:
            if cfg["network"] == "carla":
                raise
            import traceback
            log(f"  WARNING: junction network failed ({str(e)[:120]}), using the simple road model")
            log("  " + traceback.format_exc().strip().splitlines()[-3].strip())
            net = None
    if net is not None:
        ground = net["ground"]
        # footpaths and tracks keep their own mesh, cut where they cross asphalt
        walk = list(net["walk_quads"])
        for m in others:
            if m.kind not in ("path", "track"):
                continue
            # footpaths join the pavements in one merged surface; tracks keep a strip,
            # both are dropped where they lie on the asphalt
            keep = ~ground.on_asphalt(m.x, m.y, 1.0)
            for a, b in runs(keep):
                if b - a < 3:
                    continue
                sub = m.subset(a, b)
                if m.kind == "track":
                    roads.road_meshes(scene, sub, elev, cfg)
                    continue
                Lm, Rm = sub.surface_edges()
                A, B = sub.point(Lm), sub.point(Rm)
                walk += [[A[i] - [0, 0, 0.12], A[i + 1] - [0, 0, 0.12], B[i + 1] - [0, 0, 0.12],
                          B[i] - [0, 0, 0.12]] for i in range(len(A) - 1)]
        W = np.array(walk)
        if len(W):
            ground_z = elev.sample_xy(W[:, :, 0].mean(1), W[:, :, 1].mean(1))
            high = W[:, :, 2].mean(1) - ground_z > 2.5  # footbridges merge on their own level
            n1, _ = network.merged_surface(scene, W[~high], "Road_Sidewalk", "M_Sidewalk", raise_z=0.12,
                                           subtract=net["road_union"])
            n2, _ = network.merged_surface(scene, W[high], "Road_Sidewalk", "M_Sidewalk", raise_z=0.12)
            log(f"  pavements and footpaths merged into {n1 + n2} surfaces")
    else:
        for i, m in enumerate(models):
            roads.road_meshes(scene, m, elev, cfg)
            if i % 50 == 0:
                pr.sub(i / max(1, len(models)))
        ground = ft.Ground(elev, models + rail_models, res)
    ridx = ft.RoadIndex(models)

    pr.stage("buildings")
    blocked = None
    buildings_union = None
    if cfg["buildings"]:
        log("Buildings...")
        buildings_union = ft.build_buildings(scene, osm, ground, corridor, cfg, log, landcover=landcover)
        blocked = buildings_union
    if cfg["water"]:
        water = ft.build_water(scene, osm, ground, corridor, log)
        if water is not None:
            blocked = water if blocked is None else blocked.union(water)

    pr.stage("terrain")
    if results.get("groundtex") and cfg["hq_textures"]:
        pr.sub(0.0, "ground textures")
        n = groundtex.apply(scene, tex_dir, gq, results["groundtex"], log)
        log(f"  {n} photo ground textures ({gq} quality, ambientCG CC0)")
    log("Building terrain...")
    scatter = ft.build_terrain(scene, ground, corridor, frame, cfg, tex_dir, log, pr.sub,
                               landcover=landcover if cfg["ground_style"] == "landcover" else None,
                               buildings=buildings_union)

    pr.stage("vegetation")
    if scatter:
        ft.build_ground_details(scene, scatter, ground, cfg, tex_dir, np.random.default_rng(int(cfg["seed"]) + 3), log)
    if cfg["vegetation"]:
        log("Vegetation...")
        ft.build_vegetation(scene, osm, ground, corridor, blocked, cfg, rng, log,
                            landcover=landcover if cfg["lidar_trees"] else None, tiles=tiles)

    pr.stage("objects")
    barrier_pts = np.zeros((0, 2))
    if cfg["barriers"]:
        log("Barriers...")
        barrier_pts = ft.build_barriers(scene, osm, ground, corridor, cfg, log)
    if cfg["infer_guardrails"]:
        if net is not None:
            network.guardrails_from_network(scene, net["lanes"], net["speeds"], ground, barrier_pts, log)
        else:
            ft.infer_guardrails(scene, models, ground, barrier_pts, cfg, log)
    if cfg["panels"]:
        log("Panels, poles, lamps...")
        pf = ft.build_point_objects(scene, osm, ground, ridx, corridor, tex_dir, cfg, log)
        ft.build_gantries(scene, osm, ground, ridx, pf, corridor, log)
        ft.build_exit_panels(scene, osm, route, ground, pf, corridor, cfg, log)

    tris, per = scene.stats()
    cfg["_scene_tris"] = tris
    log(f"Scene: {len(scene.objects)} meshes, {tris / 1e6:.2f} M triangles")
    log("  " + ", ".join(f"{k} {v / 1000:.0f}k" for k, v in sorted(per.items(), key=lambda kv: -kv[1])))
    objects = scene.finalize()

    # RoadRunner runs as its own program: build its project in the background
    # while the FBX, OpenDRIVE, Lanelet2, point cloud and preview are written
    rr_on = cfg["roadrunner"]
    if rr_on == "auto":
        rr_on = rr_proto.find_install() is not None
    rr_thread = None
    if rr_on in (True, "true", "True", 1):
        def rr_job():
            pr.tracker.begin("roadrunner")
            prog.update("roadrunner", frac=None, detail="headless RoadRunner: project, import, save, exports")
            log("Building the RoadRunner project (headless RoadRunner, in parallel)...")
            try:
                build_roadrunner(out, name, frame, models, scene, objects, tex_dir, cfg, rng_rr, log, ground=ground,
                                 xodr=net["xodr"] if net is not None else None)
                pr.tracker.end("roadrunner")
            except Exception as e:  # keep the other outputs even if RoadRunner fails
                log(f"  RoadRunner step failed: {e}")
                pr.tracker.end("roadrunner", "failed", str(e)[:80])
        rng_rr = np.random.default_rng(int(cfg["seed"]) + 1)
        rr_thread = threading.Thread(target=rr_job, daemon=True)
        rr_thread.start()
    else:
        pr.skip("roadrunner")

    pr.stage("fbx")
    fbx_path = os.path.join(carla_dir, f"{name}.fbx")
    log("Writing FBX...")
    write_fbx(fbx_path, objects, scene.materials, carla_dir, log)
    log(f"  {fbx_path} ({os.path.getsize(fbx_path) / 1e6:.0f} MB)")

    pr.stage("xodr")
    xodr_path = os.path.join(carla_dir, f"{name}.xodr")
    minx, miny, maxx, maxy = corridor.bounds
    if net is not None:
        with open(xodr_path, "w", encoding="utf-8") as f:
            f.write(net["xodr"])
    else:
        write_xodr(xodr_path, name, frame, models, (minx, miny, maxx, maxy))
    with open(os.path.join(carla_dir, f"{name}.json"), "w") as f:
        json.dump({"maps": [{"name": name, "source": f"./{name}.fbx", "use_carla_materials": True,
                             "xodr": f"./{name}.xodr"}], "props": []}, f, indent=3)
    log(f"  {xodr_path}")

    pr.stage("lanelet2")
    ll2 = os.path.join(aw_dir, "lanelet2_map.osm")
    if net is not None:
        speeds = net["speeds"]
        n_ll = network.write_lanelet2(ll2, net["lanes"], frame, speed_of=lambda ln: speeds.get(ln.road, 50))
    else:
        n_ll = write_lanelet2(ll2, route, frame,
                              extra_models=others if cfg["lanelet_other_roads"] else ())
    write_projector_info(os.path.join(aw_dir, "map_projector_info.yaml"), frame)
    log(f"  {ll2} ({n_ll} lanelets)")

    pr.stage("pcd")
    if cfg["pcd"]:
        log("Sampling point cloud map...")
        P, I = sample_objects(objects, float(cfg["pcd_density"]), rng, float(cfg["pcd_terrain_density"]))
        P, I = voxel_filter(P, I, float(cfg["pcd_voxel"]))
        n = write_pcd(os.path.join(aw_dir, "pointcloud_map.pcd"), P, I)
        log(f"  pointcloud_map.pcd ({n / 1e6:.2f} M points)")

    pr.stage("preview")
    if cfg["glb"]:
        log("Writing GLB preview...")
        prev_objs, dropped, ptris = select_for_preview(objects, float(cfg["preview_max_tris"]))
        if dropped:
            log(f"  preview limited to {ptris / 1e6:.1f} M triangles ({dropped} vegetation meshes left out)")
        write_glb(os.path.join(out, "preview", f"{name}.glb"), prev_objs, scene.materials, carla_dir)

    if rr_thread is not None:
        pr.stage("screenshots")
        pr.sub(0.0, "waiting for RoadRunner to finish")
        rr_thread.join()
    if pr.cur != "screenshots":
        pr.stage("screenshots")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = os.path.join(here, "tools", "ue_semantic_tags.py")
    if os.path.exists(tpl):
        with open(tpl) as f:
            src = f.read().replace("__PACKAGE__", name).replace("__MAP__", name)
        with open(os.path.join(out, "ue_semantic_tags.py"), "w") as f:
            f.write(src)
    start = route.point(route.c0 - route.wr[:, :1].sum(1) / 2)[5]
    yaw_enu = float(np.degrees(route.hdg[5]))
    if net is not None:
        # snap the spawn to a real driving lane of the network, facing the route direction
        import carla
        best = None
        for k in (5, 10, 20):
            w = net["map"].get_waypoint(carla.Location(float(route.x[k]), float(-route.y[k]), float(route.z[k])))
            if w is None:
                continue
            wy = -w.transform.rotation.yaw
            if math.cos(math.radians(wy - np.degrees(route.hdg[k]))) > 0.5:
                best = w
                break
        if best is not None:
            loc = best.transform.location
            start = np.array([loc.x, -loc.y, loc.z])
            yaw_enu = -best.transform.rotation.yaw
    meta = {
        "name": name, "origin": {"lat": lat0, "lon": lon0}, "proj": frame.proj_string,
        "route_length_m": float(route.s[-1]), "config": cfg,
        "spawn": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2]) + 0.5,
                  "yaw_deg_enu": yaw_enu,
                  "carla": {"x": float(start[0]), "y": float(-start[1]), "z": float(start[2]) + 0.5,
                            "yaw": -yaw_enu}},
        "goal": {"x": float(route.x[-10]), "y": float(route.y[-10]), "z": float(route.z[-10])},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    readme_src = os.path.join(here, "tools", "OUTPUT_README.md")
    if os.path.exists(readme_src):
        with open(readme_src) as f:
            txt = f.read().replace("__NAME__", name)
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write(txt)
    here_tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    if not cfg["glb"]:
        pr.skip("screenshots")
    if cfg["glb"]:
        import subprocess
        import sys
        renders = [None]
        rr_root = os.path.expanduser(cfg.get("roadrunner_project") or "") or os.path.join(out, "roadrunner_project")
        rr_obj = os.path.join(rr_root, "Exports", "preview", f"{name}_roadrunner.obj")
        if os.path.exists(rr_obj):
            renders.append(rr_obj)
        for src in renders:
            args = [sys.executable, os.path.join(here_tools, "render_preview.py"), out] + ([src] if src else [])
            r = subprocess.run(args, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                log("  screenshots: " + ", ".join(line for line in r.stdout.split() if line.endswith(".png")))
            else:
                log("  screenshots skipped (needs moderngl + a GPU): " + r.stderr.strip()[-150:])

    pr.stage("package")
    pr.finish()
    progress_cb("done", 1.0)
    log(f"Done in {time.time() - t0:.0f} s -> {out}")
    return out
