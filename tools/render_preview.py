#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Render screenshots of a generated twin (offscreen OpenGL, no window).

    python3 tools/render_preview.py output/<name>

Writes into output/<name>/preview/:
  map.png      the selected route and corridor on OpenStreetMap tiles
  aerial.png   oblique view over the middle of the route
  street.png   driver's view at the start of the route
Needs `pip install moderngl` and a GPU driver with EGL (NVIDIA works).
"""
import io
import json
import math
import os
import struct
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UA = "duplicat_env/1.0 (preview renderer)"

VS = """
#version 330
uniform mat4 mvp;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
out vec3 v_nrm; out vec2 v_uv; out vec3 v_pos;
void main() { v_nrm = in_nrm; v_uv = in_uv; v_pos = in_pos; gl_Position = mvp * vec4(in_pos, 1.0); }
"""
FS = """
#version 330
uniform sampler2D tex; uniform bool use_tex; uniform vec3 color; uniform vec3 eye;
uniform vec3 sun; uniform vec3 sky; uniform float fog;
in vec3 v_nrm; in vec2 v_uv; in vec3 v_pos;
out vec4 f;
void main() {
  vec3 base = use_tex ? texture(tex, v_uv).rgb : color;
  vec3 n = normalize(v_nrm);
  if (dot(n, eye - v_pos) < 0.0) n = -n;
  float d = max(dot(n, sun), 0.0);
  float hemi = 0.5 + 0.5 * n.z;
  vec3 c = base * (0.30 + 0.25 * hemi + 0.75 * d);
  float t = 1.0 - exp(-distance(eye, v_pos) * fog);
  f = vec4(mix(c, sky, clamp(t, 0.0, 1.0)), 1.0);
}
"""


def load_glb(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    js = json.loads(b[20:20 + jl])
    bin_off = 20 + jl + 8
    binary = b[bin_off:]

    def view(i):
        bv = js["bufferViews"][i]
        return binary[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]

    def acc(i):
        a = js["accessors"][i]
        dt = {5126: np.float32, 5125: np.uint32}[a["componentType"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        return np.frombuffer(view(a["bufferView"]), dtype=dt).reshape(-1, n)

    images = [Image.open(io.BytesIO(view(im["bufferView"]))).convert("RGB") for im in js.get("images", [])]
    prims = []
    for mesh in js["meshes"]:
        for p in mesh["primitives"]:
            pos = acc(p["attributes"]["POSITION"])
            nrm = acc(p["attributes"]["NORMAL"])
            uv = acc(p["attributes"]["TEXCOORD_0"]).copy()
            idx = acc(p["indices"]).ravel()
            # glTF Y-up back to ENU Z-up
            pos = np.column_stack([pos[:, 0], -pos[:, 2], pos[:, 1]])
            nrm = np.column_stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]])
            uv[:, 1] = 1.0 - uv[:, 1]
            mat = js["materials"][p["material"]]
            pbr = mat["pbrMetallicRoughness"]
            tex = pbr.get("baseColorTexture", {}).get("index")
            img = images[js["textures"][tex]["source"]] if tex is not None else None
            prims.append((mesh["name"], pos, nrm, uv, idx, pbr["baseColorFactor"][:3], img))
    return prims


def load_obj(path):
    """Wavefront OBJ (RoadRunner export, Z up) -> primitives like load_glb."""
    folder = os.path.dirname(path)
    mats, cur = {}, None
    mtl = None
    V, T, N = [], [], []
    faces = {}
    for line in open(path, errors="ignore"):
        if line.startswith("v "):
            V.append([float(a) for a in line.split()[1:4]])
        elif line.startswith("vt "):
            T.append([float(a) for a in line.split()[1:3]])
        elif line.startswith("vn "):
            N.append([float(a) for a in line.split()[1:4]])
        elif line.startswith("usemtl"):
            cur = line.split(None, 1)[1].strip()
        elif line.startswith("mtllib"):
            mtl = os.path.join(folder, line.split(None, 1)[1].strip())
        elif line.startswith("f "):
            idx = [tuple(int(x) - 1 if x else -1 for x in (t.split("/") + ["", ""])[:3]) for t in line.split()[1:]]
            fl = faces.setdefault(cur, [])
            for k in range(1, len(idx) - 1):
                fl.append((idx[0], idx[k], idx[k + 1]))
    if mtl and os.path.exists(mtl):
        name = None
        for line in open(mtl, errors="ignore"):
            if line.startswith("newmtl"):
                name = line.split(None, 1)[1].strip()
                mats[name] = {"kd": (0.7, 0.7, 0.7), "map": None}
            elif line.startswith("Kd") and name:
                mats[name]["kd"] = tuple(float(a) for a in line.split()[1:4])
            elif line.startswith("map_Kd") and name:
                mats[name]["map"] = os.path.join(folder, line.split(None, 1)[1].strip())
    V, T, N = np.array(V), np.array(T) if T else np.zeros((1, 2)), np.array(N) if N else np.zeros((1, 3))
    prims, cache = [], {}
    for mname, fl in faces.items():
        f = np.array(fl).reshape(-1, 3)
        pos = V[f[:, 0]]
        uv = T[f[:, 1]] if T.shape[0] > 1 else np.zeros((len(f), 2))
        nrm = N[f[:, 2]] if N.shape[0] > 1 else np.tile([0, 0, 1.0], (len(f), 1))
        m = mats.get(mname, {"kd": (0.7, 0.7, 0.7), "map": None})
        img = None
        if m["map"] and os.path.exists(m["map"]):
            if m["map"] not in cache:
                im = Image.open(m["map"]).convert("RGB")
                im.thumbnail((1024, 1024))
                cache[m["map"]] = im
            img = cache[m["map"]]
        prims.append((mname or "mesh", pos, nrm, uv, np.arange(len(pos), dtype=np.uint32), m["kd"], img))
    return prims


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov, aspect, near, far):
    t = 1 / math.tan(math.radians(fov) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = t / aspect, t
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


class Renderer:
    def __init__(self, prims, size=(1600, 900), ss=2):
        import moderngl
        self.W, self.H, self.ss = size[0], size[1], ss
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.fbo = self.ctx.framebuffer(self.ctx.texture((self.W * ss, self.H * ss), 4),
                                        self.ctx.depth_renderbuffer((self.W * ss, self.H * ss)))
        self.origin = np.concatenate([p[1][::50] for p in prims]).astype(np.float64).mean(axis=0)
        self.items = []
        texcache = {}
        for name, pos, nrm, uv, idx, col, img in prims:
            data = np.hstack([(pos - self.origin).astype("f4"), nrm.astype("f4"), uv.astype("f4")])
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(idx.astype("u4").tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")], ibo)
            tex = None
            if img is not None:
                key = id(img)
                if key not in texcache:
                    t = self.ctx.texture(img.size, 3, img.tobytes())
                    t.build_mipmaps()
                    t.repeat_x = t.repeat_y = True
                    t.anisotropy = 8.0
                    texcache[key] = t
                tex = texcache[key]
            self.items.append((name, vao, tex, col))

    def render(self, eye, target, fov=60, far=4000, fog=0.0012, hide=()):
        import moderngl
        eye = np.asarray(eye, float) - self.origin
        target = np.asarray(target, float) - self.origin
        mvp = perspective(fov, self.W / self.H, 0.5, far) @ look_at(eye, target)
        self.fbo.use()
        sky = (0.66, 0.77, 0.88)
        self.fbo.clear(*sky, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog["eye"].value = tuple(eye)
        sun = np.array([0.45, -0.35, 0.82])
        self.prog["sun"].value = tuple(sun / np.linalg.norm(sun))
        self.prog["sky"].value = sky
        self.prog["fog"].value = fog
        for name, vao, tex, col in self.items:
            if name.startswith(hide):
                continue
            self.prog["use_tex"].value = tex is not None
            self.prog["color"].value = tuple(col)
            if tex is not None:
                tex.use(0)
                self.prog["tex"].value = 0
            vao.render()
        img = Image.frombytes("RGBA", self.fbo.size, self.fbo.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
        return img.convert("RGB").resize((self.W, self.H), Image.LANCZOS)


# ---------------------------------------------------------------- map snapshot

def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def render_map(coords, out_png, corridor_m=None, size=(1200, 800), area=None):
    """OSM map with the route and its corridor, or with the area polygon (area mode)."""
    ll = np.asarray(area if area is not None else coords)
    for z in range(18, 5, -1):
        xs, ys = zip(*[_tile_xy(lo, la, z) for lo, la in ll])
        if (max(xs) - min(xs)) * 256 < size[0] * 0.75 and (max(ys) - min(ys)) * 256 < size[1] * 0.75:
            break
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    x0, y0 = cx * 256 - size[0] / 2, cy * 256 - size[1] / 2
    img = Image.new("RGB", size, (230, 230, 230))
    for tx in range(int(x0 // 256), int((x0 + size[0]) // 256) + 1):
        for ty in range(int(y0 // 256), int((y0 + size[1]) // 256) + 1):
            url = f"https://tile.openstreetmap.org/{z}/{tx}/{ty}.png"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                tile = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=30).read())).convert("RGB")
                img.paste(tile, (int(tx * 256 - x0), int(ty * 256 - y0)))
            except Exception:
                pass
    pts = [(_tile_xy(lo, la, z)[0] * 256 - x0, _tile_xy(lo, la, z)[1] * 256 - y0) for lo, la in ll]
    over = Image.new("RGBA", size, (0, 0, 0, 0))
    if area is not None:
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        over.paste((31, 111, 235, 60), mask=mask)
        d = ImageDraw.Draw(over)
        d.line(pts + [pts[0]], fill=(232, 89, 12, 255), width=5, joint="curve")
        ll = np.asarray(coords)  # then the main road on top
        pts = [(_tile_xy(lo, la, z)[0] * 256 - x0, _tile_xy(lo, la, z)[1] * 256 - y0) for lo, la in ll]
        corridor_m = None
    if corridor_m:
        # draw the corridor as an opaque mask first, then blend it once
        mpp = 156543.03 * math.cos(math.radians(ll[:, 1].mean())) / 2 ** z
        w = max(int(2 * corridor_m / mpp), 2)
        mask = Image.new("L", size, 0)
        dm = ImageDraw.Draw(mask)
        dm.line(pts, fill=255, width=w, joint="curve")
        for px, py in (pts[0], pts[-1]):
            dm.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2], fill=255)
        over.paste((31, 111, 235, 60), mask=mask)
    d = ImageDraw.Draw(over)
    d.line(pts, fill=(232, 89, 12, 255), width=6, joint="curve")
    for (px, py), c in ((pts[0], (26, 127, 55)), (pts[-1], (207, 34, 46))):
        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=c + (255,), outline=(255, 255, 255, 255), width=3)
    img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([size[0] - 250, size[1] - 22, size[0], size[1]], fill=(255, 255, 255))
    d.text((size[0] - 245, size[1] - 18), "(c) OpenStreetMap contributors", fill=(60, 60, 60))
    img.save(out_png)
    return out_png


def main(out_dir, glb=None):
    name = os.path.basename(os.path.realpath(out_dir))
    meta = json.load(open(os.path.join(out_dir, "metadata.json")))
    coords = json.load(open(os.path.join(out_dir, "route.geojson")))["geometry"]["coordinates"]
    prev = os.path.join(out_dir, "preview")
    os.makedirs(prev, exist_ok=True)
    if not os.path.exists(os.path.join(prev, "map.png")):
        area_f = os.path.join(out_dir, "area.geojson")
        area = json.load(open(area_f))["geometry"]["coordinates"][0] if os.path.exists(area_f) else None
        render_map(coords, os.path.join(prev, "map.png"), meta["config"].get("corridor"), area=area)
        print("map.png")
    glb = glb or os.path.join(prev, f"{name}.glb")
    limit = 700e6 if glb.endswith(".obj") else 1.5e9
    if os.path.getsize(glb) > limit:
        print(f"skip: {os.path.basename(glb)} is {os.path.getsize(glb) / 1e9:.1f} GB, too big to render safely")
        return
    prims = load_obj(glb) if glb.endswith(".obj") else load_glb(glb)
    tag = "_roadrunner" if glb.endswith(".obj") else ""
    r = Renderer(prims)
    cams = {}
    _render = r.render

    def render_named(key, eye, target, **kw):
        cams[key] = {"eye": [float(v) for v in eye], "target": [float(v) for v in target], **kw}
        return _render(eye, target, **kw)
    from dupenv.geo import LocalFrame, resample
    fr = LocalFrame(meta["origin"]["lat"], meta["origin"]["lon"])
    ll = np.asarray(coords)
    x, y = fr.to_xy(ll[:, 0], ll[:, 1])
    P, _ = resample(np.column_stack([x, y]), 2.0)
    allv = np.concatenate([p[1][::7] for p in prims])  # a subsample is enough for ground height
    from scipy.spatial import cKDTree
    tree = cKDTree(allv[:, :2])

    def ground_z(px, py):
        _, j = tree.query([px, py], k=20)
        return float(np.min(allv[j, 2]))

    # street view: 30 m into the route, eyes at 1.5 m, looking 60 m ahead
    i0 = min(len(P) - 1, 15)
    i1 = min(len(P) - 1, i0 + 30)
    sp = meta["spawn"]
    e = np.array([sp["x"], sp["y"], 0.0])
    k = int(np.argmin(np.hypot(P[:, 0] - e[0], P[:, 1] - e[1])))
    i0 = min(k + 5, len(P) - 2)
    i1 = min(i0 + 30, len(P) - 1)
    t = P[i1] - P[i0]
    t /= np.linalg.norm(t)
    nrm = np.array([-t[1], t[0]])
    eye2 = P[i0] - nrm * 1.8
    z0 = sp["z"] - 0.5
    street = render_named("street", [eye2[0], eye2[1], z0 + 1.5], [P[i1][0] - nrm[1] * 0, P[i1][1], z0 + 1.2], fov=70, fog=0.004)
    street.save(os.path.join(prev, f"street{tag}.png"))
    print(f"street{tag}.png")
    # aerial: behind and above the middle of the route
    mid = P[len(P) // 2]
    L = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    dist = max(120.0, min(900.0, L * 0.55))
    d = P[-1] - P[0]
    d /= max(np.linalg.norm(d), 1e-6)
    side = np.array([-d[1], d[0]])
    eye = mid - d * dist * 0.6 + side * dist * 0.55
    zc = ground_z(*mid)
    aerial = render_named("aerial", [eye[0], eye[1], zc + dist * 0.55], [mid[0], mid[1], zc], fov=50, fog=0.0006)
    aerial.save(os.path.join(prev, f"aerial{tag}.png"))
    print(f"aerial{tag}.png")
    # close-up: low oblique view over a third of the route, where trees and roofs read well
    p3 = P[len(P) // 3]
    z3 = ground_z(*p3)
    eye3 = p3 - d * 150 + side * 110
    close = render_named("closeup", [eye3[0], eye3[1], z3 + 65], [p3[0], p3[1], z3], fov=55, fog=0.002)
    close.save(os.path.join(prev, f"closeup{tag}.png"))
    print(f"closeup{tag}.png")
    json.dump(cams, open(os.path.join(prev, f"cameras{tag}.json"), "w"), indent=1)
    if tag:
        return
    # ground views: where two ground types meet near the road, and the verge with its grass tufts
    ground = [(id(p[6]), p[1][::3]) for p in prims if p[0].startswith("Ground_") and p[6] is not None]
    route_tree = cKDTree(P)
    tall = [p[1][::5] for p in prims if p[0].startswith(("Building", "Vegetation_tree", "Vegetation_broad",
                                                           "Vegetation_conif", "Wall", "Fence"))]
    tall_tree = cKDTree(np.concatenate(tall)[:, :2]) if tall else None

    def clear(q, r=30.0):
        return tall_tree is None or not tall_tree.query_ball_point(q[:2], r)
    if ground:
        pts = np.concatenate([g[1] for g in ground])
        lab = np.concatenate([np.full(len(g[1]), k) for k, g in enumerate(ground)])
        gtree = cKDTree(pts[:, :2])
        best = None
        for q in pts[:: max(1, len(pts) // 20000)]:
            dr, _ = route_tree.query(q[:2])
            if not 12 < dr < 45 or not clear(q):
                continue
            near = lab[gtree.query_ball_point(q[:2], 6.0)]
            if len(np.unique(near)) >= 2:
                sc = min(np.mean(near == near[0]), 1 - np.mean(near == near[0]))
                if best is None or sc > best[0]:
                    best = (sc, q)
        if best is not None:
            q = best[1]
            _, j = route_tree.query(q[:2])
            road = P[j]
            v = (q[:2] - road) / max(np.linalg.norm(q[:2] - road), 1e-6)
            eye = q[:2] - v * 14
            img = render_named("transition", [eye[0], eye[1], q[2] + 7], [q[0], q[1], q[2]], fov=60, fog=0.002)
            img.save(os.path.join(prev, "transition.png"))
            print("transition.png")
    tufts = [p[1] for p in prims if p[0].startswith("Vegetation_grass")]
    if tufts:
        tp = np.concatenate(tufts)
        d, _ = route_tree.query(tp[:, :2])
        cand = tp[(d > 3) & (d < 9)]
        cand = np.array([c for c in cand[:: max(1, len(cand) // 3000)] if clear(c, 15.0)]) if len(cand) else cand
        if len(cand):
            q = cand[len(cand) // 2]
            _, j = route_tree.query(q[:2])
            road = P[j]
            v = (q[:2] - road) / max(np.linalg.norm(q[:2] - road), 1e-6)
            tvec = np.array([-v[1], v[0]])
            eye = road + v * 1.0 - tvec * 3.0
            zq = ground_z(*q[:2])
            img = render_named("verge", [eye[0], eye[1], zq + 1.6], [q[0] + tvec[0] * 4, q[1] + tvec[1] * 4, zq], fov=65, fog=0.003)
            img.save(os.path.join(prev, "verge.png"))
            print("verge.png")
    json.dump(cams, open(os.path.join(prev, "cameras.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
