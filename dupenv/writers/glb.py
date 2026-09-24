# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Binary glTF (GLB) writer, used for the browser preview and for Blender."""
import json
import os
import struct

import numpy as np

from ..meshlib import face_normals, vertex_normals


def select_for_preview(objects, max_tris):
    """Keep everything except vegetation, then as many vegetation meshes as fit."""
    def tris(o):
        return sum(len(p[2]) for p in o["parts"])
    base = [o for o in objects if o["category"] != "Vegetation"]
    used = sum(tris(o) for o in base)
    veg = [o for o in objects if o["category"] == "Vegetation"]
    keep = []
    for o in veg:
        t = tris(o)
        if used + t <= max_tris:
            keep.append(o)
            used += t
    return base + keep, len(veg) - len(keep), used


def write_glb(path, objects, materials, tex_base_dir, max_tex=1024):
    """Binary chunks go to a temporary file as they are produced, so memory
    stays at one mesh at a time whatever the scene size."""
    from PIL import Image
    import io

    tmp_bin = path + ".bin.tmp"
    binf = open(tmp_bin, "wb")
    offset = [0]
    buffer_views, accessors, meshes, nodes = [], [], [], []
    gl_materials, textures, images = [], [], []
    mat_index = {}

    def add_view(data, target=None):
        pad = (-len(data)) % 4
        bv = {"buffer": 0, "byteOffset": offset[0], "byteLength": len(data)}
        if target:
            bv["target"] = target
        binf.write(data + b"\0" * pad)
        offset[0] += len(data) + pad
        buffer_views.append(bv)
        return len(buffer_views) - 1

    def add_accessor(arr, ctype, typ, target, minmax=False):
        v = add_view(arr.tobytes(), target)
        acc = {"bufferView": v, "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for name, info in materials.items():
        r, g, b = info["color"]
        m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, 1.0],
                                                    "metallicFactor": 0.0, "roughnessFactor": 0.9},
             "doubleSided": False}
        tex = info.get("texture")
        if tex:
            p = os.path.join(tex_base_dir, tex)
            if os.path.exists(p):
                im = Image.open(p).convert("RGB")
                if max(im.size) > max_tex:
                    im.thumbnail((max_tex, max_tex))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
                v = add_view(buf.getvalue())
                images.append({"bufferView": v, "mimeType": "image/jpeg"})
                textures.append({"source": len(images) - 1, "sampler": 0})
                m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(textures) - 1}
                m["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        gl_materials.append(m)
        mat_index[name] = len(gl_materials) - 1

    for obj in objects:
        prims = []
        for mat, V, F, UV in obj["parts"]:
            if obj["smooth"]:
                Vx, Nx, UVx, Fx = V, vertex_normals(V, F), UV, F
            else:  # unshare vertices for flat shading
                Vx = V[F].reshape(-1, 3)
                Nx = np.repeat(face_normals(V, F), 3, axis=0)
                UVx = UV[F].reshape(-1, 2)
                Fx = np.arange(len(Vx)).reshape(-1, 3)
            # glTF is Y-up: (x, y, z)_enu -> (x, z, -y)
            pos = np.column_stack([Vx[:, 0], Vx[:, 2], -Vx[:, 1]]).astype(np.float32)
            nrm = np.column_stack([Nx[:, 0], Nx[:, 2], -Nx[:, 1]]).astype(np.float32)
            uv = np.column_stack([UVx[:, 0], 1.0 - UVx[:, 1]]).astype(np.float32)
            idx = Fx.astype(np.uint32).ravel()
            prims.append({
                "attributes": {
                    "POSITION": add_accessor(pos, 5126, "VEC3", 34962, True),
                    "NORMAL": add_accessor(nrm, 5126, "VEC3", 34962),
                    "TEXCOORD_0": add_accessor(uv, 5126, "VEC2", 34962),
                },
                "indices": add_accessor(idx.reshape(-1, 1), 5125, "SCALAR", 34963),
                "material": mat_index[mat],
            })
        meshes.append({"name": obj["name"], "primitives": prims})
        nodes.append({"name": obj["name"], "mesh": len(meshes) - 1})

    gltf = {
        "asset": {"version": "2.0", "generator": "duplicat_env"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes, "materials": gl_materials,
        "accessors": accessors, "bufferViews": buffer_views,
        "buffers": [{"byteLength": offset[0]}],
    }
    if textures:
        gltf["textures"] = textures
        gltf["images"] = images
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    binf.close()
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blen = offset[0]
    total = 12 + 8 + len(js) + 8 + blen
    with open(path, "wb") as f, open(tmp_bin, "rb") as b:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", blen, 0x004E4942))
        while True:
            chunk = b.read(1 << 24)
            if not chunk:
                break
            f.write(chunk)
    os.remove(tmp_bin)
    return path
