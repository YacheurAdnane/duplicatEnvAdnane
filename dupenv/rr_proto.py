# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Python bindings for the RoadRunner HD Map protobuf schema.

The .proto files ship with RoadRunner (bin/glnxa64/Proto). They are compiled
with protoc into cache/rrproto the first time, so nothing from MathWorks is
copied into this project.
"""
import glob
import importlib
import os
import subprocess
import sys

from .net import CACHE_DIR

INSTALL_GLOBS = ["/usr/local/RoadRunner_R20*/bin/glnxa64", "/opt/RoadRunner*/bin/glnxa64",
                 "C:/Program Files/RoadRunner R20*/bin/win64"]


def find_install():
    env = os.environ.get("ROADRUNNER_BIN")
    if env and os.path.isdir(env):
        return env
    hits = sorted(h for g in INSTALL_GLOBS for h in glob.glob(g))
    return hits[-1] if hits else None


def load(install=None):
    """Return (hd_map_pb2, hd_map_header_pb2) modules."""
    install = install or find_install()
    if install is None:
        raise RuntimeError("RoadRunner not found (set ROADRUNNER_BIN to its bin/glnxa64 folder)")
    out = os.path.join(CACHE_DIR, "rrproto")
    marker = os.path.join(out, "mathworks", "scenario", "scene", "hd", "hd_map_pb2.py")
    if not os.path.exists(marker):
        os.makedirs(out, exist_ok=True)
        proto_root = os.path.join(install, "Proto")
        files = sorted(glob.glob(os.path.join(proto_root, "mathworks/scenario/scene/hd/*.proto"))
                       + glob.glob(os.path.join(proto_root, "mathworks/scenario/common/*.proto")))
        rel = [os.path.relpath(f, proto_root) for f in files]
        subprocess.run(["protoc", f"-I{proto_root}", f"--python_out={out}"] + rel, check=True)
        for root, dirs, _ in os.walk(out):
            for d in dirs:
                open(os.path.join(root, d, "__init__.py"), "a").close()
        open(os.path.join(out, "mathworks", "__init__.py"), "a").close()
    if out not in sys.path:
        sys.path.insert(0, out)
    hd = importlib.import_module("mathworks.scenario.scene.hd.hd_map_pb2")
    hdr = importlib.import_module("mathworks.scenario.scene.hd.hd_map_header_pb2")
    return hd, hdr
