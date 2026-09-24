# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Unreal Editor Python script: give the imported meshes their CARLA semantic tags.

CARLA's import moves every mesh that is not a road, road marking or sidewalk
into Static/Terrain/<map>, so buildings, trees, panels... would all be tagged
"Terrain" in the semantic camera and semantic LiDAR. CARLA tags a mesh by the
folder it lives in (/Game/<package>/Static/<Tag>/...), so this script moves
each mesh to the folder of its tag, keeping references valid.

Plain LiDAR (sensor.lidar.ray_cast) hits every mesh anyway; this step only
matters for sensor.lidar.ray_cast_semantic and the segmentation camera.

How to run (after `make import`):
  1. make launch, then Edit > Plugins: enable "Python Editor Script Plugin"
     and "Editor Scripting Utilities", restart the editor.
  2. File > Execute Python Script... and pick this file.
  3. Open the map, File > Save All.
"""
import re

import unreal

PACKAGE = "__PACKAGE__"
MAP = "__MAP__"

TAGS = [
    ("Building", "Building"), ("Vegetation", "Vegetation"), ("Panel", "TrafficSign"),
    ("Pole", "Pole"), ("GuardRail", "GuardRail"), ("Wall", "Wall"), ("Fence", "Fence"),
    ("Bridge", "Bridge"), ("RailTrack", "RailTrack"), ("Water", "Water"), ("Static", "Static"),
]

src = f"/Game/{PACKAGE}/Static/Terrain/{MAP}"
tools = unreal.AssetToolsHelpers.get_asset_tools()
moves = []
for path in unreal.EditorAssetLibrary.list_assets(src, recursive=True, include_folder=False):
    asset = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(asset, unreal.StaticMesh):
        continue
    name = asset.get_name()
    for prefix, tag in TAGS:
        if re.search(rf"(^|_){prefix}_", name):
            dest = f"/Game/{PACKAGE}/Static/{tag}/{MAP}"
            moves.append(unreal.AssetRenameData(asset, dest, name))
            break

unreal.log(f"duplicat_env: moving {len(moves)} meshes to their semantic folders")
if moves:
    tools.rename_assets(moves)  # same as a Content Browser move: fixes references
unreal.EditorAssetLibrary.save_directory(f"/Game/{PACKAGE}", only_if_is_dirty=True, recursive=True)
unreal.log("duplicat_env: done. Open the map and File > Save All.")
