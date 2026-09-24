# __NAME__

```
carla/__NAME__/            CARLA import package: __NAME__.fbx, __NAME__.xodr, __NAME__.json, textures/
autoware/__NAME__/         lanelet2_map.osm, pointcloud_map.pcd, map_projector_info.yaml
osm/__NAME___corridor.osm  raw OpenStreetMap data of the corridor
preview/                   __NAME__.glb (browser viewer or Blender) and screenshots
roadrunner_project/        RoadRunner project: Scenes/__NAME__.rrscene, Exports/ (CARLA, Lanelet2, OpenDRIVE)
metadata.json              frame origin, projection, spawn point, options used
ue_semantic_tags.py        optional Unreal editor script for semantic classes
```

## CARLA (0.9.15 source build)

```bash
cp -r carla/__NAME__ ~/carla/Import/
cd ~/carla && make import ARGS="--package=__NAME__"
make launch
```

In the editor open `Content/__NAME__/Maps/__NAME__` and press Play, or package it with `make package ARGS="--packages=__NAME__"`. `make import` imports every package in `~/carla/Import`, so remove old ones first.

Then drive it:

```bash
python3 ~/duplicat_env/tools/carla_drive.py .   # from this folder
```

`metadata.json` has a spawn point at the start of the route in CARLA coordinates (`spawn.carla`).

## RoadRunner

Open `roadrunner_project/` in RoadRunner (File > Open Project), then the scene `__NAME__`. Edit what you like and export to CARLA from the Export menu. `Exports/__NAME___carla/` already holds RoadRunner's CARLA export of the untouched scene (FBX + xodr with junctions). If the scene was added to one of your own projects, look there instead.

## Autoware

Use `autoware/__NAME__` as `map_path`. `map_projector_info.yaml` says `projector_type: Local`, so Autoware reads the `local_x`/`local_y` tags, which are the CARLA coordinates with y pointing north (the same thing the CARLA bridge publishes). The point cloud is sampled from the same meshes CARLA renders, so NDT localisation matches the simulated LiDAR.

## Coordinates

Local frame: `metadata.json > proj`, a transverse Mercator centred on the route. A point (x, y, z) in the xodr and lanelet map is (x, -y, z) in CARLA/Unreal.
