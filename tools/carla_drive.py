#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Drive the imported twin in CARLA with a LiDAR, after `make import`.

    make launch   (or a packaged CARLA containing the map)
    python3 tools/carla_drive.py output/<name> [--speed 110] [--semantic]

Loads the map, spawns an ego car at the start of the route with autopilot,
attaches a 64-channel LiDAR and prints how many points hit each object class.
"""
import argparse
import collections
import json
import os
import time

import carla
import numpy as np

TAG_NAMES = {0: "None", 1: "Road", 2: "Sidewalk", 3: "Building", 4: "Wall", 5: "Fence", 6: "Pole",
             7: "TrafficLight", 8: "TrafficSign", 9: "Vegetation", 10: "Terrain", 11: "Sky",
             12: "Pedestrian", 13: "Rider", 14: "Car", 15: "Truck", 16: "Bus", 17: "Train",
             18: "Motorcycle", 19: "Bicycle", 20: "Static", 21: "Dynamic", 22: "Other", 23: "Water",
             24: "RoadLine", 25: "Ground", 26: "Bridge", 27: "RailTrack", 28: "GuardRail"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--speed", type=float, default=110.0, help="km/h for the autopilot")
    ap.add_argument("--semantic", action="store_true", help="semantic LiDAR (needs ue_semantic_tags.py)")
    ap.add_argument("--traffic", type=int, default=20, help="NPC vehicles")
    a = ap.parse_args()
    name = os.path.basename(os.path.realpath(a.out_dir))
    meta = json.load(open(os.path.join(a.out_dir, "metadata.json")))

    client = carla.Client(a.host, 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    if name not in world.get_map().name:
        print(f"loading map {name}...")
        world = client.load_world(name)
    bl = world.get_blueprint_library()
    sp = meta["spawn"]["carla"]
    loc = carla.Location(sp["x"], sp["y"], sp["z"])
    wp = world.get_map().get_waypoint(loc)
    tf = wp.transform
    tf.location.z += 0.8
    ego = world.spawn_actor(bl.find("vehicle.tesla.model3"), tf)
    tm = client.get_trafficmanager()
    tm.set_desired_speed(ego, a.speed)
    ego.set_autopilot(True, tm.get_port())

    npcs = []
    spawn = [w.transform for w in world.get_map().generate_waypoints(30.0) if w.road_id == wp.road_id]
    for t in spawn[5:5 + a.traffic]:
        t.location.z += 0.8
        v = world.try_spawn_actor(np.random.choice(bl.filter("vehicle.*")), t)
        if v:
            v.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(v, a.speed * np.random.uniform(0.8, 1.05))
            npcs.append(v)

    kind = "sensor.lidar.ray_cast_semantic" if a.semantic else "sensor.lidar.ray_cast"
    lb = bl.find(kind)
    lb.set_attribute("channels", "64")
    lb.set_attribute("range", "120")
    lb.set_attribute("points_per_second", "1200000")
    lb.set_attribute("rotation_frequency", "10")
    lb.set_attribute("upper_fov", "2")
    lb.set_attribute("lower_fov", "-24.8")
    lidar = world.spawn_actor(lb, carla.Transform(carla.Location(z=1.9)), attach_to=ego)
    stats = collections.Counter()
    last = {"n": 0}

    def on_scan(data):
        if a.semantic:
            arr = np.frombuffer(data.raw_data, dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                                                               ("cos", "f4"), ("idx", "u4"), ("tag", "u4")]))
            stats.clear()
            stats.update(TAG_NAMES.get(int(t), str(t)) for t in arr["tag"])
        last["n"] = len(data)

    lidar.listen(on_scan)
    spec = world.get_spectator()
    print(f"ego on road {wp.road_id} lane {wp.lane_id}, {len(npcs)} NPCs. Ctrl+C to stop.")
    try:
        k = 0
        while True:
            t = ego.get_transform()
            spec.set_transform(carla.Transform(t.location + carla.Location(z=3) - t.get_forward_vector() * 9,
                                               carla.Rotation(pitch=-12, yaw=t.rotation.yaw)))
            time.sleep(0.05)
            k += 1
            if k % 40 == 0:
                v = ego.get_velocity()
                kmh = 3.6 * (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
                msg = f"speed {kmh:5.1f} km/h  lidar {last['n']:6d} pts"
                if stats:
                    msg += "  " + ", ".join(f"{k}:{v}" for k, v in stats.most_common(6))
                print(msg)
    except KeyboardInterrupt:
        pass
    finally:
        lidar.destroy()
        for v in npcs:
            v.destroy()
        ego.destroy()


if __name__ == "__main__":
    main()
