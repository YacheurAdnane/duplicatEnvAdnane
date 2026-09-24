#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Quick test without Unreal import: CARLA builds a road-only world from the
generated .xodr (grey procedural mesh, no buildings/trees), then spawns a car
on the route with autopilot. Good to check the road network in 1 minute.

    ./CarlaUE4.sh            # any CARLA 0.9.15 server
    python3 tools/carla_xodr_world.py output/<name>
"""
import json
import os
import sys
import time

import carla


def main(out_dir):
    name = os.path.basename(os.path.realpath(out_dir))
    xodr = open(os.path.join(out_dir, "carla", name, f"{name}.xodr")).read()
    meta = json.load(open(os.path.join(out_dir, "metadata.json")))
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=500.0, wall_height=0.0,
        additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True)
    world = client.generate_opendrive_world(xodr, params)
    time.sleep(2)
    sp = meta["spawn"]["carla"]
    tf = carla.Transform(carla.Location(sp["x"], sp["y"], sp["z"] + 1.0), carla.Rotation(yaw=sp["yaw"]))
    wp = world.get_map().get_waypoint(tf.location)
    tf = wp.transform
    tf.location.z += 1.0
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    car = world.spawn_actor(bp, tf)
    tm = client.get_trafficmanager()
    car.set_autopilot(True, tm.get_port())
    tm.set_desired_speed(car, 110.0)
    spec = world.get_spectator()
    print(f"spawned on road {wp.road_id} lane {wp.lane_id}; Ctrl+C to stop")
    try:
        while True:
            t = car.get_transform()
            spec.set_transform(carla.Transform(t.location + carla.Location(z=40) - t.get_forward_vector() * 30,
                                               carla.Rotation(pitch=-45, yaw=t.rotation.yaw)))
            time.sleep(0.05)
    except KeyboardInterrupt:
        car.destroy()


if __name__ == "__main__":
    main(sys.argv[1])
