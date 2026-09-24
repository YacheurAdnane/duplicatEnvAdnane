# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Drive RoadRunner (R2023b+) headless through its gRPC API.

Starts `AppRoadRunner --nodesktop`, creates the project, imports the
generated RoadRunner HD Map (needs the RoadRunner Scene Builder add-on), saves
the scene and exports CARLA (FBX + xodr) and Lanelet2 from RoadRunner.
"""
import os
import shutil
import socket
import subprocess
import time

from . import rr_proto


def _free_port(start=35720):
    for p in range(start, start + 200, 2):
        with socket.socket() as s1, socket.socket() as s2:
            try:
                s1.bind(("127.0.0.1", p))
                s2.bind(("127.0.0.1", p + 1))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port for RoadRunner")


class RoadRunner:
    def __init__(self, install=None, log=print, timeout=240):
        self.bin = install or rr_proto.find_install()
        if not self.bin:
            raise RuntimeError("RoadRunner not found (set ROADRUNNER_BIN)")
        self.log = log
        self.port = _free_port()
        self.logfile = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "cache", "roadrunner.log"), "a")
        exe = os.path.join(self.bin, "AppRoadRunner")
        self.proc = subprocess.Popen([exe, "--nodesktop", "--apiPort", str(self.port),
                                      "--cosimPort", str(self.port + 1)],
                                     stdout=self.logfile, stderr=subprocess.STDOUT, cwd=self.bin)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("RoadRunner exited at start (license?) see cache/roadrunner.log")
            try:
                self.cmd("GetApplicationInfo()", quiet=True)
                return
            except RuntimeError:
                time.sleep(2)
        raise RuntimeError("RoadRunner API did not start")

    STEPS = {"NewProject": "new project (copies the asset library)", "ChangeWorldSettings": "world origin",
             "Import": "importing roads and props", "SaveScene": "saving the scene", "Export": "exporting"}

    def cmd(self, command, quiet=False, timeout=3600):
        from . import progress as prog
        name = command.split("(")[0]
        if name in self.STEPS:
            what = self.STEPS[name]
            if name == "Export" and "format_name='" in command:
                what += " " + command.split("format_name='")[1].split("'")[0]
            prog.update("roadrunner", frac=None, detail=what)
        exe = os.path.join(self.bin, "CmdRoadRunnerApi")
        r = subprocess.run([exe, "--serverAddress", f"localhost:{self.port}", command],
                           capture_output=True, text=True, timeout=timeout, cwd=self.bin)
        if r.returncode != 0:
            raise RuntimeError(f"RoadRunner: {command[:80]}... failed: {(r.stderr or r.stdout).strip()[-300:]}")
        if not quiet:
            self.log(f"  RoadRunner: {command.split('(')[0]} ok")
        return r.stdout

    def close(self):
        try:
            self.cmd("Exit()", quiet=True, timeout=60)
        except Exception:
            pass
        try:
            self.proc.wait(30)
        except Exception:
            self.proc.kill()
        self.logfile.close()


def new_project(rr, project_dir):
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    rr.cmd(f"NewProject(folder_path='{project_dir}' asset_libraries[0]='RoadRunner_Asset_Library')")


def build_scene(rr, project_dir, name, frame, rrhd_rel, xodr_rel=None):
    rr.cmd(f"ChangeWorldSettings(world_origin.latitude='{frame.lat0:.9f}' "
           f"world_origin.longitude='{frame.lon0:.9f}')")
    if xodr_rel:
        # roads and junctions exactly as in the OpenDRIVE we give to CARLA
        o = "open_drive_settings"
        rr.cmd(f"Import(file_path='{xodr_rel}' format_name='OpenDRIVE' {o}.import_signals.value='true' "
               f"{o}.import_props.value='false')")
    s = "roadrunner_hd_map_settings.build_settings"
    clear = "false" if xodr_rel else "true"
    rr.cmd(f"Import(file_path='{rrhd_rel}' format_name='RoadRunner HD Map' "
           f"{s}.clear_scene_of_existing_data.value='{clear}' {s}.fit_cross_sections.value='true' "
           f"{s}.detect_asphalt_surfaces.value='true' {s}.auto_detect_bridges.enable.value='true' "
           f"{s}.enable_overlap_groups.enable.value='true' "
           f"{s}.fix_inconsistent_lane_connections.value='true')")
    rr.cmd(f"SaveScene(file_path='{name}.rrscene')")


def export_all(rr, project_dir, name, log, preview=True):
    out = {}
    jobs = [("CARLA Filmbox", f"{name}_carla/{name}.fbx"),
            ("Lanelet2 Map", f"{name}_lanelet2/lanelet2_map.osm"),
            ("OpenDRIVE", f"{name}.xodr")]
    if preview:
        jobs.append(("Wavefront", f"preview/{name}_roadrunner.obj"))
    for fmt, rel in jobs:
        try:
            rr.cmd(f"Export(file_path='{rel}' format_name='{fmt}')")
            out[fmt] = os.path.join(project_dir, "Exports", rel)
        except RuntimeError as e:
            log(f"  RoadRunner export {fmt} failed: {e}")
    return out
