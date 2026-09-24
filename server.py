#!/usr/bin/env python3
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Local web server for the route picker GUI.

    python3 server.py            # then open the printed URL (default http://127.0.0.1:8777)
"""
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dupenv import pipeline, routing  # noqa: E402  (pipeline: defaults only; jobs run in a child process)

WEB = os.path.join(ROOT, "web")
OUT = os.path.join(ROOT, "output")
JOBS = {}
RUN_LOCK = threading.Lock()   # one generation at a time (public APIs are shared)
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/xml", ".xodr")
mimetypes.add_type("application/xml", ".osm")


def list_files(folder):
    files = []
    for root, _, names in os.walk(folder):
        if os.path.basename(root) == "textures":
            files.append({"path": os.path.relpath(root, OUT) + "/", "size": sum(
                os.path.getsize(os.path.join(root, n)) for n in names), "count": len(names)})
            continue
        for n in sorted(names):
            p = os.path.join(root, n)
            files.append({"path": os.path.relpath(p, OUT), "size": os.path.getsize(p)})
    return sorted(files, key=lambda f: f["path"])


def _mem_limit_prefix():
    """systemd-run scope with a RAM ceiling, so a huge job is stopped instead of
    freezing the desktop. Empty list when systemd-run is not usable."""
    if os.environ.get("DUPENV_NO_MEMLIMIT") or not shutil.which("systemd-run"):
        return []
    try:
        total_kb = int(open("/proc/meminfo").readline().split()[1])
    except (OSError, ValueError, IndexError):
        return []
    # hard stop only near the top: the job's own watchdog already pauses new
    # parallel work when the PC reaches 80 % RAM in use
    frac = float(os.environ.get("DUPENV_MEM_FRACTION", "0.9"))
    limit_mb = int(total_kb * frac / 1024)
    prefix = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={limit_mb}M",
              "-p", "MemorySwapMax=0"]
    try:
        subprocess.run(prefix + ["true"], check=True, capture_output=True, timeout=20)
    except Exception:
        return []
    return prefix


def run_job(job_id, coords, options):
    job = JOBS[job_id]

    def log(msg):
        job["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
        print(msg, flush=True)

    job["status"] = "queued"
    with RUN_LOCK:
        job["status"] = "running"
        tmpdir = tempfile.mkdtemp(prefix="dupenv_job_")
        cpath, opath = os.path.join(tmpdir, "coords.json"), os.path.join(tmpdir, "options.json")
        with open(cpath, "w") as f:
            json.dump(coords, f)
        with open(opath, "w") as f:
            json.dump(options, f)
        prefix = _mem_limit_prefix()
        if prefix:
            log(f"job runs with a memory ceiling of {prefix[5].split('=')[1]} (no swap)")
        cmd = prefix + [sys.executable, "-u", os.path.join(ROOT, "cli.py"), "--out", OUT,
                        "--coords-json", cpath, "--options-json", opath]
        tail = []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
            out_dir = None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("@@LOG "):
                    log(line[6:])
                elif line.startswith("@@PROG "):
                    _, st, fr = line.split(" ", 2)
                    job["stage"], job["progress"] = st, float(fr)
                elif line.startswith("@@TASKS "):
                    try:
                        job["tasks"] = json.loads(line[8:])
                        job["tasks_at"] = time.time()
                    except ValueError:
                        pass
                elif line.startswith("@@OUT "):
                    out_dir = line[6:]
                elif line.strip() and "warnings.warn" not in line and "UserWarning" not in line:
                    tail.append(line)
                    tail[:] = tail[-40:]
            rc = proc.wait()
            if rc == 0 and out_dir:
                job["out"] = os.path.relpath(out_dir, OUT)
                job["files"] = list_files(out_dir)
                job["status"] = "done"
            else:
                if rc in (-9, 137) or any("Killed" in t for t in tail):
                    log("ERROR: the job hit the memory ceiling and was stopped. Use a shorter route, "
                        "a smaller corridor, or fewer trees (Terrain, trees, lines > Max trees).")
                else:
                    log(f"ERROR: job failed (exit code {rc})")
                job["log"].extend(tail[-15:])
                job["status"] = "error"
        except Exception as e:
            log("ERROR: " + str(e))
            job["log"].append(traceback.format_exc())
            job["status"] = "error"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        if "/files/" in self.path and not ctype.startswith(("text/html", "image/")):
            self.send_header("Content-Disposition", f'inline; filename="{os.path.basename(path)}"')
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = urllib.parse.unquote(u.path)
        if p in ("/", "/index.html"):
            return self._file(os.path.join(WEB, "index.html"))
        if p == "/viewer.html":
            return self._file(os.path.join(WEB, "viewer.html"))
        if p == "/api/defaults":
            return self._json(pipeline.DEFAULTS)
        if p == "/api/capabilities":
            from dupenv import rr_proto
            rr = rr_proto.find_install()
            return self._json({"roadrunner": os.path.basename(os.path.dirname(os.path.dirname(rr))) if rr else None})
        if p.startswith("/api/job/"):
            job = JOBS.get(p.rsplit("/", 1)[-1])
            if not job:
                return self._json({"error": "no such job"}, 404)
            since = int(urllib.parse.parse_qs(u.query).get("since", ["0"])[0])
            j = {k: v for k, v in job.items() if k != "log"}
            j["now"] = time.time()
            j["log"] = job["log"][since:]
            j["log_len"] = len(job["log"])
            return self._json(j)
        if p == "/api/current":
            if not JOBS:
                return self._json({"job": None})
            job = max(JOBS.values(), key=lambda j: j["started"])
            return self._json({"job": job["id"], "status": job["status"]})
        if p == "/api/outputs":
            items = []
            if os.path.isdir(OUT):
                for n in sorted(os.listdir(OUT)):
                    meta = os.path.join(OUT, n, "metadata.json")
                    if os.path.exists(meta):
                        with open(meta) as f:
                            m = json.load(f)
                        items.append({"name": n, "created": m.get("created"),
                                      "length_m": m.get("route_length_m"), "files": list_files(os.path.join(OUT, n))})
            return self._json(items)
        if p.startswith("/docs/"):  # README images, also used by the page's "?" help
            docs = os.path.join(ROOT, "docs")
            full = os.path.normpath(os.path.join(docs, p[len("/docs/"):]))
            if not full.startswith(docs):
                return self._json({"error": "forbidden"}, 403)
            return self._file(full)
        if p.startswith("/files/"):
            rel = os.path.normpath(p[len("/files/"):])
            full = os.path.join(OUT, rel)
            if not os.path.abspath(full).startswith(os.path.abspath(OUT)):
                return self._json({"error": "forbidden"}, 403)
            return self._file(full)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        if self.path == "/api/route":
            try:
                coords, dist, info = routing.route(data["waypoints"])
                return self._json({"coords": coords, "distance": dist, "info": info})
            except Exception as e:
                return self._json({"error": str(e)}, 502)
        if self.path == "/api/generate":
            coords = data.get("coords")
            if not coords or len(coords) < 2:
                return self._json({"error": "no route"}, 400)
            job_id = uuid.uuid4().hex[:10]
            JOBS[job_id] = {"id": job_id, "status": "queued", "stage": "", "progress": 0.0,
                            "log": [], "out": None, "files": [], "started": time.time()}
            threading.Thread(target=run_job, args=(job_id, coords, data.get("options", {})), daemon=True).start()
            return self._json({"job": job_id})
        return self._json({"error": "not found"}, 404)


def main():
    port = int(os.environ.get("PORT", "8777"))
    for p in range(port, port + 20):  # skip ports already used by other programs
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {port}-{port + 19}")
    url = f"http://127.0.0.1:{port}/"
    print(f"duplicat_env GUI running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
