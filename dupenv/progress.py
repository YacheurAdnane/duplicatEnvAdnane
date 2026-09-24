# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Task tracker for the progress panel, and the RAM watchdog.

Every step of a job is a task with its own progress (0..1, or None when the
step can't tell, like the CARLA converter), a short detail line ("tile
45/119") and start/end times. Tasks can run at the same time (downloads run
in parallel). The pipeline sets TRACKER at the start of a job; library code
calls the module functions, which do nothing when no job is tracking.

The watchdog reads /proc/meminfo every second. Above `high` (80 % of the
machine's RAM in use, all programs counted) it closes a gate: parallel
workers wait at `wait_for_ram()` before starting a new download or tile
until usage drops below `low`, so the job slows down instead of freezing the
desktop.
"""
import json
import threading
import time

TRACKER = None
WATCHDOG = None


class Tracker:
    def __init__(self, emit, weights, min_interval=0.4):
        self.emit = emit
        self.weights = dict(weights)
        self.order = [k for k, _ in weights]
        self.tasks = {}
        self.lock = threading.RLock()
        self.min_interval = min_interval
        self._last = 0.0
        self.t0 = time.time()
        for k, _ in weights:
            self.tasks[k] = {"key": k, "label": k, "state": "pending", "frac": 0.0, "detail": "",
                             "t0": None, "t1": None, "parent": None}

    # -------------------------------------------------------------- updates
    def begin(self, key, label=None, parent=None, frac=0.0, detail=""):
        with self.lock:
            t = self.tasks.get(key)
            if t is None:
                t = {"key": key, "parent": parent}
                self.tasks[key] = t
                self.order.append(key)
            t.update(label=label or t.get("label") or key, state="running", frac=frac, detail=detail,
                     t0=time.time(), t1=None)
        self.flush(force=True)

    def update(self, key, frac=None, detail=None, done=None, total=None):
        with self.lock:
            t = self.tasks.get(key)
            if t is None or t["state"] != "running":
                return
            if done is not None and total:
                t["frac"] = min(1.0, done / total)
                if detail is None:
                    detail = f"{done}/{total}"
            elif frac is not None or "frac" not in t:
                t["frac"] = None if frac is None else max(0.0, min(1.0, frac))
            if detail is not None:
                t["detail"] = detail
        self.flush()

    def end(self, key, state="done", detail=None):
        with self.lock:
            t = self.tasks.get(key)
            if t is None:
                return
            if t["t0"] is None:
                t["t0"] = time.time()
            t["state"] = state
            t["t1"] = time.time()
            if state == "done":
                t["frac"] = 1.0
            if detail is not None:
                t["detail"] = detail
        self.flush(force=True)

    # -------------------------------------------------------------- output
    def overall(self):
        tot = sum(self.weights.values())
        got = 0.0
        for k, w in self.weights.items():
            t = self.tasks[k]
            if t["state"] in ("done", "skipped", "failed"):
                got += w
            elif t["state"] == "running":
                got += w * (t["frac"] or 0.0)
        return got / tot if tot else 0.0

    def snapshot(self):
        with self.lock:
            rows = [dict(self.tasks[k]) for k in self.order]
        now = time.time()
        for r in rows:
            if r["t0"] is not None:
                r["elapsed"] = round((r["t1"] or now) - r["t0"], 1)
        snap = {"overall": round(self.overall(), 4), "tasks": rows, "elapsed": round(now - self.t0, 1)}
        if WATCHDOG is not None:
            snap["ram"] = WATCHDOG.status()
        return snap

    def flush(self, force=False):
        now = time.time()
        if not force and now - self._last < self.min_interval:
            return
        self._last = now
        try:
            self.emit(self.snapshot())
        except Exception:
            pass


def begin(key, label=None, parent=None, frac=0.0, detail=""):
    if TRACKER is not None:
        TRACKER.begin(key, label, parent, frac, detail)


def update(key, frac=None, detail=None, done=None, total=None):
    if TRACKER is not None:
        TRACKER.update(key, frac, detail, done, total)


def end(key, state="done", detail=None):
    if TRACKER is not None:
        TRACKER.end(key, state, detail)


class Counter:
    """Thread-safe done/total counter bound to one task."""

    def __init__(self, key, total, what=""):
        self.key, self.total, self.what, self.n = key, max(1, int(total)), what, 0
        self.lock = threading.Lock()

    def tick(self, k=1):
        with self.lock:
            self.n += k
            n = self.n
        update(self.key, done=n, total=self.total, detail=f"{self.what} {n}/{self.total}".strip())


# ------------------------------------------------------------------ RAM

def meminfo():
    """(total bytes, available bytes) of the machine."""
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            vals[k] = int(v.split()[0]) * 1024
            if len(vals) > 20:
                break
    return vals["MemTotal"], vals.get("MemAvailable", vals.get("MemFree", 0))


def own_rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Watchdog(threading.Thread):
    def __init__(self, high=0.80, low=0.72, log=print):
        super().__init__(daemon=True)
        self.high, self.low, self.log = high, low, log
        self.ok = threading.Event()
        self.ok.set()
        self.used = 0.0
        self.total = meminfo()[0]
        self.throttled_s = 0.0
        self._halt = False

    def run(self):
        while not self._halt:
            try:
                total, avail = meminfo()
                self.used = 1.0 - avail / total
            except Exception:
                self.used = 0.0
            if self.ok.is_set() and self.used > self.high:
                self.ok.clear()
                self.log(f"  RAM at {self.used * 100:.0f}% of the PC, pausing new parallel work until it drops")
            elif not self.ok.is_set():
                self.throttled_s += 1.0
                if self.used < self.low:
                    self.ok.set()
                    self.log(f"  RAM back to {self.used * 100:.0f}%, full speed again")
            self._n = getattr(self, "_n", 0) + 1
            if TRACKER is not None and self._n % 2 == 0:
                TRACKER.flush(force=True)  # heartbeat: the page sees the job is alive
            time.sleep(1.0)

    def stop(self):
        self._halt = True
        self.ok.set()

    def status(self):
        return {"used": round(self.used, 3), "total_gb": round(self.total / 2 ** 30, 1),
                "job_gb": round(own_rss() / 2 ** 30, 2), "throttled": not self.ok.is_set()}


def wait_for_ram(max_wait=600):
    """Called by parallel workers before starting a new unit of work."""
    if WATCHDOG is not None and not WATCHDOG.ok.is_set():
        WATCHDOG.ok.wait(max_wait)


def workers(kind="cpu"):
    """How many parallel workers to use. Downloads are network-bound; CPU work
    is limited by cores and by the RAM budget (half the machine)."""
    import os
    if os.environ.get("DUPENV_WORKERS"):
        return max(1, int(os.environ["DUPENV_WORKERS"]))
    cores = os.cpu_count() or 4
    if kind == "net":
        return 12
    total = meminfo()[0]
    per_worker = 1.5 * 2 ** 30  # a terrain/vegetation worker peaks around 1-1.5 GB
    return max(1, min(cores - 2, int(0.5 * total / per_worker)))


def dumps(snap):
    return json.dumps(snap, separators=(",", ":"))
