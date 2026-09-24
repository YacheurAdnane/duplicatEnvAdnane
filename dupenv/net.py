# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""Small HTTP helper with retries and an on-disk cache (stdlib only)."""
import hashlib
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "duplicat_env/1.0 (digital twin generator for CARLA/Autoware)"
CACHE_DIR = os.environ.get("DUPENV_CACHE") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


# parallel requests allowed per server; more gets 429 "too many requests"
HOST_SLOTS = {"overpass-api.de": 2, "overpass.kumi.systems": 2, "maps.mail.ru": 2,
              "overpass.private.coffee": 2, "data.geopf.fr": 6, "s3.amazonaws.com": 12}
_host_sem = {}
_host_lock = threading.Lock()


def _slot(url):
    host = urllib.parse.urlsplit(url).hostname or ""
    with _host_lock:
        if host not in _host_sem:
            _host_sem[host] = threading.BoundedSemaphore(HOST_SLOTS.get(host, 6))
        return _host_sem[host]


def cached(url, data=None):
    if isinstance(data, dict):
        data = urllib.parse.urlencode(data)
    return os.path.exists(_cache_path(url, data))


def _cache_path(url, data):
    h = hashlib.sha1((url + "|" + (data or "")).encode()).hexdigest()
    return os.path.join(CACHE_DIR, h[:2], h)


def http_get(url, data=None, timeout=180, retries=4, cache=True, log=None):
    """GET (or form POST when `data` is a dict/str) returning bytes.

    Successful responses are cached on disk so that re-running a job does not
    hit the public servers again.
    """
    if isinstance(data, dict):
        data = urllib.parse.urlencode(data)
    path = _cache_path(url, data)
    if cache and os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()

    from .progress import wait_for_ram
    wait_for_ram()
    with _slot(url):
        return _download(url, data, path, timeout, retries, cache, log)


def _download(url, data, path, timeout, retries, cache, log):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data.encode() if data is not None else None,
                headers={"User-Agent": UA, "Accept": "*/*"},
            )
            # hard deadline for the whole download: some servers keep the
            # connection open and trickle (or send nothing), which a plain
            # per-read socket timeout never catches
            t0 = time.monotonic()
            chunks = []
            with urllib.request.urlopen(req, timeout=min(timeout, 60)) as resp:
                while True:
                    part = resp.read(1 << 16)
                    if not part:
                        break
                    chunks.append(part)
                    if time.monotonic() - t0 > timeout:
                        raise TimeoutError(f"download took more than {timeout}s")
            body = b"".join(chunks)
            if cache:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(body)
                os.replace(tmp, path)
            return body
        except urllib.error.HTTPError as e:
            last_err = e
            # 400 means a bad query and retrying will not help, except on IGN's
            # Geoplateforme, where some backends answer "LayerNotDefined" for a
            # layer that exists; the caller handles 504/429 when retries == 1
            flaky = False
            if e.code == 400:
                try:
                    flaky = b"LayerNotDefined" in e.read(4096)
                except Exception:
                    pass
            if (e.code == 400 and not flaky) or retries == 1:
                raise
            if flaky:
                last_err = RuntimeError("IGN answered LayerNotDefined (flaky backend)")
            wait = 10 * (attempt + 1) if e.code in (429, 503, 504) else (1 + attempt if flaky else 3 * (attempt + 1))
        except Exception as e:  # timeouts, resets
            last_err = e
            wait = 3 * (attempt + 1)
        if log:
            log(f"  request failed ({last_err}), retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"HTTP request failed after {retries} attempts: {url} ({last_err})")
