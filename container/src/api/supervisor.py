"""BackendSupervisor — one isolated uvicorn subprocess per project's backend.py.

Design decision (see Architect run 56726fc6 on Kanban card 39e5bf3b...):
subprocess isolation, not importlib-namespaced import. UX-Proto's whole
purpose is running AI-authored code, so a broken backend.py (bad syntax,
infinite loop, sys.exit()) is the *normal* case — it must only take down
that one project's subprocess, never the shared FastAPI worker serving the
dashboard/WS hub for every other project.

Port pool 30021-30120 (AW_APP_PORT + 20000, avoids collision with every
other custom app sharing aw-sandbox's network namespace — see memory
aw-app-builder-shared-netns-backend-port). Lazy spawn on first proxied
request, idle reaper after 15 minutes with no requests.

Error contract (Architect finding #3): a dead/never-started project
backend must surface as HTTP 500 + JSON, never a bare 502/503/504 — Caddy's
_splash_fallback_block intercepts exactly {502,503,504} server-wide and
swaps in the awserv "app starting…" splash, which would silently mask a
broken project as a platform outage instead of a project bug.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time

import httpx

PORT_RANGE = range(30021, 30121)
IDLE_TIMEOUT_SECONDS = 15 * 60


class BackendSupervisor:
    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}
        self._ports: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        # Guards all three dicts above, and serializes spawn/kill so two
        # concurrent calls for the same slug (e.g. a proxied request racing
        # a hot_reload_backend) can't both decide the backend is missing and
        # each spawn their own subprocess. ensure_running/reload/reap_idle
        # run the blocking parts (Popen, the up-to-5s port-open wait,
        # proc.wait) off the event loop via asyncio.to_thread, so holding
        # this for that whole span never blocks other projects' requests.
        self._lock = threading.Lock()

    def _free_port(self) -> int:
        used = set(self._ports.values())
        for port in PORT_RANGE:
            if port not in used and _port_is_free(port):
                return port
        raise RuntimeError("BackendSupervisor: no free port in 30021-30120")

    def _spawn(self, slug: str, backend_path: str) -> int:
        port = self._free_port()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "backend:app",
                "--host", "127.0.0.1",
                "--port", str(port),
            ],
            cwd=os.path.dirname(backend_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._procs[slug] = proc
        self._ports[slug] = port
        self._last_used[slug] = time.time()
        return port

    def _ensure_running_locked(self, slug: str, backend_path: str) -> int | None:
        """Body of ensure_running — caller must hold self._lock."""
        if not os.path.exists(backend_path):
            return None

        proc = self._procs.get(slug)
        if proc is not None and proc.poll() is None:
            self._last_used[slug] = time.time()
            return self._ports[slug]

        # Dead or never started — (re)spawn, then wait briefly for the
        # socket to accept connections before handing the port back.
        port = self._spawn(slug, backend_path)
        deadline = time.time() + 5
        while time.time() < deadline:
            if not _port_is_free(port):
                self._last_used[slug] = time.time()
                return port
            if self._procs[slug].poll() is not None:
                return None  # crashed on startup (syntax error, etc.)
            time.sleep(0.1)
        return None

    def _ensure_running_sync(self, slug: str, backend_path: str) -> int | None:
        with self._lock:
            return self._ensure_running_locked(slug, backend_path)

    async def ensure_running(self, slug: str, backend_path: str) -> int | None:
        """Return the project's backend port, spawning it if needed.

        Returns None if backend.py doesn't exist (project has no mock
        backend) or the subprocess died and won't come back up. Runs off
        the event loop (asyncio.to_thread) — the up-to-5s wait for a
        (re)spawned subprocess's port to open no longer blocks every other
        project's traffic on this worker.
        """
        return await asyncio.to_thread(self._ensure_running_sync, slug, backend_path)

    def _reload_sync(self, slug: str, backend_path: str) -> int | None:
        with self._lock:
            proc = self._procs.pop(slug, None)
            self._ports.pop(slug, None)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            return self._ensure_running_locked(slug, backend_path)

    async def reload(self, slug: str, backend_path: str) -> int | None:
        """Force-restart this project's backend, isolated from all others."""
        return await asyncio.to_thread(self._reload_sync, slug, backend_path)

    def _reap_idle_sync(self) -> None:
        now = time.time()
        with self._lock:
            for slug in list(self._procs):
                if now - self._last_used.get(slug, now) > IDLE_TIMEOUT_SECONDS:
                    proc = self._procs.pop(slug)
                    self._ports.pop(slug, None)
                    self._last_used.pop(slug, None)
                    if proc.poll() is None:
                        proc.terminate()

    async def reap_idle(self) -> None:
        await asyncio.to_thread(self._reap_idle_sync)

    async def proxy(self, slug: str, backend_path: str, subpath: str, request) -> tuple[int, bytes, dict]:
        """Proxy one request to the project's backend. Never raises a bare
        502/503/504 — callers get a (status, body, headers) tuple to build
        their own response, defaulting to 500 + JSON on any failure."""
        port = await self.ensure_running(slug, backend_path)
        if port is None:
            body = b'{"error":"project backend not running","slug":"%s"}' % slug.encode()
            return 500, body, {"content-type": "application/json"}

        url = f"http://127.0.0.1:{port}/api/{subpath}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.request(
                    request.method, url,
                    params=dict(request.query_params),
                    content=await request.body(),
                    headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                )
            return resp.status_code, resp.content, dict(resp.headers)
        except httpx.HTTPError as exc:
            body = (f'{{"error":"project backend unreachable: {exc}","slug":"{slug}"}}').encode()
            return 500, body, {"content-type": "application/json"}


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0
