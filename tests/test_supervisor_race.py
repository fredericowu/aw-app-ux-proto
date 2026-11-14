"""Regression test for the BackendSupervisor race (Kanban card 4185bf3b).

Root cause: ensure_running()/reload() ran their blocking subprocess-(re)spawn
wait (up to 5s) directly on the event loop, with no lock around the shared
_procs/_ports/_last_used dicts. A burst of concurrent reload/proxy calls for
one project could therefore (a) stall every other project's traffic on this
worker for up to 5s per respawn, and (b) let two calls simultaneously decide
the backend was missing and each spawn their own subprocess, corrupting the
shared state. Needs no Postgres — exercises BackendSupervisor directly
against a real throwaway uvicorn subprocess, same as the live bug.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "container" / "src" / "api"))

from supervisor import BackendSupervisor  # noqa: E402

_BACKEND_PY = """
from fastapi import FastAPI

app = FastAPI()


@app.get("/api/ping")
def ping():
    return {"ok": True}
"""


class _FakeRequest:
    """Minimal stand-in for the Starlette Request that supervisor.proxy() uses."""

    method = "GET"
    query_params: dict = {}
    headers: dict = {}

    async def body(self) -> bytes:
        return b""


@pytest.fixture
def backend_path(tmp_path) -> str:
    path = tmp_path / "backend.py"
    path.write_text(_BACKEND_PY)
    return str(path)


def test_concurrent_reload_and_proxy_never_overlap_or_stall_event_loop(backend_path):
    """Hammers proxy() and reload() concurrently for one slug.

    Proves two things the race depended on:
    1. The lock actually serializes the critical section — instrumented
       directly, so an overlap here means two calls decided "not running"
       at once and could double-spawn, the exact corruption the card
       describes.
    2. The event loop keeps making progress throughout — a heartbeat task
       ticking every 20ms should not go quiet for multi-second stretches,
       which is what "blocks the WHOLE event loop for up to 5s" looked like
       before the fix.
    """
    sup = BackendSupervisor()
    slug = "race-test"

    overlap_detected = threading.Event()
    in_section = threading.Event()
    orig_locked = sup._ensure_running_locked

    def _instrumented(slug_: str, backend_path_: str):
        if in_section.is_set():
            overlap_detected.set()
        in_section.set()
        try:
            return orig_locked(slug_, backend_path_)
        finally:
            in_section.clear()

    sup._ensure_running_locked = _instrumented

    async def _heartbeat(counter: list, stop: asyncio.Event) -> None:
        while not stop.is_set():
            counter[0] += 1
            await asyncio.sleep(0.02)

    async def _hammer_proxy(stop: asyncio.Event, errors: list) -> None:
        req = _FakeRequest()
        while not stop.is_set():
            try:
                await sup.proxy(slug, backend_path, "ping", req)
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(exc)
            await asyncio.sleep(0.01)

    async def _hammer_reload(n: int) -> None:
        for _ in range(n):
            await sup.reload(slug, backend_path)

    async def _run():
        stop = asyncio.Event()
        counter = [0]
        errors: list = []
        heartbeat_task = asyncio.create_task(_heartbeat(counter, stop))
        proxy_tasks = [asyncio.create_task(_hammer_proxy(stop, errors)) for _ in range(5)]

        start = time.monotonic()
        # 4 "agent sessions" each hot-reloading 3 times in a tight burst —
        # the normal edit/hot_reload_backend pattern the card describes.
        await asyncio.gather(*[_hammer_reload(3) for _ in range(4)])
        elapsed = time.monotonic() - start

        stop.set()
        heartbeat_task.cancel()
        for t in proxy_tasks:
            t.cancel()
        await asyncio.gather(heartbeat_task, *proxy_tasks, return_exceptions=True)
        return counter[0], errors, elapsed

    beats, errors, elapsed = asyncio.run(_run())

    assert not overlap_detected.is_set(), (
        "two calls were inside the locked critical section at once — "
        "the shared _procs/_ports race is NOT closed"
    )
    assert not errors, f"proxy() raised unexpectedly during the hammering: {errors}"

    # 12 reloads serialized by the lock, each with up to a ~5s worst-case
    # respawn wait, would silently stall the event loop for most of
    # `elapsed` if the blocking-event-loop bug were still present — the
    # heartbeat (every 20ms) would barely tick. Require it to have ticked
    # at a rate consistent with the loop staying live throughout, not just
    # in the gaps between reloads.
    min_expected_beats = elapsed / 0.05  # generous: allow 2.5x the 20ms interval
    assert beats > min_expected_beats, (
        f"event loop only produced {beats} heartbeats over {elapsed:.1f}s — "
        "looks stalled, the blocking-event-loop bug is back"
    )

    # Supervisor ends up in a consistent state: exactly one live process for
    # the slug, and it actually answers.
    assert slug in sup._procs
    assert sup._procs[slug].poll() is None, "no live process tracked after the hammering"
    status, body, _ = asyncio.run(sup.proxy(slug, backend_path, "ping", _FakeRequest()))
    assert status == 200, body

    sup._procs[slug].terminate()
