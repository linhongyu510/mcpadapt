"""Lifecycle rollback tests for the synchronous MCPAdapt startup.

These tests exercise the real production runtime: a real background thread running
an asyncio event loop and a real spawned stdio child process. The only test-side
spy is a controlled patch of ``ready.wait`` for the interrupt boundary; the owner
thread and the spawned process stay real.

Child-process bookkeeping is strictly per-test: a before-snapshot of direct
children is captured, and the fixture finalizer reaps ONLY the delta pids
(after minus before) whose create_time still matches. Pre-existing children of
any other fixture are never signaled. Rollback assertions run inside the test
body, BEFORE any fixture finalizer, so they prove the owner really cleaned up.
"""

import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from textwrap import dedent
from typing import Any

import mcp
import psutil
import pytest
from mcp import StdioServerParameters

from mcpadapt.core import MCPAdapt, ToolAdapter

SILENT_SERVER_SCRIPT = "import time\ntime.sleep(300)\n"

ECHO_SERVER_SCRIPT = dedent(
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Echo Server")

    @mcp.tool()
    def echo_tool(text: str) -> str:
        # Echo the input text
        return f"Echo: {text}"

    mcp.run()
    """
)


class DummyAdapter(ToolAdapter):
    """A dummy adapter that returns the function as is"""

    def adapt(
        self,
        func: Callable[[dict | None], mcp.types.CallToolResult],
        mcp_tool: mcp.types.Tool,
    ):
        return func

    def async_adapt(
        self,
        afunc: Callable[[dict | None], Coroutine[Any, Any, mcp.types.CallToolResult]],
        mcp_tool: mcp.types.Tool,
    ):
        return afunc


def _children_snapshot() -> dict[int, float]:
    """Map pid -> create_time for the direct children of this process."""
    me = psutil.Process(os.getpid())
    out = {}
    for c in me.children(recursive=False):
        try:
            out[c.pid] = c.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


@pytest.fixture
def tracked_children():
    """Per-test child tracking.

    Yields the before-snapshot. On teardown, reaps only pids that appeared after
    the snapshot and still match their create_time (i.e. children THIS test run
    started). Pre-existing children are left untouched.
    """
    before = _children_snapshot()
    yield before
    after = _children_snapshot()
    for pid, create_time in after.items():
        if pid in before:
            continue  # pre-existing, never touch
        try:
            p = psutil.Process(pid)
            if p.create_time() != create_time:
                continue  # pid reuse, not ours
            p.terminate()
            p.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                p.kill()
                p.wait(timeout=5)
            except psutil.Error:
                pass


def _delta_children(before: dict[int, float]) -> dict[int, float]:
    """Children spawned since ``before`` (delta of this run)."""
    now = _children_snapshot()
    return {pid: ct for pid, ct in now.items() if pid not in before}


@pytest.fixture
def silent_server_params() -> StdioServerParameters:
    """A stdio server that never performs the MCP handshake.

    ``start()`` therefore always gives up via ``connect_timeout``.
    """
    return StdioServerParameters(
        command=sys.executable, args=["-c", SILENT_SERVER_SCRIPT]
    )


@pytest.fixture
def echo_server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=["-c", ECHO_SERVER_SCRIPT]
    )


def test_start_timeout_rolls_back_partial_startup(
    silent_server_params, tracked_children
):
    """Timeout must reap thread/loop/child, and finish in bounded time."""
    adapter = MCPAdapt(silent_server_params, DummyAdapter(), connect_timeout=2)

    t0 = time.monotonic()
    with pytest.raises(
        TimeoutError, match="Couldn't connect to the MCP server after 2 seconds"
    ):
        adapter.start()
    elapsed = time.monotonic() - t0

    # pre-finalizer assertions: the rollback itself must have cleaned up
    assert not adapter.thread.is_alive()
    assert adapter.loop.is_closed()
    assert _delta_children(tracked_children) == {}, "stdio child was not reaped"

    # elapsed ~= connect_timeout, never an unbounded block
    assert 2.0 <= elapsed < 20.0, f"rollback took {elapsed:.1f}s"


def test_interrupt_rolls_back_partial_startup(silent_server_params, tracked_children):
    """A KeyboardInterrupt during ready.wait must roll back and preserve it.

    Repeated 20x to cover the scheduling race where the startup thread has not
    yet published ``self.task`` when the interrupt lands.
    """
    for _ in range(20):
        adapter = MCPAdapt(silent_server_params, DummyAdapter(), connect_timeout=30)
        adapter.ready.wait = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())

        t0 = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            adapter.start()
        elapsed = time.monotonic() - t0

        assert not adapter.thread.is_alive()
        assert adapter.loop.is_closed()
        assert elapsed < 20.0, f"rollback hung for {elapsed:.1f}s (task=None race?)"


def test_interrupt_before_task_creation_race(
    silent_server_params, tracked_children, monkeypatch
):
    """Race window: the interrupt lands while the startup thread is stalled
    BEFORE it assigns ``self.task``.

    The owner thread still really starts and really spawns the child; only the
    thread's own entry point is gated by a barrier (monkeypatched wrapper around
    the real ``_run_loop``). The rollback must still complete -- close() must not
    join an unbounded thread that no one will ever cancel.
    """
    orig_run_loop = MCPAdapt._run_loop
    gate = threading.Event()

    def stalled_run_loop(self):
        gate.wait(timeout=4)  # hold the thread before create_task()
        orig_run_loop(self)

    monkeypatch.setattr(MCPAdapt, "_run_loop", stalled_run_loop)

    adapter = MCPAdapt(silent_server_params, DummyAdapter(), connect_timeout=30)
    adapter.ready.wait = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())

    def run_start():
        try:
            adapter.start()
        except KeyboardInterrupt:
            pass  # the intentionally propagated primary exception

    probe = threading.Thread(target=run_start)
    probe.start()
    probe.join(timeout=15)
    gate.set()  # release the stalled thread whatever happened

    assert not probe.is_alive(), "rollback hung unbounded: close() joined task=None"

    assert not adapter.thread.is_alive()
    assert adapter.loop.is_closed()
    assert _delta_children(tracked_children) == {}, "stdio child not reaped"


def test_cleanup_failure_does_not_mask_timeout(
    silent_server_params, tracked_children, caplog
):
    """If rollback cleanup explodes, the primary TimeoutError still wins.

    The cleanup error must surface as a log record with the exact warning message.
    """
    adapter = MCPAdapt(silent_server_params, DummyAdapter(), connect_timeout=2)

    def boom():
        raise RuntimeError("cleanup exploded")

    adapter.close = boom

    try:
        with (
            caplog.at_level(logging.WARNING, logger="mcpadapt.core"),
            pytest.raises(
                TimeoutError, match="Couldn't connect to the MCP server after 2 seconds"
            ),
        ):
            adapter.start()

        # the secondary cleanup error must be logged, not raised, with the exact message
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            r.getMessage() == "Error while rolling back a partially started MCPAdapt"
            for r in warnings
        )
    finally:
        # undo the spy and reap the partial startup for real
        del adapter.close
        adapter.close()


def test_normal_exit_cleans_up(echo_server_params, tracked_children):
    """Happy path: enter + normal exit leaves no thread/loop/child behind."""
    adapter = MCPAdapt(echo_server_params, DummyAdapter())
    with adapter as tools:
        assert tools
    assert not adapter.thread.is_alive()
    assert adapter.loop.is_closed()
    assert _delta_children(tracked_children) == {}


def test_exception_inside_with_cleans_up(echo_server_params, tracked_children):
    """An exception inside the with-block is cleaned up via __exit__."""
    adapter = MCPAdapt(echo_server_params, DummyAdapter())

    with pytest.raises(RuntimeError, match="boom"), adapter as tools:
        assert tools
        raise RuntimeError("boom")

    assert not adapter.thread.is_alive()
    assert adapter.loop.is_closed()
    assert _delta_children(tracked_children) == {}


def test_base_exception_inside_with_cleans_up(echo_server_params, tracked_children):
    """KeyboardInterrupt inside the with-block is cleaned up via __exit__."""
    adapter = MCPAdapt(echo_server_params, DummyAdapter())

    with pytest.raises(KeyboardInterrupt), adapter as tools:
        assert tools
        raise KeyboardInterrupt()

    assert not adapter.thread.is_alive()
    assert adapter.loop.is_closed()
    assert _delta_children(tracked_children) == {}


def test_repeated_timeouts_do_not_accumulate(silent_server_params, tracked_children):
    """Two failed starts in a row must not accumulate threads or children."""
    for _ in range(2):
        adapter = MCPAdapt(silent_server_params, DummyAdapter(), connect_timeout=2)
        with pytest.raises(TimeoutError):
            adapter.start()
        assert not adapter.thread.is_alive()
        assert adapter.loop.is_closed()

    live = [t for t in threading.enumerate() if "_run_loop" in t.name]
    assert live == []
    assert _delta_children(tracked_children) == {}
