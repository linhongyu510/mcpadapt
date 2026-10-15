import time
from textwrap import dedent
from typing import Any, Callable, Coroutine

import mcp
import pytest
from mcp import StdioServerParameters

from mcpadapt.core import MCPAdapt, ToolAdapter
from tests._server_utils import launch_mcp_server, terminate_mcp_server


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


@pytest.fixture
def echo_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server")

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"
        
        mcp.run()
        '''
    )


@pytest.fixture
def update_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP
        
        mcp = FastMCP("Echo Server")
        
        def new_tool(text: str) -> str:
            """New tool"""
            return f"New: {text}"
        
        def update_tool():
            mcp.add_tool(new_tool)
        
        updated = False
        
        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            # update tool when invoke
            global updated
            if not updated:
                update_tool()
                updated = True
            return f"Echo: {text}"
        
        mcp.run()
        '''
    )


@pytest.fixture
def echo_server_sse_script():
    return dedent(
        '''
        import os
        from mcp.server.fastmcp import FastMCP

        port = int(os.environ.get("MCP_TEST_PORT", "8000"))

        mcp = FastMCP("Echo Server", host="127.0.0.1", port=port)

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"

        mcp.run("sse")
        '''
    )


@pytest.fixture
async def echo_sse_server(echo_server_sse_script):
    try:
        process, port = launch_mcp_server(echo_server_sse_script)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    try:
        yield {"url": f"http://127.0.0.1:{port}/sse"}
    finally:
        terminate_mcp_server(process)


@pytest.fixture
def echo_server_streamable_http_script():
    return dedent(
        '''
        import os
        from mcp.server.fastmcp import FastMCP
        
        port = int(os.environ.get("MCP_TEST_PORT", "8000"))

        mcp = FastMCP("Echo Server", host="127.0.0.1", port=port, stateless_http=True, json_response=True)
        
        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"

        mcp.run("streamable-http")
        '''
    )


@pytest.fixture
async def echo_streamable_http_server(echo_server_streamable_http_script):
    try:
        process, port = launch_mcp_server(echo_server_streamable_http_script)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    try:
        yield {"url": f"http://127.0.0.1:{port}/mcp", "transport": "streamable-http"}
    finally:
        terminate_mcp_server(process)


@pytest.fixture
def echo_server_websocket_script():
    return dedent(
        '''
        import asyncio
        import json
        import websockets

        async def handle_mcp_client(websocket):
            """Handle MCP protocol over websocket"""
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get("method") == "initialize":
                        # Send initialize response
                        response = {
                            "jsonrpc": "2.0",
                            "id": data.get("id"),
                            "result": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {
                                    "tools": {}
                                },
                                "serverInfo": {
                                    "name": "Echo Server",
                                    "version": "1.0.0"
                                }
                            }
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif data.get("method") == "tools/list":
                        # Send tools list response
                        response = {
                            "jsonrpc": "2.0",
                            "id": data.get("id"),
                            "result": {
                                "tools": [{
                                    "name": "echo_tool",
                                    "description": "Echo the input text",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"}
                                        },
                                        "required": ["text"]
                                    }
                                }]
                            }
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif data.get("method") == "tools/call":
                        # Handle tool call
                        tool_name = data.get("params", {}).get("name")
                        arguments = data.get("params", {}).get("arguments", {})
                        
                        if tool_name == "echo_tool":
                            text = arguments.get("text", "")
                            response = {
                                "jsonrpc": "2.0",
                                "id": data.get("id"),
                                "result": {
                                    "content": [{
                                        "type": "text",
                                        "text": f"Echo: {text}"
                                    }]
                                }
                            }
                            await websocket.send(json.dumps(response))
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            
        async def main():
            server = await websockets.serve(handle_mcp_client, "127.0.0.1", 8001)
            await server.wait_closed()
        
        if __name__ == "__main__":
            asyncio.run(main())
        '''
    )


@pytest.fixture
def echo_websocket_server(echo_server_websocket_script):
    import subprocess

    # Start the WebSocket server process
    process = subprocess.Popen(
        ["python", "-c", echo_server_websocket_script],
    )

    # Give the server a moment to start up
    time.sleep(1)

    try:
        yield {"url": "ws://127.0.0.1:8001/ws", "transport": "ws"}
    finally:
        # Clean up the process when test is done
        process.kill()
        process.wait()


@pytest.fixture
async def echo_websocket_server_async(echo_server_websocket_script):
    import subprocess

    # Start the WebSocket server process
    process = subprocess.Popen(
        ["python", "-c", echo_server_websocket_script],
    )

    # Give the server a moment to start up
    time.sleep(1)

    try:
        yield {"url": "ws://127.0.0.1:8001/ws", "transport": "ws"}
    finally:
        # Clean up the process when test is done
        process.kill()
        process.wait()


@pytest.fixture
def slow_start_server_script():
    return dedent(
        '''
        import time
        from mcp.server.fastmcp import FastMCP

        # Sleep for 2 seconds to simulate slow startup
        time.sleep(2)
        
        mcp = FastMCP("Slow Server")

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"
        
        mcp.run()
        '''
    )


def test_basic_sync(echo_server_script):
    with MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_script]
        ),
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"


def test_basic_sync_multiple_tools(echo_server_script):
    with MCPAdapt(
        [
            StdioServerParameters(
                command="uv", args=["run", "python", "-c", echo_server_script]
            ),
            StdioServerParameters(
                command="uv", args=["run", "python", "-c", echo_server_script]
            ),
        ],
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 2
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"
        assert tools[1]({"text": "world"}).content[0].text == "Echo: world"


def test_basic_sync_update_tools(update_server_script):
    adapter = MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", update_server_script]
        ),
        DummyAdapter(),
    )
    with adapter as tools:
        assert len(tools) == 1
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"
        # get latest tools
        tools = adapter.tools()
        assert len(tools) == 2
        # assert all tools are valid
        res = (
            tools[0]({"text": "hello"}).content[0].text,
            tools[1]({"text": "hello"}).content[0].text,
        )
        assert "Echo: hello" in res
        assert "New: hello" in res


async def test_basic_async(echo_server_script):
    async with MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_script]
        ),
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        mcp_tool_call_result = await tools[0]({"text": "hello"})
        assert mcp_tool_call_result.content[0].text == "Echo: hello"


async def test_basic_async_multiple_tools(echo_server_script):
    async with MCPAdapt(
        [
            StdioServerParameters(
                command="uv", args=["run", "python", "-c", echo_server_script]
            ),
            StdioServerParameters(
                command="uv", args=["run", "python", "-c", echo_server_script]
            ),
        ],
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 2
        mcp_tool_call_result = await tools[0]({"text": "hello"})
        assert mcp_tool_call_result.content[0].text == "Echo: hello"
        mcp_tool_call_result = await tools[1]({"text": "world"})
        assert mcp_tool_call_result.content[0].text == "Echo: world"


async def test_basic_async_update_tools(update_server_script):
    adapter = MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", update_server_script]
        ),
        DummyAdapter(),
    )
    async with adapter as tools:
        assert len(tools) == 1
        assert (await tools[0]({"text": "hello"})).content[0].text == "Echo: hello"
        # get latest tools
        tools = await adapter.atools()
        assert len(tools) == 2
        # assert all tools are valid
        res = (
            (await tools[0]({"text": "hello"})).content[0].text,
            (await tools[1]({"text": "hello"})).content[0].text,
        )
        assert "Echo: hello" in res
        assert "New: hello" in res


def test_basic_sync_sse(echo_sse_server):
    sse_serverparams = echo_sse_server
    with MCPAdapt(
        sse_serverparams,
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"


def test_basic_sync_multiple_sse(echo_sse_server):
    sse_serverparams = echo_sse_server
    with MCPAdapt(
        [sse_serverparams, sse_serverparams],
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 2
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"
        assert tools[1]({"text": "world"}).content[0].text == "Echo: world"


async def test_basic_async_sse(echo_sse_server):
    sse_serverparams = echo_sse_server
    async with MCPAdapt(
        sse_serverparams,
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        mcp_tool_call_result = await tools[0]({"text": "hello"})
        assert mcp_tool_call_result.content[0].text == "Echo: hello"


async def test_basic_async_multiple_sse(echo_sse_server):
    sse_serverparams = echo_sse_server
    async with MCPAdapt(
        [sse_serverparams, sse_serverparams],
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 2
        mcp_tool_call_result = await tools[0]({"text": "hello"})
        assert mcp_tool_call_result.content[0].text == "Echo: hello"
        mcp_tool_call_result = await tools[1]({"text": "world"})
        assert mcp_tool_call_result.content[0].text == "Echo: world"


def test_basic_sync_streamable_http(echo_streamable_http_server):
    http_serverparams = echo_streamable_http_server
    with MCPAdapt(
        http_serverparams,
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"


async def test_basic_async_streamable_http(echo_streamable_http_server):
    http_serverparams = echo_streamable_http_server
    async with MCPAdapt(
        http_serverparams,
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        assert (await tools[0]({"text": "hello"})).content[0].text == "Echo: hello"


def test_basic_sync_websocket(echo_websocket_server):
    ws_serverparams = echo_websocket_server
    with MCPAdapt(
        ws_serverparams,
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"


def test_basic_sync_multiple_websocket(echo_websocket_server):
    ws_serverparams = echo_websocket_server
    with MCPAdapt(
        [ws_serverparams, ws_serverparams],
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 2
        assert tools[0]({"text": "hello"}).content[0].text == "Echo: hello"
        assert tools[1]({"text": "world"}).content[0].text == "Echo: world"


async def test_basic_async_websocket(echo_websocket_server):
    ws_serverparams = echo_websocket_server
    async with MCPAdapt(
        ws_serverparams,
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 1
        mcp_tool_call_result = await tools[0]({"text": "hello"})
        assert mcp_tool_call_result.content[0].text == "Echo: hello"


async def test_basic_async_multiple_websocket(echo_websocket_server):
    ws_serverparams = echo_websocket_server
    async with MCPAdapt(
        [ws_serverparams, ws_serverparams],
        DummyAdapter(),
    ) as tools:
        assert len(tools) == 2
        mcp_tool_call_result = await tools[0]({"text": "hello"})
        assert mcp_tool_call_result.content[0].text == "Echo: hello"
        mcp_tool_call_result = await tools[1]({"text": "world"})
        assert mcp_tool_call_result.content[0].text == "Echo: world"


def test_connect_timeout(slow_start_server_script):
    """Test that connect_timeout raises TimeoutError when server starts slowly"""
    with pytest.raises(
        TimeoutError, match="Couldn't connect to the MCP server after 1 seconds"
    ):
        with MCPAdapt(
            StdioServerParameters(
                command="uv", args=["run", "python", "-c", slow_start_server_script]
            ),
            DummyAdapter(),
            connect_timeout=1,  # 1 second timeout, server takes 2 seconds to start
        ):
            pass


def test_client_session_timeout_parameter_propagation(echo_server_script):
    """Test that client_session_timeout_seconds parameter is properly stored and accessible"""
    from datetime import timedelta

    # Test with float value
    adapter_float = MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_script]
        ),
        DummyAdapter(),
        client_session_timeout_seconds=2.5,
    )
    assert adapter_float.client_session_timeout_seconds == 2.5

    # Test with timedelta value
    timeout_td = timedelta(seconds=3.0)
    adapter_td = MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_script]
        ),
        DummyAdapter(),
        client_session_timeout_seconds=timeout_td,
    )
    assert adapter_td.client_session_timeout_seconds == timeout_td

    # Test with None value
    adapter_none = MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_script]
        ),
        DummyAdapter(),
        client_session_timeout_seconds=None,
    )
    assert adapter_none.client_session_timeout_seconds is None

    # Test default value
    adapter_default = MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_script]
        ),
        DummyAdapter(),
    )
    assert adapter_default.client_session_timeout_seconds == 5


# --- Regression: failed start()/__enter__ must roll back partial resources ---

import threading  # noqa: E402
import sys  # noqa: E402

# A valid FastMCP stdio server that deliberately cold-starts much later than the
# connect_timeout: the command is runnable and correct, it just never answers
# ``initialize`` in time. This mirrors a hung / very slow MCP server, not a
# contract violation.
_HUNG_SERVER_SCRIPT = (
    "import time\n"
    "time.sleep(30)\n"
    "from mcp.server.fastmcp import FastMCP\n"
    "mcp = FastMCP('Hung')\n"
    "@mcp.tool()\n"
    "def echo_tool(text: str) -> str:\n"
    "    return 'Echo: ' + text\n"
    "mcp.run()\n"
)


def _hung_params():
    return StdioServerParameters(
        command=sys.executable, args=["-c", _HUNG_SERVER_SCRIPT]
    )


def test_connect_timeout_rolls_back_thread_loop_and_task():
    """A timed-out start() must not leak the loop thread, task or event loop.

    Regression test: when readiness times out, ``start`` used to raise
    ``TimeoutError`` without tearing the partially-started connection down.
    Because ``__enter__`` raised, ``__exit__``/``close`` was never invoked,
    leaving the ``_run_loop`` thread blocked, the task pending, the event loop
    open and the owned subprocess alive.
    """
    adapter = MCPAdapt(_hung_params(), DummyAdapter(), connect_timeout=1)

    with pytest.raises(
        TimeoutError, match="Couldn't connect to the MCP server after 1 seconds"
    ):
        adapter.start()

    # Rollback must have cancelled the task, joined the loop thread and closed
    # the event loop.
    assert adapter.task is not None
    assert adapter.task.done(), "setup task was not cancelled on rollback"
    assert not adapter.thread.is_alive(), "_run_loop thread leaked after failed start"
    assert adapter.loop.is_closed(), "event loop leaked after failed start"


def test_context_manager_timeout_rolls_back():
    """The `with` path (__enter__ raises) must still roll back."""
    adapter = MCPAdapt(_hung_params(), DummyAdapter(), connect_timeout=1)
    with pytest.raises(TimeoutError):
        with adapter:  # __enter__ raises -> __exit__ must NOT be relied upon
            pass

    assert not adapter.thread.is_alive(), (
        "_run_loop thread leaked after __enter__ timeout"
    )
    assert adapter.loop.is_closed(), "event loop leaked after __enter__ timeout"


def test_no_dangling_run_loop_thread_after_timeout():
    """No background thread targeting ``_run_loop`` may survive a failed start."""
    before = {t.ident for t in threading.enumerate()}
    adapter = MCPAdapt(_hung_params(), DummyAdapter(), connect_timeout=1)
    with pytest.raises(TimeoutError):
        adapter.start()
    survivors = [
        t for t in threading.enumerate() if t.ident not in before and t.is_alive()
    ]
    assert survivors == [], f"leaked background threads after timeout: {survivors}"


def test_close_idempotent_after_successful_start(echo_server_script):
    """close() may be safely called twice (success path stays idempotent)."""
    adapter = MCPAdapt(
        StdioServerParameters(command=sys.executable, args=["-c", echo_server_script]),
        DummyAdapter(),
        connect_timeout=15,
    )
    with adapter as tools:
        assert len(tools) == 1
    # Already closed by __exit__; a second close must not raise.
    adapter.close()
    assert not adapter.thread.is_alive()
    assert adapter.loop.is_closed()
