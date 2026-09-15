import pytest
import unittest
import platform
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import importlib.util

from lazyllm.tools import MCPClient


@pytest.mark.parametrize('schema_field', ['inputSchema', 'input_schema'])
def test_tool_schema_sdk_field_variants(schema_field):
    from lazyllm.tools.mcp.tool_adaptor import generate_lazyllm_tool
    tool = SimpleNamespace(name='search', description='Search documents.', **{
        schema_field: {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']},
    })
    converted = generate_lazyllm_tool(None, tool)
    assert converted.__name__ == 'search'
    assert converted.__signature__.parameters['query'].annotation is str
    setattr(tool, schema_field, {})
    assert not generate_lazyllm_tool(None, tool).__signature__.parameters


@pytest.mark.parametrize('stream_count', [2, 3])
def test_streamable_http_sdk_tuple_variants(stream_count, monkeypatch):
    import lazyllm.tools.mcp.client as client_module
    read, write = object(), object()
    events = []

    @asynccontextmanager
    async def http_client(**kwargs):
        assert kwargs['timeout'] == 5
        yield object()

    @asynccontextmanager
    async def transport(**kwargs):
        try:
            yield (read, write, lambda: 'session-id')[:stream_count]
        finally:
            events.append('closed')

    class Session:
        def __init__(self, r, w):
            assert (r, w) == (read, write)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def initialize(self):
            events.append('initialized')

        async def list_tools(self):
            return 'tools'

    sdk = SimpleNamespace(streamable_http_client=transport, create_mcp_http_client=http_client)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
    monkeypatch.setattr(importlib.util, 'find_spec', lambda _: spec)
    monkeypatch.setattr(importlib.util, 'module_from_spec', lambda _: sdk)
    monkeypatch.setattr(client_module, 'mcp', SimpleNamespace(ClientSession=Session))
    client = MCPClient('https://example.test/mcp', transport='streamable-http')
    assert asyncio.run(client.list_tools()) == 'tools'
    assert events == ['initialized', 'closed']


class TestMCP(unittest.TestCase):
    def setUp(self):
        if platform.system() == "Windows":
            self.config = {
                "command": "cmd",
                "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", "./"]
            }
        else:
            self.config = {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"]
            }
        self.client = MCPClient(command_or_url=self.config["command"], args=self.config["args"])

    def test_get_tools_sync(self):
        tools = self.client.get_tools(allowed_tools=["list_allowed_directories"])
        assert len(tools) == 1, f"Expected one tool 'list_allowed_directories', got {len(tools)}"

    @pytest.mark.asyncio
    async def test_get_tools_async(self):
        tools = await self.client.aget_tools(allowed_tools=["list_allowed_directories"])
        assert len(tools) == 1, f"Expected one tool 'list_allowed_directories', got {len(tools)}"

    def test_tool_call(self):
        tool = self.client.get_tools(allowed_tools=["list_allowed_directories"])[0]
        res = tool()
        assert "Tool call result:" in res
