"""Connect to Swiggy MCP servers and load their tools for the agent, minus anything that spends money."""

import asyncio
import email.utils
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone

import httpx
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT, McpHttpClientFactory
from mcp.types import BlobResourceContents, CallToolResult, EmbeddedResource, ResourceLink, TextContent

from cravio.swiggy.oauth import build_oauth_provider

log = logging.getLogger(__name__)

BASE_URL = os.getenv("SWIGGY_MCP_BASE_URL", "https://mcp.swiggy.com")
SERVERS = {"food": "/food", "im": "/im", "dineout": "/dineout"}

# The demo stops at the cart, so the agent never gets a tool that places, pays for,
# books or cancels a real order, even ones Swiggy marks read-only (the payment tools).
_ORDER_OR_PAYMENT = re.compile(r"place|confirm|checkout|pay|book|cancel")

# Writes we've reviewed and let the agent use. True = each call waits for the user's OK.
# Any other write Swiggy offers (delete_address, report_error, tools added later) is withheld.
REVIEWED_WRITES = {
    "update_food_cart": True,  # replaces the cart without warning if the restaurant changes
    "flush_food_cart": True,  # empties the user's real Swiggy cart
    "create_address": True,  # sends the user's name and phone number to Swiggy
    "apply_food_coupon": False,  # only changes the price, and the user can remove it
    "update_cart": True,  # Instamart: replaces the whole grocery cart with the items sent
}

# Swiggy answers 429 with a Retry-After when Cravio calls too fast. Its guidance: wait that long
# and try again, and give up after about 30 seconds in all.
RATE_LIMIT_BUDGET_SECONDS = 30


def is_blocked(tool_name: str, read_only: bool = False) -> bool:
    if _ORDER_OR_PAYMENT.search(tool_name):
        return True
    return not read_only and tool_name not in REVIEWED_WRITES


def needs_approval(tool_name: str) -> bool:
    return REVIEWED_WRITES.get(tool_name, False)


def enabled_servers() -> list[str]:
    names = [s.strip() for s in os.getenv("SWIGGY_SERVERS", "food,im").split(",") if s.strip()]
    unknown = set(names) - SERVERS.keys()
    if unknown:
        raise ValueError(f"Unknown SWIGGY_SERVERS {sorted(unknown)}; choose from {sorted(SERVERS)}")
    return names


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds Swiggy asks us to wait: Retry-After is either a number or an HTTP date."""
    value = response.headers.get("Retry-After", "")
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return (email.utils.parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
    except (TypeError, ValueError):
        return None


class _WaitOutRateLimit(httpx.AsyncBaseTransport):
    """Retries a request Swiggy rejected with 429, after the wait it asks for."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        waited, attempt = 0.0, 0
        while True:
            response = await self._inner.handle_async_request(request)
            if response.status_code != 429:
                return response
            delay = _retry_after(response)
            if not delay or delay < 0:  # missing, zero or already past: back off 1s, 2s, 4s...
                delay = 2.0**attempt
            if waited + delay > RATE_LIMIT_BUDGET_SECONDS:
                return response
            await response.aclose()
            log.warning("Swiggy is rate-limiting Cravio (429); trying again in %.0fs", delay, extra={"agent": "mcp"})
            await asyncio.sleep(delay)
            waited, attempt = waited + delay, attempt + 1

    async def aclose(self) -> None:
        await self._inner.aclose()


def http_client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    """MCP's usual HTTP client, except that it waits out Swiggy's rate limit instead of failing."""
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(MCP_DEFAULT_TIMEOUT, read=MCP_DEFAULT_SSE_READ_TIMEOUT),
        auth=auth,
        transport=_WaitOutRateLimit(httpx.AsyncHTTPTransport()),
    )


def build_client(
    servers: list[str] | None = None, httpx_client_factory: McpHttpClientFactory = http_client
) -> MultiServerMCPClient:
    connections: dict[str, StreamableHttpConnection] = {}
    for name in servers or enabled_servers():
        url = BASE_URL + SERVERS[name]
        print(f"SETUP  Swiggy MCP server {name} at {url} (streamable HTTP, OAuth)")
        log.info("SETUP  Swiggy MCP server %r at %s (streamable HTTP, OAuth)", name, url)
        connections[name] = {
            "transport": "streamable_http",
            "url": url,
            "auth": build_oauth_provider(url),
            "httpx_client_factory": httpx_client_factory,
        }
    # Food and Instamart both have tools like get_addresses. Names stay as Swiggy wrote them (the prompts use
    # them); agent.py gives each agent tools from one server only, so no agent sees two with the same name.
    return MultiServerMCPClient(connections, tool_name_prefix=False)


async def _files_as_text(request, handler):
    """The local model reads text only. When Swiggy sends back a link or a file, give the model a line of text
    about it instead: Ollama can't take a file and the whole turn would crash."""
    result = await handler(request)
    if not isinstance(result, CallToolResult):
        return result
    content = []
    for block in result.content:
        if isinstance(block, ResourceLink):
            block = TextContent(type="text", text=f"[link] {block.name}: {block.uri}")
        elif isinstance(block, EmbeddedResource) and isinstance(block.resource, BlobResourceContents):
            block = TextContent(type="text", text=f"[file not shown] {block.resource.uri} ({block.resource.mimeType})")
        content.append(block)
    return result.model_copy(update={"content": content})


@asynccontextmanager
async def swiggy_tools(client: MultiServerMCPClient) -> AsyncIterator[list[BaseTool]]:
    """Connect to each server and yield the tools the agent may use.

    Swiggy asks for one MCP session per user rather than a new handshake per tool call, so the
    sessions stay open until the block ends. Servers connect in turn, so only the first triggers
    a browser login.
    """
    async with AsyncExitStack() as stack:
        tools: list[BaseTool] = []
        for server in client.connections:
            session = await stack.enter_async_context(client.session(server))
            for tool in await load_mcp_tools(session, server_name=server, tool_name_prefix=client.tool_name_prefix,
                                             tool_interceptors=[_files_as_text]):
                name = tool.name.removeprefix(f"{server}_") if client.tool_name_prefix else tool.name
                read_only = bool((tool.metadata or {}).get("readOnlyHint"))
                if is_blocked(name, read_only):
                    log.info("SETUP  withholding %s/%s from the agent", server, name)
                    continue
                log.info("SETUP  giving the agent %s/%s%s", server, name, " (asks you first)" if needs_approval(name) else "")
                tool.metadata = {**(tool.metadata or {}), "swiggy_name": name, "swiggy_server": server}
                tools.append(tool)
        yield tools


async def _main() -> None:
    """Quick check: sign in to Swiggy and show each tool the agent would get: what it does and its inputs."""
    client = build_client()
    async with swiggy_tools(client) as tools:
        for tool in tools:
            print("=" * 80)
            print(tool.name, "(asks you first)" if needs_approval(tool.name) else "")
            print("DESCRIPTION:", tool.description)
            print("INPUT:", json.dumps(tool.args_schema, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())