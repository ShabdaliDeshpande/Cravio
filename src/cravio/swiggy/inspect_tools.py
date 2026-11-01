"""Sign in to Swiggy MCP and save every server's tool list, so we build against the real schemas.

Run:  uv run cravio-swiggy-tools [food im dineout]

Only lists tools; it never calls one, so nothing touches your cart or orders.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before cravio imports, which read settings at import

from mcp.types import PaginatedRequestParams  # noqa: E402

from cravio.swiggy.client import SERVERS, build_client, is_blocked  # noqa: E402

OUT_DIR = Path("docs/swiggy_tools")


async def dump(servers: list[str]) -> None:
    client = build_client(servers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # One server at a time: the first signs in, the rest reuse the saved token.
    for server in servers:
        tools, cursor = [], None
        async with client.session(server) as session:
            while True:
                page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
                tools.extend(page.tools)
                cursor = page.nextCursor
                if not cursor:
                    break

        out = OUT_DIR / f"{server}.json"
        out.write_text(
            json.dumps([t.model_dump(mode="json", exclude_none=True) for t in tools], indent=2),
            encoding="utf-8",
        )
        print(f"\n{server}: {len(tools)} tools -> {out}")
        for t in sorted(tools, key=lambda t: t.name):
            required = t.inputSchema.get("required", [])
            read_only = bool(t.annotations and t.annotations.readOnlyHint)
            mark = "BLOCKED " if is_blocked(t.name, read_only) else "        "
            print(f"  {mark}{t.name}({', '.join(required)})")


def main() -> None:
    servers = sys.argv[1:] or list(SERVERS)
    asyncio.run(dump(servers))


if __name__ == "__main__":
    main()
