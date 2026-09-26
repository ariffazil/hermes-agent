"""AAA State Agent — Unified Web Intelligence Gateway.

Routes Hermes web_search / web_extract through A-FORGE MCP server,
which provides governed multi-provider search, evidence tracking,
and audit receipts.

Config: ``web.search_backend`` / ``web.extract_backend`` / ``web.backend: "aaa-state"``
Env: ``AAA_WEB_GATEWAY_URL`` (default: http://127.0.0.1:7072/mcp)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from plugins.web._common import (
    BaseWebSearchProvider,
    document,
    extract_fail,
    page_error,
    run_extract,
    run_search,
    search_fail,
    search_ok,
    title_hit,
)

logger = logging.getLogger(__name__)

# A-FORGE MCP HTTP endpoint on localhost (same box, SSRF-safe)
_DEFAULT_GATEWAY_URL = "http://127.0.0.1:7072/mcp"


def _gateway_url() -> str:
    """Resolve gateway URL: env var first, then localhost default."""
    from plugins.web._common import provider_env
    return (provider_env("AAA_WEB_GATEWAY_URL") or _DEFAULT_GATEWAY_URL).rstrip("/")


def _call_forge_mcp(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call an A-FORGE MCP tool via HTTP Streamable transport.

    Uses the MCP JSON-RPC protocol directly over HTTP.
    Returns the parsed result content.
    """
    import httpx

    url = _gateway_url()
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": tool,
            "arguments": arguments,
        },
    }

    try:
        resp = httpx.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-AAA-Caller": "hermes",
            },
            timeout=35.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # MCP JSON-RPC response
        result = data.get("result", {})

        # Extract text content from MCP response
        content_list = result.get("content", [])
        if content_list and isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return {"raw_text": text}

        # Fallback: return result directly
        return result

    except httpx.ConnectError:
        logger.error("A-FORGE MCP unreachable at %s", url)
        return {"error": f"A-FORGE MCP unreachable at {url}"}
    except httpx.TimeoutException:
        logger.error("A-FORGE MCP timeout calling %s", tool)
        return {"error": f"A-FORGE MCP timeout calling {tool}"}
    except Exception as exc:
        logger.error("A-FORGE MCP call failed: %s", exc)
        return {"error": f"A-FORGE MCP call failed: {exc}"}


async def _run_extract_async(
    vendor: str,
    urls: List[str],
    body,
) -> List[Dict[str, Any]]:
    """Async extract guard — mirrors _common.run_extract_async."""
    try:
        from plugins.web._common import _interrupted, _extract_interrupted
        if _interrupted():
            return _extract_interrupted(urls)
        return await body()
    except Exception as exc:
        msg = f"{vendor} extract failed: {exc}"
        logger.warning("%s extract error: %s", vendor, exc)
        return extract_fail(urls, msg)


class AAAStateWebProvider(BaseWebSearchProvider):
    """Unified web intelligence gateway — search + extract via A-FORGE.

    Provider name: ``aaa-state``
    MCP backend: A-FORGE gatewayTools (Brave + SearXNG + Context7)
    """

    NAME = "aaa-state"
    DISPLAY_NAME = "AAA State Web Gateway"
    KEY_ENV = "AAA_WEB_GATEWAY_URL"

    # --- Availability ---

    def is_available(self) -> bool:
        """Cheap check — verify A-FORGE MCP is reachable. Single fast probe, timeout 3s."""
        import httpx
        url = _gateway_url()
        try:
            resp = httpx.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 0,
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "hermes-aaa-probe", "version": "0.1.0"},
                    },
                },
                timeout=3.0,
            )
            return resp.status_code < 500
        except Exception:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # --- Search ---

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search via A-FORGE forge_search (Brave primary, SearXNG fallback)."""
        def _do():
            limit_clamped = max(1, min(int(limit), 20))

            result = _call_forge_mcp("forge_search", {
                "query": query,
                "source": "web",
                "count": limit_clamped,
                "freshness": "any",
                "safesearch": "moderate",
                "request_id": f"hermes-aaa-{int(time.time())}",
            })

            if "error" in result and not result.get("results"):
                return search_fail(result["error"])

            # Extract results from forge_search response
            results = result.get("results", [])

            # Normalize to Hermes contract: {title, url, description, position}
            web_results = []
            for i, r in enumerate(results[:limit_clamped]):
                web_results.append(
                    title_hit(
                        title=str(r.get("title", "")),
                        url=str(r.get("url", "")),
                        description=str(r.get("snippet", r.get("description", ""))),
                        position=i + 1,
                    )
                )

            receipt_id = result.get("receipt_id", "")
            provider = result.get("provider", "unknown")
            logger.info(
                "AAA gateway search '%s': %d results (provider=%s, receipt=%s)",
                query, len(web_results), provider, receipt_id,
            )

            return search_ok(web_results)

        return run_search("AAA State", logger, _do)

    # --- Extract ---

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from URLs via A-FORGE forge_web_extract."""

        async def _do():
            results = []
            for url in urls:
                try:
                    result = _call_forge_mcp("forge_web_extract", {
                        "url": url,
                        "render": kwargs.get("render", "auto"),
                        "max_chars": kwargs.get("max_chars", 50000),
                        "request_id": f"hermes-aaa-ext-{int(time.time())}",
                    })

                    if "error" in result and not result.get("content"):
                        results.append(page_error(url, result["error"]))
                        continue

                    # Extract content from forge_web_extract response
                    content_text = result.get("content", "")
                    title = result.get("title", "")

                    results.append(
                        document(
                            url=url,
                            title=str(title),
                            content=str(content_text),
                        )
                    )

                except Exception as exc:
                    logger.warning("AAA extract failed for %s: %s", url, exc)
                    results.append(page_error(url, str(exc)))

            return results

        return await _run_extract_async("AAA State", urls, _do)

    # --- Setup ---

    def get_setup_schema(self) -> Dict[str, Any]:
        from plugins.web._common import setup_schema
        return setup_schema(
            "AAA State Web Gateway",
            "free",
            "A-FORGE governed search — routes through Brave, SearXNG, Context7 with evidence tracking and audit receipts.",
            "AAA_WEB_GATEWAY_URL",
            "A-FORGE MCP gateway URL (default: http://127.0.0.1:7072/mcp)",
            "http://127.0.0.1:7072/mcp",
        )
