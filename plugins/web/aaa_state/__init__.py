"""AAA State web intelligence gateway plugin — routes through A-FORGE governed search."""
from __future__ import annotations
from plugins.web.aaa_state.provider import AAAStateWebProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(AAAStateWebProvider())
