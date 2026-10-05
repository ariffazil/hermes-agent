"""Entity-grounding gate — witness-before-assertion for biographical claims.

Born from the 2026-10-05 BANGANG incident (scar Genesis): a fresh guest-lane
session rendered parametric memory as biography — invented a chairmanship
with tenure for a semi-specified person name, zero probes, first turn.

This module gives the runtime a structural memory of that scar:

1. Probe tracking: :func:`note_probe_tool` records grounding-capable tool
   calls (web search/fetch, browser) per session.
2. Claim scanning: :func:`scan_ungrounded_entity_claims` flags assistant
   responses asserting biographical facts (titles, orgs, tenure ranges)
   while the session made no such probe.
3. Advisory footer: :func:`build_grounding_footer` marks those claims as
   UNKNOWN on the turn itself, mirroring the file-mutation verifier footer
   pattern — over-claiming becomes visible, not silent.

Advisory, not censoring: the response is delivered; the footer marks what
was never probed. Config: ``agent.entity_grounding`` (default True).
The guest-lane spine prompt text lives in ``agent.prompt_builder``.
"""

from __future__ import annotations

import re
from typing import Any


# Tools that can ground an entity claim against external reality.
# Deliberately narrow: local file reads and execute_code do NOT verify a
# named person's biography — the incident session ran plenty of local code
# while still confabulating. Only external-world probes count.
PROBE_TOOL_NAMES = frozenset({
    "web_search",
    "web_fetch",
    "fetch_url",
    "search_web",
    "web_search_exa",
    "browser_navigate",
    "browser_snapshot",
})

# Biographical claim shapes observed in the incident (title + org, tenure
# ranges, "ex-" roles, appointment dates). Conservative by design: the
# footer is advisory, so false positives cost one extra line, while false
# negatives repeat the scar.
_BIO_CLAIM_PATTERNS = [
    # "ex-Chairman PETRONAS", "bekas CEO X"
    r"\b(?:ex-|bekas\s+)(?:Chairman|Chairperson|CEO|CFO|COO|President|Minister|Governor)\b[^\.\n]{0,60}",
    # "Chairman of PETRONAS from 2014", "CEO X sejak 2020"
    r"\b(?:Chairman|Chairperson|CEO|CFO|COO|President|Director.General|Minister|Governor)\b"
    r"[^\.\n]{0,80}?\b(?:from|sejak|since|pada|in)\s+(?:\d{4}|[A-Z][a-z]+\s+\d{4})\b",
    # "2010-2017 ... Chairman" / "Chairman ... 2010–2017"
    r"\b\d{4}\s*(?:-|–|—|to|hingga)\s*\d{4}\b[^\.\n]{0,60}?"
    r"\b(?:Chairman|Chairperson|CEO|CFO|COO|President|Minister|Governor|tenure|mandat)\b",
    r"\b(?:Chairman|Chairperson|CEO|President|tenure|mandat)\b[^\.\n]{0,40}?"
    r"\b\d{4}\s*(?:-|–|—|to|hingga)\s*\d{4}\b",
]


def note_probe_tool(agent: Any, tool_name: str) -> None:
    """Record a grounding-capable probe on the agent (session-scoped)."""
    try:
        if tool_name in PROBE_TOOL_NAMES:
            agent._grounding_probes_this_session = True
    except Exception:
        pass


def probed_this_session(agent: Any) -> bool:
    """True once any grounding-capable probe ran this session."""
    return bool(getattr(agent, "_grounding_probes_this_session", False))


def scan_ungrounded_entity_claims(response_text: str) -> list:
    """Return biographical-claim snippets found in *response_text*.

    Pure function — no agent state. The caller decides whether the session
    had probes; if it did, the hits are ignored.
    """
    if not response_text:
        return []
    hits: list = []
    for pattern in _BIO_CLAIM_PATTERNS:
        for match in re.finditer(pattern, response_text, flags=re.IGNORECASE):
            snippet = " ".join(match.group(0).split())
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            if snippet not in hits:
                hits.append(snippet)
    return hits[:5]


def build_grounding_footer(hits: list) -> str:
    """Advisory footer marking unprobed biographical claims as UNKNOWN."""
    lines = [
        "⚖️ Grounding note (entity-grounding gate):",
        "Claims below were asserted WITHOUT any external probe this session "
        "— treat as UNKNOWN until verified:",
    ]
    for hit in hits:
        lines.append(f"- {hit}")
    lines.append(
        "Rule: probe-before-assertion. Memory recall without provenance is a "
        "hypothesis, not a fact."
    )
    return "\n".join(lines)
