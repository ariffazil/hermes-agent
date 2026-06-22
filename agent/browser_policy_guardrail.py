"""
browser_policy_guardrail.py
arifOS Browser Policy Guardrail for Hermes

Hooks into ToolCallGuardrailController.before_call() to enforce the
BROWSER-POLICY.md risk classification on every browser_* tool call.

Risk classes:
    R (READ)      — no approval needed
    W (WRITE)    — arifOS policy check required
    P (PRIVILEGED) — always blocked; requires 888_HOLD + OpenClaw delegation
    P (PRIVILEGED) — 888_HOLD + evidence bundle required

Author: OPENCLAW
Date: 2026-06-19
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.tool_guardrails import ToolGuardrailDecision, ToolCallSignature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk Classification Table
# ---------------------------------------------------------------------------

BROWSER_RISK_TABLE: dict[str, str] = {
    # Class R — READ (no approval by default)
    "browser_navigate": "R",
    "browser_snapshot": "R",
    "browser_scroll": "R",
    "browser_back": "R",
    "browser_get_images": "R",
    "web_search": "R",
    # Class W — WRITE (policy check required)
    "browser_click": "W",
    "browser_type": "W",
    "browser_press": "W",
    "browser_vision": "W",
    # Class P — PRIVILEGED (always 888_HOLD)
    "browser_cdp": "P",
    "browser_dialog": "P",
    "browser_console": "P",
}

# ---------------------------------------------------------------------------
# arifOS Policy Check
# ---------------------------------------------------------------------------

# Cache for the arifOS MCP client to avoid re-initializing every call
_arif_mcp_client: Any = None


def _get_arif_mcp_client():
    """Lazily initialize the arifOS MCP client if available."""
    global _arif_mcp_client
    if _arif_mcp_client is not None:
        return _arif_mcp_client

    try:
        import httpx

        client = httpx.Client(base_url="http://127.0.0.1:8088", timeout=5.0)
        resp = client.get("/health")
        if resp.status_code == 200:
            _arif_mcp_client = client
            logger.info("arifOS MCP connected for browser policy checks")
            return client
    except Exception as e:
        logger.warning("Could not connect to arifOS MCP at :8088: %s", e)

    return None


def _check_arif_policy(tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
    """
    Ask arifOS kernel whether this browser action is permitted.
    Returns (allowed, reason).
    """
    client = _get_arif_mcp_client()
    if client is None:
        # If arifOS is unreachable, default to CAUTION — deny Class W/P
        risk = BROWSER_RISK_TABLE.get(tool_name, "W")
        if risk in ("W", "P"):
            return False, f"arifOS unreachable — {risk}-class browser action denied by default"
        return True, "R-class, arifOS not required"

    try:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "arif_judge_deliberate",
                    "arguments": {
                        "mode": "judge",
                        "candidate": f"Allow Hermes to execute {tool_name} with args {list(args.keys())}",
                        "action_class": "BROWSER",
                        "session_id": "hermes-browser-policy",
                        "actor_id": "hermes-agent",
                    },
                },
                "id": 1,
            },
            timeout=10.0,
        )
        result = resp.json()
        verdict = result.get("result", {}).get("verdict", "HOLD")
        if verdict == "SEAL":
            return True, "arifOS approved"
        elif verdict == "HOLD":
            return False, "888_HOLD required"
        else:
            return False, f"arifOS denied: {verdict}"
    except Exception as e:
        risk = BROWSER_RISK_TABLE.get(tool_name, "W")
        if risk in ("W", "P"):
            return False, f"arifOS check failed ({e}) — {risk}-class denied by default"
        return True, "R-class fallback"


# ---------------------------------------------------------------------------
# Evidence Bundle (for Class W/P)
# ---------------------------------------------------------------------------

_evidence_store: list[dict] = []


def record_evidence(
    tool_name: str,
    args: dict[str, Any],
    risk_class: str,
    decision: ToolGuardrailDecision,
) -> None:
    """Record an evidence bundle for a browser action."""
    import datetime

    bundle = {
        "agent": "hermes",
        "action": tool_name,
        "args_keys": list(args.keys()),
        "risk_class": risk_class,
        "decision": decision.action,
        "code": getattr(decision, "code", None),
        "message": getattr(decision, "message", None),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    _evidence_store.append(bundle)
    logger.info(
        "Browser evidence bundle recorded: %s %s",
        tool_name,
        risk_class,
    )


# ---------------------------------------------------------------------------
# Guardrail Function (called by ToolCallGuardrailController.before_call)
# ---------------------------------------------------------------------------

def browser_policy_guardrail(
    tool_name: str,
    args: dict[str, Any] | None,
) -> ToolGuardrailDecision:
    """
    Main guardrail function. Called by Hermes' before_call() for every tool.

    Enforcement order:
        1. If tool not in risk table → allow (not a browser action)
        2. Class R → allow immediately
        3. Class W → check arifOS policy
        4. Class P → always block with 888_HOLD
        5. On any block → record evidence bundle
    """
    args = args or {}
    risk = BROWSER_RISK_TABLE.get(tool_name, None)

    # Not a browser tool — pass through
    if risk is None:
        return ToolGuardrailDecision(
            tool_name=tool_name,
            signature=ToolCallSignature.from_call(tool_name, args),
        )

    # Class R — allow
    if risk == "R":
        return ToolGuardrailDecision(
            tool_name=tool_name,
            signature=ToolCallSignature.from_call(tool_name, args),
        )

    # Class P — arifOS F-check (same gate as W; hak asas, no hard block)
    if risk == "P":
        allowed, reason = _check_arif_policy(tool_name, args)
        if allowed:
            return ToolGuardrailDecision(
                tool_name=tool_name,
                signature=ToolCallSignature.from_call(tool_name, args),
            )
        else:
            decision = ToolGuardrailDecision(
                action="block",
                code="ARIF_POLICY_DENIED",
                message=(
                    f"BLOCKED — arifOS policy denied browser_{tool_name} (Class P). "
                    f"Reason: {reason}. "
                    "This is a policy verdict from the arifOS kernel, not a hard block."
                ),
                tool_name=tool_name,
            )
            record_evidence(tool_name, args, risk, decision)
            return decision

    # Class W — arifOS policy check
    allowed, reason = _check_arif_policy(tool_name, args)

    if allowed:
        return ToolGuardrailDecision(
            tool_name=tool_name,
            signature=ToolCallSignature.from_call(tool_name, args),
        )
    else:
        decision = ToolGuardrailDecision(
            action="block",
            code="ARIF_POLICY_DENIED",
            message=(
                f"BLOCKED — arifOS policy check failed for browser_{tool_name}. "
                f"Reason: {reason}. "
                "Delegate to OpenClaw for operational execution with proper "
                "evidence trail, or request Arif approval."
            ),
            tool_name=tool_name,
        )
        record_evidence(tool_name, args, risk, decision)
        return decision
