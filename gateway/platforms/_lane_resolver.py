"""Lightweight lane resolver for outbound shape enforcement.

Why this exists
---------------
The Telegram adapter boundary (`hermes_mcp._send_boundary.apply_mode_shape`)
needs the lane/room identity to apply LANE_CEILING — e.g., SADO group chats
must clamp to "light" regardless of what the mode classifier decided.

Before SCAR-2026-09-28-006, `metadata["hermes_lane"]` was never populated
upstream, so the boundary never knew what room a message was headed to.

This module resolves lane_id from the same lanes.yaml registry the
lane_switch plugin uses, with the same Pass 1 (user+chat exact) -> Pass 2
(user-only) -> guest fallback precedence. Keeps a single source of truth
for lane identity; the plugin does lane-card injection, this does shape
geometry.

Standalone by design — does not import the lane_switch plugin (avoids the
plugin's transitive hooks/skills imports). Imports yaml lazily so the
gateway can run even when PyYAML is absent on this machine.

Public API
----------
``resolve_lane_id(user_id, chat_id) -> str | None``
    Returns lane_id ("arif" | "syed_dm" | "syed_sado" | "guest" | ...) or
    None if lanes.yaml cannot be loaded. Caller treats None as "no ceiling
    information" and proceeds without LANE_CEILING.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_LANES_YAML_CANDIDATES = (
    Path("/root/HERMES/lanes/lanes.yaml"),
    Path("/root/.hermes/lanes/lanes.yaml"),
)

_cache: dict = {"data": None, "mtime": 0.0}


def _load_lanes_yaml() -> dict:
    """Load lanes.yaml with mtime-based caching. Returns {} on any failure."""
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    for path in _LANES_YAML_CANDIDATES:
        try:
            if not path.exists():
                continue
            mtime = path.stat().st_mtime
            if _cache["data"] is not None and mtime == _cache["mtime"]:
                return _cache["data"]
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            _cache["data"] = data
            _cache["mtime"] = mtime
            return data
        except Exception:
            continue
    return {}


def resolve_lane_id(user_id: str = "", chat_id: str = "") -> Optional[str]:
    """Return lane_id for a (user_id, chat_id) tuple, mirroring lane_switch.

    Precedence (matches lane_switch plugin):
      Pass 1: exact user+chat match (e.g. Syed DM vs Syed in SADO group).
      Pass 2: user-only fallback (known user in unmapped room) — first lane
              in registry order wins; group-safe registers should be listed
              before deeper personal registers.
      Default: None (caller treats as "no ceiling").

    Returns None when lanes.yaml is absent or unreadable — boundary falls
    back to DEFAULT_MODE, no lane ceiling applied this turn.
    """
    reg = _load_lanes_yaml()
    lanes = reg.get("lanes", {}) if isinstance(reg, dict) else {}
    if not lanes:
        return None
    uid = str(user_id or "")
    cid = str(chat_id or "")

    # Pass 1: exact user+chat match
    if uid and cid:
        for lane_id, cfg in lanes.items():
            if lane_id == "guest":
                continue
            if not isinstance(cfg, dict):
                continue
            tr = cfg.get("triggers", {}) or {}
            uids = tr.get("telegram_user_ids", []) or []
            cids = tr.get("telegram_chat_ids", []) or []
            if uid in uids and cid in cids:
                return lane_id

    # Pass 2: user-only fallback
    if uid:
        for lane_id, cfg in lanes.items():
            if lane_id == "guest":
                continue
            if not isinstance(cfg, dict):
                continue
            tr = cfg.get("triggers", {}) or {}
            uids = tr.get("telegram_user_ids", []) or []
            if uid in uids:
                return lane_id

    return None


__all__ = ["resolve_lane_id"]