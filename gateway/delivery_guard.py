"""gateway.delivery_guard — fail-closed outbound delivery target verification.

WHY THIS EXISTS
  Cron/sub-agent delivery resolved the destination from ``origin.chat_id``. When
  that origin was the bot's own id (8410138119), Telegram returned
  ``Forbidden: the bot can't send messages to the bot`` and the output never
  reached the human. A send whose destination is retrieved from session state
  must not authorize itself.

MECHANISM
  Every outbound Telegram target is resolved to a concrete id and checked
  BEFORE any network call:

    1. bot ids declared in /root/.hermes/IDENTITY_LOCK.json  -> HOLD
    2. ids present in the active profile channel_directory   -> ALLOW
    3. everything else, or an unreadable lock/directory      -> HOLD (fail closed)

  The allowlist is the live channel directory minus the lock's bot ids. It is
  read, never hardcoded, so a human id change cannot silently invalidate the
  gate. An unknown id is not "probably fine"; it is unverifiable, so it is
  refused.

CALLERS
  - cron.scheduler_delivery._resolve_single_delivery_target (pre-send)
  - plugins.platforms.telegram.adapter._send_text_locked / _send_local_file
    (last line of defence; catches ad-hoc sends that bypass cron resolution)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

IDENTITY_LOCK = Path("/root/.hermes/IDENTITY_LOCK.json")
PROFILE_ROOT = Path("/usr/local/lib/hermes-agent/profiles")
ACTIVE_PROFILE_FILE = Path("/usr/local/lib/hermes-agent/active_profile")


def _load_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _active_profile_dir() -> Optional[Path]:
    """Resolve the active Hermes profile directory without trusting CWD."""
    name = ""
    try:
        if ACTIVE_PROFILE_FILE.is_file():
            name = ACTIVE_PROFILE_FILE.read_text().strip()
    except Exception:
        name = ""
    if not name:
        name = os.environ.get("HERMES_PROFILE", "aaa-hermes")
    candidate = PROFILE_ROOT / name
    return candidate if candidate.is_dir() else None


def _bot_ids(lock: dict) -> set[str]:
    out: set[str] = set()
    for rec in (lock.get("bots") or {}).values():
        if isinstance(rec, dict) and rec.get("id") is not None:
            out.add(str(rec["id"]).strip())
    return out


def _directory_ids(directory: dict) -> dict[str, str]:
    """id -> kind for every DM/group in the live channel directory."""
    out: dict[str, str] = {}
    platforms = directory.get("platforms") or {}
    entries = platforms.get("telegram") or []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        kind = str(entry.get("type") or "").strip().lower()
        if raw_id is None or kind not in ("dm", "group"):
            continue
        base = str(raw_id).split(":", 1)[0].strip()
        if base:
            out[base] = kind
    return out


def target_allowed(platform: str, chat_id: str) -> Tuple[bool, str]:
    """Return (allowed, reason). Never raises; unknown state fails closed."""
    platform = str(platform or "").strip().lower()
    base = str(chat_id or "").split(":", 1)[0].strip()

    if platform != "telegram":
        # Only Telegram has an identity lock + channel directory on this box.
        # Other platforms keep their existing resolver behaviour.
        return True, f"platform {platform!r} is outside the telegram delivery guard"

    if not base:
        return False, "empty delivery target — cannot verify, refusing to send"

    lock = _load_json(IDENTITY_LOCK)
    if lock is None:
        return False, f"identity lock unreadable at {IDENTITY_LOCK} — fail closed"

    bots = _bot_ids(lock)
    if base in bots:
        return False, (
            f"{base} is a bot id in {IDENTITY_LOCK} — Telegram refuses "
            "bot-to-bot delivery; refusing before the network call"
        )

    profile_dir = _active_profile_dir()
    if profile_dir is None:
        return (
            False,
            f"active profile directory not found under {PROFILE_ROOT} — fail closed",
        )

    directory_path = profile_dir / "channel_directory.json"
    directory = _load_json(directory_path)
    if directory is None:
        return False, f"channel directory unreadable at {directory_path} — fail closed"

    known = _directory_ids(directory)
    kind = known.get(base)
    if kind in ("dm", "group"):
        return True, f"{base} is a known telegram {kind} in {directory_path.name}"

    return False, (
        f"{base} is not a known human/group in {directory_path.name} — fail closed "
        "(unknown destination is not permission to send)"
    )
