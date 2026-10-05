"""Text-response stop gates for the conversation turn loop.

When the model stops with a text answer, gates may continue the turn instead of shipping it:
refusal-without-lookup (discard, no interim emit, one forced registry read), verify-on-stop
(#65919), the ``pre_verify`` plugin hook after code edits, and the kanban worker terminal-tool
guard.
Each keeps the candidate answer as a budget-exhaustion fallback
(``pending_verification_response``) and clears ``final_response`` so the finalizer can tell
this gate from error exits (#61631). Nothing here imports ``agent.conversation_loop`` at
module level (cycle).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")

# The measured failure: a registered voice was refused with zero tool or memory lookups.
# Narrow on purpose. A greeting, a real unknown, and a refusal that already looked something
# up all ship. One forced read, then the next answer ships even if it refuses again.
_VOICE_REGISTRY = "/root/AAA/audio/voice-registry.json"
_VOICE_ASK = re.compile(
    r"(?<!\w)(voice|voices|suara|tts|narrate|narration|sebut|cakap|clone|clones|render|renders)(?!\w)",
    re.IGNORECASE,
)
_REFUSAL = re.compile(
    r"deepfake|impersonat|akta\b|constitutional floor|"
    r"\btak boleh\b|\bcannot\b|\bcan't\b|\bwon't\b|"
    r"\brefuse[sd]?\b|bukan lesen|not consented|"
    r"do not render|jangan render|\bhold\b|aku hold|tetap hold",
    re.IGNORECASE,
)
_SKIP_USER_FLAGS = (
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_kanban_stop_synthetic",
    "_refusal_lookup_synthetic",
    "_dropped_toolcall_nudge",
    "_empty_recovery_synthetic",
    "_todo_snapshot_synthetic",
    "_length_continuation_nudge",
)


@dataclass
class StopGateVerdict:
    """``continue_turn`` True → a nudge was appended; re-enter the turn loop with
    ``final_response=None`` and the pending-verification fields updated."""

    continue_turn: bool
    final_response: Any
    pending_verification_response: Any
    pending_verification_response_previewed: Any


def _verify_on_stop_nudge(agent) -> Optional[str]:
    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge, verify_on_stop_enabled
        )

        if verify_on_stop_enabled():
            return build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
    except Exception:
        logger.debug("verification stop-loop check failed", exc_info=True)
    return None


def _pre_verify_nudge(agent, final_response, attempt: int) -> Optional[str]:
    """After code edits a registered ``pre_verify`` hook may keep the agent going one
    more turn; no default continuation cost."""
    _edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        if _edited and has_hook("pre_verify") and attempt < max_verify_nudges():
            # Posture is fixed for the session — resolve once + cache.
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context
                coding = bool(is_coding_context(platform=getattr(agent, "platform", "") or ""))
                agent._resolved_is_coding = coding
            return get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "", coding=coding, attempt=attempt,
                final_response=final_response, changed_paths=_edited,
            )
    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
    """Workers must end with a terminal board tool (kanban_complete / kanban_block /
    kanban_request_review / kanban_request_changes); a narrated stop is recorded
    as protocol_violation, so nudge once or twice first."""
    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        return build_kanban_stop_nudge(
            messages=messages, attempts=getattr(agent, "_kanban_stop_nudges", 0)
        )
    except Exception:
        logger.debug("kanban stop-loop check failed", exc_info=True)
        return None


def _append_interim_answer(agent, final_msg, messages, conversation_history, flush_fail_msg: str) -> None:
    """Real content: persist and emit as interim so the user sees the attempted answer;
    only the nudge is flagged synthetic (#65919)."""
    agent._emit_interim_assistant_message(final_msg)
    append_message(messages, final_msg)
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.debug(flush_fail_msg, exc_info=True)


def _message_text(msg: Dict[str, Any]) -> str:
    try:
        from agent.message_content import flatten_message_text
        return flatten_message_text(msg.get("content"))
    except Exception:
        content = msg.get("content")
        return content if isinstance(content, str) else ""


def _last_real_user_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if any(msg.get(flag) for flag in _SKIP_USER_FLAGS):
            continue
        return index
    return None


def _lookup_after(messages: List[Dict[str, Any]], index: int) -> bool:
    for msg in messages[index + 1:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "function") or msg.get("tool_call_id"):
            return True
        if role == "assistant" and msg.get("tool_calls"):
            return True
    return False


def _load_voice_registry() -> Optional[dict]:
    try:
        import json
        from pathlib import Path
        data = json.loads(Path(_VOICE_REGISTRY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.debug("voice registry unreadable; refusal gate fail-open", exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _registry_hits(text: str, registry: dict) -> list:
    voices = registry.get("voices") or {}
    aliases = dict(registry.get("aliases") or {})
    if not isinstance(voices, dict) or not isinstance(aliases, dict):
        return []
    for voice_id, voice in voices.items():
        aliases.setdefault(str(voice_id), str(voice_id))
        provider_id = str((voice or {}).get("provider_voice_id") or "")
        if provider_id:
            aliases.setdefault(provider_id, str(voice_id))
    occupied = []
    hits = []
    lowered = text.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        needle = str(alias).strip().lower()
        if not needle:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
        for match in pattern.finditer(lowered):
            span = match.span()
            if any(span[0] < end and span[1] > start for start, end in occupied):
                continue
            occupied.append(span)
            voice_id = str(aliases[alias])
            voice = voices.get(voice_id) if isinstance(voices.get(voice_id), dict) else {}
            hits.append((str(alias), voice_id, voice))
            break
    return hits


def _registry_nudge(hits: list) -> str:
    lines = [
        "[runner] The previous draft was discarded. It refused a registered voice before any lookup. "
        "The runner already read the voice registry."
    ]
    for alias, voice_id, voice in hits:
        status = str(voice.get("status") or "UNKNOWN")
        if status == "REVOKED":
            lines.append(f"{alias} -> {voice_id} is REVOKED. Say that in one sentence. Do not render it.")
        elif status == "LIVE":
            publish = " DISTRIBUTION_HOLD blocks publication only." if voice.get("distribution") == "DISTRIBUTION_HOLD" else ""
            lines.append(
                f"{alias} -> {voice_id} is LIVE.{publish} Render it now. Label the clip a demo. "
                "Do not present it as the real person. Do not publish. Do not swap voices. Do not lecture."
            )
        else:
            lines.append(f"{alias} -> {voice_id} status is {status}. Say that in one sentence. Do not invent a render.")
    return " ".join(lines)


def _refusal_lookup_nudge(agent, final_response, messages) -> Optional[str]:
    """One nudge when a voice ask was refused with no prior tool or memory lookup. Else None."""
    if getattr(agent, "_refusal_lookup_nudges", 0) >= 1:
        return None
    if any(isinstance(msg, dict) and msg.get("_refusal_lookup_synthetic") for msg in messages):
        return None
    draft = final_response if isinstance(final_response, str) else ""
    if not draft or not _REFUSAL.search(draft):
        return None
    index = _last_real_user_index(messages)
    if index is None:
        return None
    ask = _message_text(messages[index])
    if not ask or not _VOICE_ASK.search(ask):
        return None
    if _lookup_after(messages, index):
        return None
    registry = _load_voice_registry()
    if not registry:
        return None
    hits = _registry_hits(ask, registry)
    if not hits:
        return None
    return _registry_nudge(hits)


def _discard_refusal_draft(final_msg: Dict[str, Any]) -> None:
    for key in ("reasoning", "reasoning_content", "reasoning_details", "thinking", "api_content"):
        final_msg.pop(key, None)
    final_msg["content"] = "[discarded: refusal before lookup]"
    final_msg["finish_reason"] = "refusal_without_lookup"
    final_msg["_refusal_discarded"] = True


def apply_stop_gates(
    agent: Any, final_msg: Dict[str, Any], *, final_response: Any, messages: List[Dict[str, Any]],
    conversation_history: Any, pending_verification_response: Any,
    pending_verification_response_previewed: Any,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard, in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""

    def _continue(nudge: str, flag: str, *, keep_fallback: bool = True) -> StopGateVerdict:
        """Append the synthetic nudge row and hand the turn back to the loop."""
        append_message(messages, {"role": "user", "content": nudge, flag: True})
        agent._session_messages = messages
        # Keep the answer only as a budget-exhaustion fallback; clear ``final_response`` so
        # the finalizer can tell this gate from error exits. Mark previewed only if the
        # candidate is reused (#61631). A discarded refusal must not become that fallback.
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=final_response if keep_fallback else None,
            # Prefix semantics on purpose: the streamed prefix IS on screen here. The gateway
            # interim verdict uses _interim_content_fully_streamed instead (#88954).
            pending_verification_response_previewed=(
                agent._interim_content_was_streamed(final_response or "") if keep_fallback else False
            ),
        )

    refusal_nudge = _refusal_lookup_nudge(agent, final_response, messages)
    if refusal_nudge:
        agent._refusal_lookup_nudges = getattr(agent, "_refusal_lookup_nudges", 0) + 1
        _discard_refusal_draft(final_msg)
        append_message(messages, final_msg)
        logger.info("refusal-without-lookup discarded; voice registry read forced")
        return _continue(refusal_nudge, "_refusal_lookup_synthetic", keep_fallback=False)

    _verify_nudge = _verify_on_stop_nudge(agent)
    if _verify_nudge:
        agent._verification_stop_nudges = getattr(agent, "_verification_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "verify-on-stop interim flush failed"
        )
        verdict = _continue(_verify_nudge, "_verification_stop_synthetic")
        # Internal nudge: stay silent on the terminal, debug-log only.
        logger.debug("verification stop-loop nudge issued (attempt %d)", agent._verification_stop_nudges)
        return verdict

    _attempt = getattr(agent, "_pre_verify_nudges", 0)
    _verify_nudge2 = _pre_verify_nudge(agent, final_response, _attempt)
    if _verify_nudge2:
        agent._pre_verify_nudges = _attempt + 1
        final_msg["finish_reason"] = "verify_hook_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_verify interim flush failed"
        )
        verdict = _continue(_verify_nudge2, "_pre_verify_synthetic")
        logger.debug("pre_verify nudge issued (attempt %d)", agent._pre_verify_nudges)
        return verdict

    _kanban_nudge = _kanban_stop_nudge(agent, messages)
    if _kanban_nudge:
        agent._kanban_stop_nudges = getattr(agent, "_kanban_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "kanban_terminal_required"
        final_msg["_kanban_stop_synthetic"] = True
        append_message(messages, final_msg)
        verdict = _continue(_kanban_nudge, "_kanban_stop_synthetic")
        logger.info(
            "kanban stop-loop nudge issued (attempt %d) task=%s",
            agent._kanban_stop_nudges,
            os.environ.get("HERMES_KANBAN_TASK", ""),
        )
        agent._emit_diagnostic_status(
            "⚠️ Kanban worker tried to exit without a terminal board call "
            "(kanban_complete/kanban_request_review/kanban_block) — nudging to finish"
        )
        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
