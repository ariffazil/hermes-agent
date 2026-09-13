"""Mem0 memory plugin — MemoryProvider interface.

Server-side fact extraction and semantic search via the Mem0 Platform API (cloud), a
self-hosted Mem0 server (MEM0_HOST, HTTP), or OSS Memory. Secrets live in $HERMES_HOME/.env
(MEM0_API_KEY, MEM0_HOST); settings in $HERMES_HOME/mem0.json via `hermes memory setup`:
mode ("platform"|"oss"), host, user_id (canonical id across gateways; unset → gateway-native
id), agent_id. MEM0_* env vars remain a fallback.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider, spawn_context_thread
from agent.secret_scope import get_secret
from tools.registry import tool_error
from utils import atomic_json_write, read_json_or_empty

logger = logging.getLogger(__name__)

# Circuit breaker: after _BREAKER_THRESHOLD consecutive failures, pause API
# calls for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD, _BREAKER_COOLDOWN_SECS, _PREFETCH_WAIT_SECS = 5, 120, 3
_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")
# Placeholder user_id. initialize() treats it as "no operator-configured user_id"
# so legacy mem0.json files written by the wizard don't override gateway-native ids.
_DEFAULT_USER_ID = "hermes-user"

# sync_turn sends the whole turn to the backend for fact extraction. OSS embedding
# models often have small context windows (bge-small-zh-v1.5: 512 tokens ≈ 500 chars;
# jina-embeddings-v3: 8192), and oversized turns make backend.add() raise — Ollama
# answers HTTP 500, hosted APIs return INPUT_TOKEN_LIMIT_EXCEEDED — which _try only
# logs, silently dropping the turn's memory extraction. Cap each message up front.
# The default fits a 512-token embedder (measured: 450 OK, 600 -> HTTP 500 on
# bge-small-zh-v1.5:f16); ``sync_max_chars`` in mem0.json raises it for larger windows.
_SYNC_MSG_MAX_CHARS = 450

# ---------------------------------------------------------------------------
# AAA carry (2026-09-11): PROMOTION GATE — "Witness is cheap, Memory is expensive."
# Root cause of mem0 inflation (11,288 points in 6 days ≈ 1,900/day): sync_turn
# sent EVERY turn to the LLM extractor with infer=True, so episodic chit-chat
# ("shared a photo of nasi lemak", "sent a test voice message") was promoted to
# permanent semantic memory alongside sealed doctrine. Recall then drowned in
# episodic noise, and the same doctrine was re-extracted in 2-8 paraphrases.
#
# Arif's four gates, applied as a cheap deterministic pre-filter (no LLM cost,
# no latency). A rejected turn stays in the conversation log — it is witnessed,
# just not promoted. Every rejection is appended to a ledger (F11 auditability).
#   Gate C — does this change future decisions?  trivial/no  -> witness only
#   Gate D — new primitive or new example?       example     -> witness only
# ---------------------------------------------------------------------------
_GATE_LEDGER_NAME = "mem0-promotion-ledger.jsonl"

# Machine-generated scaffolding. These are NOT human content and NOT facts about
# the world — they are runtime plumbing that repeats identically every fire.
# Falsification test (2026-09-11, 4,999 real turns) found 1,249 of these being
# promoted because their *system instructions* contain "must"/"never"/"should",
# tripping the signal rule. A daily cron prompt re-extracted every day is a
# duplication engine, not a memory. Reject before the signal check.
_GATE_SCAFFOLD_PATTERNS = (
    re.compile(r"\[IMPORTANT: You are running as a scheduled cron job", re.I),
    re.compile(r"\[ASYNC DELEGATION COMPLETE\b", re.I),
    re.compile(r"The user sent a (?:document|file|attachment):", re.I),
    re.compile(r"^\s*<memory-context>", re.I),
    re.compile(r"\[System note:", re.I),
    re.compile(r"^\s*(?:tool[_-]?result|function[_-]?result)\b", re.I),
)

# Genuine human mid-turn steering, wrapped in a marker. NOT scaffolding — it
# carries the same authority as the original request, so a correction here must
# not be dropped. Unwrap and evaluate the inner content.
_GATE_OOB_WRAP = re.compile(
    r"\[OUT-OF-BAND USER MESSAGE.*?\](.*?)(?:\[/OUT-OF-BAND USER MESSAGE\]|$)", re.S)

# Identity prefixes injected by the gateway (e.g. "[ARIF|267378578] ...").
# These ARE real human content — strip the prefix and evaluate what remains.
_GATE_IDENTITY_PREFIX = re.compile(r"^\s*\[[A-Z0-9_@\.\-\s|]{1,60}\]\s*")

# Pure-noise turn shapes: media plumbing, greetings, acks, routine status.
_GATE_NOISE_PATTERNS = (
    re.compile(r"^\s*(MEDIA|ATTACHMENT|FILE):", re.I),
    re.compile(r"^\s*\[(?:image|photo|sticker|video|audio|document|voice)\b", re.I),
    re.compile(r"^\s*(?:salam|hi|hello|hey|ok|okay|oke|ya|yup|yes|no|thx|thanks|thank you|ty|k|👍|❤|🔥)[\s!.,]*$", re.I),
    re.compile(r"^\s*(?:good\s*(?:morning|night|afternoon|evening)|selamat\s*(?:pagi|malam))[\s!.,]*$", re.I),
    re.compile(r"\b(?:generated and shared a media file|sent a test (?:voice )?message|"
               r"(?:requested|generated) an? (?:ai[- ]generated )?(?:image|photo|picture|sticker|video)|"
               r"shared a (?:photo|screenshot|image|picture) of|forwarded a photo)\b", re.I),
    re.compile(r"\b(?:no new traces from|health (?:check|probe) (?:showed|reported)|"
               r"all (?:organs|surfaces) (?:are )?(?:healthy|green))\b", re.I),
)

# Substantive markers that OVERRIDE a length-based rejection — a short turn can
# still carry doctrine or a durable correction. Never gate these out.
_GATE_SIGNAL_PATTERNS = (
    re.compile(r"\b(SEALED|canonical|doctrine|invariant|primitive|axiom|EUREKA|"
               r"F(?:1[0-3]|[1-9])\b|HARAM|scar|constitution|ratified|DITEMPA)\b"),
    re.compile(r"\b(prefer|always|never|must|should|stop|don'?t|jangan|sentiasa|ingat)\b", re.I),
    re.compile(r"\b(remember|recall|note that|correct(?:ion)?|actually|sebenarnya)\b", re.I),
    # Bahasa Melayu markers. Arif's primary register is BM, and a short BM
    # correction ("Hang salah. Abang sado x amik kasut hitam") is exactly the
    # high-value/low-length content the length rule would otherwise drop.
    # Caught by the honest eval on 4,999 real turns (2026-09-11) as the sole
    # substantive loss before this rule existed.
    re.compile(r"\b(salah|silap|betul|betol|bukan|tak\s|tidak|x\s|jangan|"
               r"sebenarnya|ingat|sentiasa|selalu|kena|perlu|mesti)\b", re.I),
)


def _promotion_gate(user_content: str, *, min_chars: int, ledger_path: Path | None = None) -> tuple[bool, str]:
    """Decide whether a turn deserves promotion to semantic memory.

    Returns (allow, reason). Deterministic and cheap by design — this runs on
    every turn, so it must never call a model or block.
    """
    text = (user_content or "").strip()
    if not text:
        return False, "empty"

    # Unwrap genuine human mid-turn steering before anything else, so the
    # markers never make it look like scaffolding.
    if _GATE_OOB_WRAP.search(text):
        inner = " ".join(m.group(1).strip() for m in _GATE_OOB_WRAP.finditer(text)).strip()
        if inner:
            text = inner

    # Machine scaffolding is rejected outright: it is runtime plumbing that
    # repeats verbatim every fire, not a fact about the world.
    for pat in _GATE_SCAFFOLD_PATTERNS:
        if pat.search(text):
            return False, "scaffolding"

    # Gateway identity prefix ("[ARIF|267378578] ...") is real human content —
    # strip it so the prefix can't inflate the substance measurement.
    text = _GATE_IDENTITY_PREFIX.sub("", text, count=1).strip()

    # Strip tool-result / scaffolding bulk so length reflects real human content.
    body = re.sub(r"<(tool[_-]?result|system-reminder|memory-context)\b[^>]*>.*?</\1>", " ", text,
                  flags=re.S | re.I)
    body = re.sub(r"https?://\S+", " ", body).strip()

    # Signal is checked FIRST and overrides noise shape. A turn can look routine
    # and still carry a sealed verdict — falsification test (2026-09-11) caught
    # "Milestone Receipt: Cognition Spine Sealed" being dropped by the
    # all-organs-healthy noise rule. Losing a SEAL is worse than keeping noise.
    if any(p.search(body) for p in _GATE_SIGNAL_PATTERNS):
        return True, "promoted_signal"

    for pat in _GATE_NOISE_PATTERNS:
        if pat.search(body):
            return False, "noise_shape"

    if len(body) < min_chars:
        # Gate C: too little substance to change a future decision.
        return False, "insufficient_substance"

    return True, "promoted"


def _log_gate_decision(ledger_path: Path | None, allowed: bool, reason: str, preview: str) -> None:
    """Append the decision to the promotion ledger (F11). Never raises."""
    if ledger_path is None:
        return
    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "verdict": "PROMOTE" if allowed else "WITNESS_ONLY",
                "reason": reason,
                "preview": preview[:160],
            }) + "\n")
    except Exception:  # ledger must never break the memory path
        logger.debug("mem0 promotion ledger write failed", exc_info=True)


# Sentence ends recognized when trimming a synced message. Deliberately unordered:
# the LAST boundary of ANY kind wins, so one CJK stop early in a mixed-script turn
# cannot outrank a Latin stop near the end of the window. ``".\n"`` is not listed —
# its index can never exceed the bare ``"."`` it starts with.
_SYNC_SENTENCE_ENDS = ("。", "！", "？", ".", "!", "?")

def _truncate_for_sync(text: str, max_len: int = _SYNC_MSG_MAX_CHARS) -> str:
    """Cap a synced message at its last sentence boundary within ``max_len``.

    Short messages pass through unchanged; long ones keep the last complete
    sentence inside the window so fact extraction still sees coherent statements,
    with a hard cut as fallback when no boundary exists (or one only appears in
    the first third of the window, which usually means unsegmented input).
    """
    if len(text) <= max_len:
        return text
    window = text[:max_len]
    cut = max(window.rfind(sep) for sep in _SYNC_SENTENCE_ENDS)
    if cut > max_len // 3:
        return text[:cut + 1]
    return text[:max_len]


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    err_str = str(exc).lower()
    return type(exc).__name__ in _CLIENT_ERROR_TYPES or any(s in err_str for s in ("404", "not found", "valid uuid"))


def _load_config() -> dict:
    """Env vars provide defaults; $HERMES_HOME/mem0.json overrides individual keys.
    Layering avoids a silent failure when the JSON file exists but lacks fields
    like ``api_key`` that the user set in ``.env``."""
    from hermes_constants import get_hermes_home
    # Identity (user/agent id), host and mode are .env values like the key: read them through the
    # profile scope too, or a secondary profile's memories land in the default profile's account.
    # A scope-less multiplex caller raises here on purpose — that is a spawn-site bug, and
    # swallowing it would silently route the turn's memories to the default profile.
    config = {"mode": get_secret("MEM0_MODE", "") or "platform", "host": get_secret("MEM0_HOST", "") or "",
              "agent_id": get_secret("MEM0_AGENT_ID", "") or "hermes", "oss": {}}
    if user_id := get_secret("MEM0_USER_ID", ""):  # only when explicitly configured, so initialize() can fall back to the gateway-native id
        config["user_id"] = user_id
    file_cfg = read_json_or_empty(get_hermes_home() / "mem0.json")
    config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
    # MEM0_API_KEY authenticates the Platform and self-hosted HTTP backends; pure OSS mode builds its
    # backend from the local ``oss`` config and has no platform credential to resolve, so a profile
    # scope WITHOUT the key must still load an OSS config (#99121 as it stands today: the caller is
    # scoped, the scope is just empty). Decided after mem0.json overrode the env defaults because
    # the file may be what selects ``oss``. Scope-less callers already raised above.
    if config.get("mode", "platform") == "oss":
        config.setdefault("api_key", "")
    elif not config.get("api_key"):
        config["api_key"] = get_secret("MEM0_API_KEY", "")
    return config


def _schema(name: str, description: str, properties: dict[str, tuple[str, str]], required: list[str]) -> dict:
    props = {k: {"type": t, "description": d} for k, (t, d) in properties.items()}
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}


TOOL_SCHEMAS = [
    _schema("mem0_search", "Search the user's memories by meaning; returns facts ranked by relevance. Use this before answering any question that may depend on what you know about the user (preferences, facts, history, people, projects, past decisions). For multi-part or multi-hop questions, call it several times — vary the wording and run follow-up searches on what earlier results reveal; one search is rarely enough.",
            {"query": ("string", "What to search for."), "top_k": ("integer", "Max results (default: 10, max: 50)."), "rerank": ("boolean", "Rerank results for relevance (default: false, platform mode only).")}, ["query"]),
    _schema("mem0_add", "Store a durable fact about the user, verbatim (no LLM extraction). Call this the moment the user states a lasting preference, correction, decision, or personal detail worth recalling on future turns — don't wait to be asked to remember. Skip transient chit-chat and facts you've already stored.",
            {"content": ("string", "The fact to store.")}, ["content"]),
    _schema("mem0_update", "Replace the text of an existing memory by its ID (take the ID from a mem0_search result). Use when a stored fact has changed or was wrong — correct it in place instead of adding a duplicate.",
            {"memory_id": ("string", "Memory UUID to update."), "text": ("string", "New text content.")}, ["memory_id", "text"]),
    _schema("mem0_delete", "Delete a memory by its ID (take the ID from a mem0_search result). Use when a stored fact is obsolete or the user asks you to forget it; prefer mem0_update if the fact merely changed.",
            {"memory_id": ("string", "Memory UUID to delete.")}, ["memory_id"]),
]

_PROMPT_BODY = (
    "You have persistent memory of this user from past conversations. You should call mem0_search before answering anything that could depend on prior context (the user's preferences, facts, history, people, projects, or earlier decisions) — do not rely on the chat window alone, and do not assume you have no memory.\n"
    "For multi-part or multi-hop questions, run several searches with different wording/angles and follow-up searches on what the first results surface; one search is rarely enough. Keep searching until you have every fact the question needs before you answer.\n"
    "Tools: mem0_search to find memories, mem0_add to store facts, mem0_update and mem0_delete to manage by ID."
)


class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search (platform, self-hosted or OSS)."""

    def __init__(self):
        self._config = self._backend = self._sync_thread = self._prefetch_thread = None
        self._mode, self._api_key, self._host, self._user_id, self._agent_id = "platform", "", "", _DEFAULT_USER_ID, "hermes"
        self._rerank_default, self._channel = False, "cli"  # channel = gateway name (cli/telegram/discord/...)
        self._sync_max_chars = _SYNC_MSG_MAX_CHARS
        self._prefetch_query = self._prefetch_result = ""
        self._prefetch_done = self._atexit_registered = False
        self._consecutive_failures, self._breaker_open_until = 0, 0.0  # circuit breaker state
        self._breaker_lock, self._sync_lock, self._prefetch_lock = threading.Lock(), threading.Lock(), threading.Lock()
        # AAA carry (2026-09-11): promotion gate state (configured in initialize()).
        self._gate_enabled, self._gate_min_chars, self._gate_ledger = True, 60, None

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        if cfg.get("mode", "platform") == "oss":
            return bool(cfg.get("oss", {}).get("vector_store"))
        return bool(cfg.get("api_key") or cfg.get("host"))  # platform needs a key; self-hosted a host (key optional with AUTH_DISABLED)

    def save_config(self, values, hermes_home):
        """Merge-write config to $HERMES_HOME/mem0.json."""
        config_path = Path(hermes_home) / "mem0.json"
        atomic_json_write(config_path, {**read_json_or_empty(config_path), **values}, mode=0o600)

    def get_config_schema(self):
        api_key_required = _load_config().get("mode", "platform") != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 server URL (leave blank for cloud)", "required": False, "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from ._setup import post_setup
        post_setup(hermes_home, config)

    def _oss_hint(self, template: str, default: str = "vector store") -> str:
        """OSS-only hint; ``{vs}`` is the configured vector-store provider. "" in other modes."""
        return template.format(vs=self._config.get("oss", {}).get("vector_store", {}).get("provider", default)) if self._mode == "oss" else ""

    def _create_backend(self):
        # Lazy-install the mem0 SDK before the backend imports it (honors security.allow_lazy_installs);
        # on failure the backend import raises the canonical error, captured below.
        with suppress(Exception):
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("memory.mem0", prompt=False)
        try:
            from . import _backend
            if self._mode == "oss":
                return _backend.OSSBackend(self._config.get("oss", {}))
            return _backend.SelfHostedBackend(self._api_key, self._host) if self._host else _backend.PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

    def _is_breaker_open(self) -> bool:
        """True while the breaker is tripped; an expired cooldown resets the failure count."""
        with self._breaker_lock:
            if self._consecutive_failures >= _BREAKER_THRESHOLD and time.monotonic() < self._breaker_open_until:
                return True
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._consecutive_failures = 0
            return False

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if any(s in str(exc).lower() for s in ("connection", "refused", "timeout")):
            msg += self._oss_hint(" (check that {vs} is running)")
        return msg

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures = count = self._consecutive_failures + 1
            if count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        if count >= _BREAKER_THRESHOLD:
            hint = self._oss_hint(" Check that your {vs} vector store is running and reachable.", "unknown")
            logger.warning("Mem0 circuit breaker tripped after %d consecutive failures. Pausing API calls for %ds.%s", count, _BREAKER_COOLDOWN_SECS, hint)

    def _try(self, call, log, msg: str):
        """Background-path wrapper: run ``call`` under the breaker; on error log ``msg`` and return None."""
        try:
            result = call()
        except Exception as e:
            self._record_failure()
            log(msg, e)
            return None
        self._record_success()
        return result

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = cfg = _load_config()
        self._mode, self._api_key, self._host, self._agent_id = cfg.get("mode", "platform"), cfg.get("api_key", ""), cfg.get("host", ""), cfg.get("agent_id", "hermes")
        # user_id precedence: operator-configured (env/mem0.json) > gateway-native id (kwargs) > _DEFAULT_USER_ID.
        # The literal placeholder counts as unset so wizard users still get gateway-native ids.
        configured = cfg.get("user_id")
        self._user_id = (None if configured == _DEFAULT_USER_ID else configured) or kwargs.get("user_id") or _DEFAULT_USER_ID
        # AAA carry (2026-09-11): principal_map unifies the channel identities that belong to
        # the SAME human into one recall pool. Without it, search is scoped to a single
        # gateway-native user_id, so Telegram writes (267378578) are invisible to CLI recall
        # (hermes-user) and vice versa — 37.8% of one person's memory was unreachable.
        # Identities NOT listed stay isolated by design (F6 MARUAH air-gap between people).
        _pmap = cfg.get("principal_map") or {}
        if isinstance(_pmap, dict) and _pmap:
            self._user_id = str(_pmap.get(str(self._user_id), self._user_id))
        # Persisted rerank preference: default for mem0_search when the model omits ``rerank``. Platform-only.
        _rr = cfg.get("rerank", False)
        self._rerank_default = _rr.lower() in ("true", "1", "yes") if isinstance(_rr, str) else bool(_rr)
        self._channel = kwargs.get("platform") or "cli"
        self._sync_max_chars = int(cfg.get("sync_max_chars") or _SYNC_MSG_MAX_CHARS)
        # AAA carry (2026-09-11): promotion gate config. Defaults ON — the whole
        # point of the carry. ``promotion_gate`` in mem0.json can disable or tune it:
        #   {"promotion_gate": {"enabled": true, "min_chars": 60}}
        _pg = cfg.get("promotion_gate")
        if isinstance(_pg, dict):
            self._gate_enabled = bool(_pg.get("enabled", True))
            try:
                self._gate_min_chars = int(_pg.get("min_chars", 60))
            except (TypeError, ValueError):
                self._gate_min_chars = 60
        else:
            self._gate_enabled, self._gate_min_chars = True, 60
        if self._gate_enabled:
            from hermes_constants import get_hermes_home
            with suppress(Exception):
                self._gate_ledger = Path(get_hermes_home()) / _GATE_LEDGER_NAME
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _search(self, query: str, top_k: int = 10, rerank: bool = False, backend=None) -> list:
        # Scoped to user_id only — by design — so recall surfaces memories from any gateway/agent under this
        # principal; writes attach agent_id and metadata.channel so narrower views remain possible at query time.
        return (backend or self._backend).search(query, filters={"user_id": self._user_id}, top_k=top_k, rerank=rerank)

    def _add(self, messages: list, infer: bool):
        metadata = {"channel": self._channel} if self._channel else {}
        return self._backend.add(messages, user_id=self._user_id, agent_id=self._agent_id, infer=infer, metadata=metadata)

    def system_prompt_block(self) -> str:
        # Mirror _create_backend precedence (oss > host > platform). Rerank is a Mem0 Platform feature only.
        mode_label = "OSS (self-hosted)" if self._mode == "oss" else "self-hosted (HTTP API)" if self._host else "platform (cloud API)"
        rerank_note = " Rerank is available on search." if (self._mode == "platform" and not self._host) else ""
        return f"# Mem0 Memory\nActive. Mode: {mode_label}. User: {self._user_id}.\n{_PROMPT_BODY}{rerank_note}"

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _consume_prefetch_result(self, query: str) -> str | None:
        """Pop the finished prefetch body for ``query`` (None if absent or still running)."""
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result, self._prefetch_result, self._prefetch_done = self._prefetch_result, "", False
            return result

    def _start_prefetch(self, query: str) -> None:
        backend = self._backend
        if not query or backend is None or self._is_breaker_open():
            return

        def _run():
            results = self._try(lambda: self._search(query, backend=backend), logger.debug, "Mem0 prefetch failed: %s")
            lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
            body = "## Mem0 Memory\n" + "\n".join(f"- {l}" for l in lines) if lines else ""
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result, self._prefetch_done = body, True

        with self._prefetch_lock:
            # Same query already answered or still in flight: don't restart it.
            if self._prefetch_query == query and (self._prefetch_done or (self._prefetch_thread and self._prefetch_thread.is_alive())):
                return
            self._prefetch_query, self._prefetch_result, self._prefetch_done = query, "", False
            self._prefetch_thread = t = spawn_context_thread(_run, name="mem0-prefetch")
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        if (cached := self._consume_prefetch_result(query)) is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        return self._consume_prefetch_result(query) or ""  # slow backend: skip injection; mem0_search remains the backstop

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._backend is None or self._is_breaker_open():
            return

        # AAA carry (2026-09-11): promotion gate. Fail-open — any gate error
        # must never cost the turn its memory (F1 AMANAH: prefer a little
        # noise over silent loss of a real correction).
        if self._gate_enabled:
            try:
                allow, reason = _promotion_gate(
                    user_content,
                    min_chars=self._gate_min_chars,
                    ledger_path=self._gate_ledger,
                )
                _log_gate_decision(self._gate_ledger, allow, reason, (user_content or "").strip())
                if not allow:
                    logger.debug("mem0 promotion gate: witness-only (%s)", reason)
                    return
            except Exception:
                logger.debug("mem0 promotion gate failed open", exc_info=True)

        def _sync():
            if self._backend is not None:
                messages = [
                    {"role": "user", "content": _truncate_for_sync(user_content, self._sync_max_chars)},
                    {"role": "assistant", "content": _truncate_for_sync(assistant_content, self._sync_max_chars)},
                ]
                self._try(lambda: self._add(messages, infer=True), logger.warning, "Mem0 sync failed: %s")

        with self._sync_lock:
            prev = self._sync_thread
            if prev and prev.is_alive():
                prev.join(timeout=5.0)
                if prev.is_alive():  # still busy after the wait: skip to avoid duplicate ingestion
                    return
            self._sync_thread = spawn_context_thread(_sync, name="mem0-sync")
            self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    # -- tool handlers: (required params, error label, body, client-error policy) ---
    # Client errors (bad ID / not found) never trip the breaker, except for mem0_add
    # where they count as failures; update/delete answer them with "Memory not found".

    def _tool_search(self, args: dict) -> str:
        top_k = max(1, min(int(args.get("top_k", 10)), 50))
        rerank_raw = args.get("rerank", self._rerank_default)
        rerank = rerank_raw.lower() not in ("false", "0", "no") if isinstance(rerank_raw, str) else bool(rerank_raw)
        results = self._search(args["query"], top_k, rerank)
        if not results:
            return json.dumps({"result": "No relevant memories found."})
        items = [{"id": r.get("id"), "memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
        return json.dumps({"results": items, "count": len(items)})

    def _tool_add(self, args: dict) -> str:
        result = self._add([{"role": "user", "content": args["content"]}], infer=False)
        event_id = result.get("event_id") if isinstance(result, dict) else None
        # Cloud add is async (server-side extraction); OSS and self-hosted store synchronously.
        msg = "Fact stored." if (self._mode == "oss" or self._host) else "Fact queued for storage."
        return json.dumps({"result": msg, "event_id": event_id})

    _TOOL_HANDLERS = {
        "mem0_search": (("query",), "Search failed", _tool_search, "skip"),
        "mem0_add": (("content",), "Failed to store", _tool_add, "count"),
        "mem0_update": (("memory_id", "text"), "Update failed", lambda self, a: json.dumps(self._backend.update(a["memory_id"], a["text"])), "not_found"),
        "mem0_delete": (("memory_id",), "Delete failed", lambda self, a: json.dumps(self._backend.delete(a["memory_id"])), "not_found"),
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None:
            err = getattr(self, "_init_error", "unknown error")
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{self._oss_hint(' Check that {vs} is running and reachable.')}"})
        if self._is_breaker_open():
            return json.dumps({"error": f"Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically.{self._oss_hint(' Check that your {vs} is running.')}"})
        if tool_name not in self._TOOL_HANDLERS:
            return tool_error(f"Unknown tool: {tool_name}")
        required, label, body, on_client_error = self._TOOL_HANDLERS[tool_name]
        if missing := next((k for k in required if not args.get(k, "")), None):
            return tool_error(f"Missing required parameter: {missing}")
        try:
            result = body(self, args)
        except Exception as e:
            client = _is_client_error(e)
            if client and on_client_error == "not_found":
                return tool_error(f"Memory not found: {args['memory_id']}")
            if not client or on_client_error == "count":
                self._record_failure()
            return tool_error(self._format_error(label, e))
        self._record_success()
        return result

    def _shutdown_backend(self):
        with suppress(Exception):
            if self._backend:
                self._backend.close()
                self._backend = None

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_backend()


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user, verbatim (no LLM extraction). "
        "Call this the moment the user states a lasting preference, correction, "
        "decision, or personal detail worth recalling on future turns — don't "
        "wait to be asked to remember. Skip transient chit-chat and facts you've "
        "already stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a mem0_search "
        "result). Use when a stored fact is obsolete or the user asks you to "
        "forget it; prefer mem0_update if the fact merely changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search the user's memories by meaning; returns facts ranked by "
        "relevance. Use this before answering any question that may depend on "
        "what you know about the user (preferences, facts, history, people, "
        "projects, past decisions). For multi-part or multi-hop questions, "
        "call it several times — vary the wording and run follow-up searches "
        "on what earlier results reveal; one search is rarely enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "rerank": {"type": "boolean", "description": "Rerank results for relevance (default: false, platform mode only)."},
        },
        "required": ["query"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Replace the text of an existing memory by its ID (take the ID from a "
        "mem0_search result). Use when a stored fact has changed "
        "or was wrong — correct it in place instead of adding a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}
# ---- END PLUGIN-COMPAT ----
