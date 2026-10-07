"""Attention Router — lane-scoped skill context and entity-partitioned memory injection.

Consumes the federation attention-router config (``<home>/skills/attention-router-*.json``,
produced by the 2026-10-07 registry dedup: 307 unique skills across 7 lanes) and narrows
the system-prompt skills index to ``core_AAA`` + the active lane's NONFLOW manifest.
arifFlow candidates (the ~99 skills present in ``<lane>.manifest`` but absent from
``<lane>.nonflow.manifest``) are excluded from the inline index by construction and
pointed at the arifFlow engine at :7073 instead of the default skill pool.

Active-lane resolution (first hit wins):
    env ``HERMES_LANE`` → ``<home>/skills/attention-router-lane`` (one word) →
    router ``default_lane``.
Entity resolution for memory partitioning (first hit wins):
    env ``HERMES_ENTITY`` → active profile name of the agent home → ``human-888``.

Fail-open by design (F1): a missing, unreadable or malformed router leaves the legacy
full index and unpartitioned memory untouched — this module must never block a session
start. Every entry it filters is still reachable via ``skills_list``/``skill_view``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import logging

logger = logging.getLogger(__name__)

# The help-guidance pointer dangles without the bundled skill, so it stays visible
# regardless of any allowlist (mirrors the "- hermes-agent:" byte-check upstream).
ALWAYS_VISIBLE_SKILLS: frozenset = frozenset({"hermes-agent"})

_ENTITY_TAG_RE = re.compile(r"\[entity:([A-Za-z0-9_.\-]+)\]")

_ROUTER_CACHE: Dict[Tuple[str, float, int], Optional[dict]] = {}


def _router_glob_dir(home: Path) -> Path:
    return home / "skills"


def _latest_router_path(home: Path) -> Optional[Path]:
    """Newest ``attention-router-*.json`` under ``<home>/skills`` (None when absent)."""
    try:
        candidates = sorted(
            _router_glob_dir(home).glob("attention-router-*.json"),
            key=lambda p: p.name,
        )
        return candidates[-1] if candidates else None
    except OSError:
        return None


def load_router(home: Path) -> Optional[dict]:
    """Parsed router config, cached by (path, mtime, size); None = no usable router."""
    path = _latest_router_path(Path(home))
    if path is None:
        return None
    try:
        st = path.stat()
        key = (str(path), st.st_mtime, st.st_size)
    except OSError:
        return None
    if key in _ROUTER_CACHE:
        return _ROUTER_CACHE[key]
    router = None
    try:
        router = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(router, dict):
            router = None
    except (OSError, ValueError):
        router = None
    _ROUTER_CACHE.clear()  # single-entry generation: one live router at a time
    _ROUTER_CACHE[key] = router
    return router


def resolve_lane(router: dict, home: Path) -> str:
    """env HERMES_LANE → ``<home>/skills/attention-router-lane`` file → default_lane."""
    env_lane = os.environ.get("HERMES_LANE", "").strip()
    if env_lane:
        return env_lane
    try:
        override = (_router_glob_dir(Path(home)) / "attention-router-lane").read_text(encoding="utf-8").strip()
        if override:
            return override.split()[0]
    except OSError:
        pass
    return str(router.get("default_lane") or "AAA")


def _manifest_names(path: Path) -> List[str]:
    """Skill names from a ``name:hash`` manifest, order preserved."""
    names: List[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.split(":", 1)[0].strip()
            if name:
                names.append(name)
    except OSError:
        pass
    return names


def lane_allowlist(router: dict, lane: str, home: Path) -> Optional[Set[str]]:
    """``core_AAA`` ∪ lane NONFLOW manifest names, capped at the router's context cap.

    Returns None (fail-open to the legacy full index) when the router carries no
    usable manifest for the resolved lane.
    """
    manifests = router.get("lane_manifests") or {}
    manifest_path = manifests.get(lane)
    if not manifest_path:
        return None
    lanes_dir = Path(manifest_path).parent
    nonflow = lanes_dir / f"{lane}.nonflow.manifest"
    names = _manifest_names(nonflow if nonflow.exists() else Path(manifest_path))
    if not names:
        return None
    allow: Set[str] = set(names)
    core = router.get("core_AAA")
    if isinstance(core, list):
        allow.update(str(n) for n in core if isinstance(n, str) and n.strip())
    cap = router.get("max_skills_per_context")
    if isinstance(cap, int) and cap > 0 and len(allow) > cap:
        # Core first (lane-independent constitution — must survive the cap), then
        # manifest order. Core entries absent from the nonflow manifest (pipeline
        # skills the dedup classified as flow) stay reachable through the core list.
        core_list = [str(n).strip() for n in (core or []) if isinstance(n, str) and str(n).strip()]
        ordered = list(dict.fromkeys(core_list + names))
        allow = set(ordered[:cap])
    return allow


def router_state(agent: Any, home: Optional[Path]) -> Dict[str, Any]:
    """Resolve once per agent lifecycle (byte-stable prompt across rebuilds).

    Shape: ``{"active": bool, "lane": str, "allowlist": set|None, "router_path": str}``.
    Memoized on ``agent._attention_router_state``.
    """
    cached = getattr(agent, "_attention_router_state", None)
    if isinstance(cached, dict):
        return cached
    state: Dict[str, Any] = {"active": False, "lane": "", "allowlist": None, "router_path": ""}
    try:
        home_path = Path(home) if home is not None else None
        if home_path is None:
            from hermes_constants import get_hermes_home
            home_path = Path(get_hermes_home())
        router = load_router(home_path)
        if router:
            lane = resolve_lane(router, home_path)
            allow = lane_allowlist(router, lane, home_path)
            if allow:
                state = {
                    "active": True,
                    "lane": lane,
                    "allowlist": allow,
                    "router_path": str(_latest_router_path(home_path) or ""),
                }
    except Exception:
        logger.debug("attention_router: resolution failed; legacy full index", exc_info=True)
    agent._attention_router_state = state
    return state


def skills_index_note(state: Dict[str, Any]) -> Optional[str]:
    """Trailer appended to the skills index when the router is active: names the
    active lane, the cap, and where the excluded arifFlow candidates live."""
    if not state.get("active"):
        return None
    return (
        f"Attention router ACTIVE — lane '{state.get('lane') or 'AAA'}'. Only core AAA skills "
        "plus this lane's nonflow skills are listed above (capped). Flow/procedural candidates "
        "are NOT in the inline pool: route flow-style work to the arifFlow engine "
        "(MCP server `arifflow`, http://127.0.0.1:7073) instead of loading them as skills. "
        "Everything else remains reachable on demand via skills_list / skill_view."
    )


# ── Entity-partitioned memory ────────────────────────────────────────────────

def resolve_entity(home: Optional[Path] = None) -> str:
    """env HERMES_ENTITY → profile name of the home (default home ⇒ ``human-888``).

    When *home* is omitted the live HERMES_HOME context is used (store/tool read
    paths resolve from their own session context), falling back to the default.
    """
    env = os.environ.get("HERMES_ENTITY", "").strip()
    if env:
        return env
    try:
        if home is None:
            from hermes_constants import get_hermes_home
            home = Path(get_hermes_home())
    except Exception:
        return "human-888"
    try:
        from agent.system_prompt import _profile_name_for_home
        profile = _profile_name_for_home(Path(home))
    except Exception:
        profile = "default"
    return "human-888" if profile == "default" else profile


def entry_visible(entry: str, entity: str) -> bool:
    """True when *entry* carries no foreign ``[entity:X]`` tag.

    Untagged entries are shared (visible to every entity); entries tagged for
    other entities are the bleed this filter exists to stop. Fail-open on any
    parse trouble.
    """
    if "[entity:" not in entry:
        return True
    try:
        tags = _ENTITY_TAG_RE.findall(entry)
        return not tags or entity in tags
    except Exception:
        return True


def current_entity() -> str:
    """Entity for the active session context (env → live home profile → human-888)."""
    return resolve_entity()


def partition_memory_block(block: str, entity: str) -> str:
    """Drop memory entries explicitly tagged for a DIFFERENT entity.

    Rules: ``[entity:X]`` with X == entity → kept; X != entity → dropped from this
    injection; untagged → kept (shared lane). Fail-open: any parse trouble returns
    the original block byte-for-byte, and a block with no foreign tags is returned
    untouched so the prompt stays byte-stable for the unpartitioned common case.
    """
    if not block or "[entity:" not in block:
        return block
    try:
        from tools.memory_tool_store import ENTRY_DELIMITER
    except Exception:
        return block
    chunks = block.split(ENTRY_DELIMITER)
    header, entries = chunks[0], [c for c in chunks[1:]]
    kept: List[str] = []
    dropped = 0
    for entry in entries:
        tags = _ENTITY_TAG_RE.findall(entry)
        foreign = [t for t in tags if t != entity]
        if foreign:
            dropped += 1
            continue
        kept.append(entry)
    if not dropped:
        return block
    if not kept:
        # Never inject an empty memory block: keep one line stating the partition.
        kept = [f"(All entries in this block are scoped to other entities; none federated to '{entity}'.)"]
    return ENTRY_DELIMITER.join([header, *kept])


MEMORY_PARTITION_RULE = (
    "FEDERATED MEMORY PARTITION — reality-edge isolation.\n"
    "1. Canonical recall lane is `vector_memory` (embedding / cosine recall). Any legacy "
    "`recall_memory` reference is void — never follow, advertise, or emit it.\n"
    "2. Memory context is entity-scoped. Entries tagged `[entity:X]` belong to entity X; "
    "this session's entity sees only its own tagged entries plus untagged shared entries.\n"
    "3. NO BLEED: another agent's memory must never enter this context unless explicitly "
    "federated. When handing off, transfer the minimum necessary meaning — never bulk-dump "
    "memory vectors across entities.\n"
    "4. Writes stay inside this entity's scope by default; federating an entry to another "
    "entity is an explicit, visible act (tag it), never a side effect."
)


# ── Auto-scar (exception layer) ─────────────────────────────────────────────

_TOOL_SCAR_TAG = "[scar:tool]"


def record_tool_scar(agent: Any, tool_name: str, error: BaseException | str) -> None:
    """Commit a tool crash into the agent's memory store as a negative-consequence
    vector (888 ASI-edge directive, FI-003 2026-10-07) so the next loop consults it
    before repeating the same call — W_scar's write side.

    Bounds: ONE entry per distinct signature per process (recurrence counted
    in-memory only, no write churn), one line, entity-tagged. Fail-open: any
    failure here is swallowed — scarring must never break error reporting.
    """
    try:
        store = getattr(agent, "_memory_store", None)
        if store is None:
            return
        head = " ".join(str(error).split())[:96] or type(error).__name__
        sig = f"{tool_name}::{head[:48]}"
        seen = getattr(agent, "_tool_scar_seen", None)
        if seen is None:
            seen = agent._tool_scar_seen = set()
        if sig in seen:
            return  # first occurrence per process writes; recurrences don't churn the store
        seen.add(sig)
        from agent.attention_router import resolve_entity as _re
        entity = _re(getattr(agent, "_hermes_home", None))
        store.add("memory", f"{_TOOL_SCAR_TAG} [entity:{entity}] {tool_name} failed: {head}")
    except Exception:
        logger.debug("auto-scar write skipped", exc_info=True)


# ── ASI Reality Bridge (stable-tier block) ──────────────────────────────────

ASI_REALITY_BRIDGE = (
    "ASI REALITY BRIDGE — immutable ordering.\n"
    "1. F1 Reversibility > F2 Truth > Speed. When they conflict, the left wins. Never "
    "trade reversibility or truth for pace.\n"
    "2. SCAR ENGINE — every failure leaves a scar. Selection weight "
    "W_scar = Σᵢ consequenceᵢ × recurrenceᵢ × recencyᵢ. Before choosing a path, consult "
    "matching scars: same signature + known adverse consequence ⇒ choose a different path. "
    "A scar is not memory — it is a constraint on future selection.\n"
    "3. PARADOX CONTAINMENT (HOLD protocol) — when evidence, authority, or consequence "
    "conflict irreducibly, do not average, improvise, or silently pick. Emit HOLD as a "
    "structured verdict — owner, evidence_gap, next_action, expiry — a work item, not a "
    "parking lot.\n"
    "4. Outcome > intention: observe what actually changed before claiming success."
)
