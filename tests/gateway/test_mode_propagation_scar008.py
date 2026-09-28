"""SCAR-2026-09-28-008 — the classifier verdict must survive the prepare→send seam.

What this falsifies (three independent things, no grep-testing):

1. STRUCTURAL  ``GatewayTurnMixin._PreparedTurn`` declares a ``source`` field.
2. WIRING      ``_handle_message_with_agent`` rebinds ``source`` from the prepared
   turn *after* awaiting ``_hmwa_prepare_turn`` (AST-verified, order-checked).
3. BEHAVIOUR   ``apply_mode_shape`` under ``analyst`` preserves a 12-row enumeration,
   and under ``light`` provably destroys it — pinning the arithmetic that made HERMES
   look incoherent, so no one can re-narrow a cap without failing this file.

The defect this guards: Patch C (SCAR-2026-09-28-005) stamped ``source.mode`` on the
LOCAL binding inside ``_hmwa_prepare_turn``. ``_PreparedTurn`` carried no ``source``,
so the stamp died with the helper. ``_thread_metadata_for_source`` then saw mode=None,
never set ``hermes_mode``, and ``_send_boundary`` normalised to ``DEFAULT_MODE="light"``
(240 chars) for **every** reply — while ``state.db.delivery_obligations`` recorded the
untrimmed payload as ``state='delivered'``. Measured loss on 2026-09-28: 84.9% of the
day's characters, 94.6% in the hour after the kernel refactor, 47 replies shipped empty.

``DEFAULT_MODE="light"`` was itself a stop-gap — installed 2026-09-28 by Hermes under a
F13 directive, with the comment that caller enrichment of ``hermes_mode`` is "the durable
fix; this default is belt-and-suspenders until caller is patched". The patched caller was
the broken one. The stop-gap became the behaviour.

Runs WITHOUT pytest (guaranteed consumer):
    python3 tests/gateway/test_mode_propagation_scar008.py
Runs under pytest:
    /usr/local/lib/hermes-agent/venv/bin/python -m pytest tests/gateway/test_mode_propagation_scar008.py
Also wired as a behaviour gate in /root/scripts/hermes-overlay-verify.sh.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HERMES_MCP = Path("/root/.hermes/hermes_mcp")
RUN_TURN = REPO / "gateway" / "run_turn.py"

# A 12-row enumeration of the shape Arif actually asked for on 2026-09-28
# ("Siapa model Calvin Klein yang paling sado? List down all").
TABLE_HEADER = "| Rank | Model | Era |\n|---|---|---|\n"
TABLE_ROWS = "".join(
    f"| {i} | Model-{i:02d} | 20{i:02d} |\n" for i in range(1, 13)
)
ENUMERATION = (TABLE_HEADER + TABLE_ROWS).rstrip("\n")


def _load_boundary():
    """Load the live send boundary exactly as the Telegram adapter does, or fail loudly."""
    for parent in (str(HERMES_MCP), "/root/.hermes"):
        if parent not in sys.path:
            sys.path.insert(0, parent)
    try:
        import _send_boundary  # type: ignore

        return _send_boundary
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "_send_boundary", str(HERMES_MCP / "_send_boundary.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_send_boundary"] = mod
        spec.loader.exec_module(mod)
        return mod


def test_send_boundary_imports_for_real():
    """An unavailable shaper must never degrade to a silent no-op."""
    b = _load_boundary()
    assert b.import_ok is True, f"send boundary degraded: {b.import_error}"


def test_prepared_turn_carries_source():
    """AST-based: importing gateway.run_turn needs engine deps (ruamel) that the gateway
    runtime does not carry, so the field is read from the source tree instead."""
    tree = ast.parse(RUN_TURN.read_text(encoding="utf-8"), filename=str(RUN_TURN))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "_PreparedTurn"):
        fields = [
            t.id
            for st in cls.body
            if isinstance(st, ast.AnnAssign)
            for t in ([st.target] if isinstance(st.target, ast.Name) else [])
        ]
        assert "source" in fields, (
            "_PreparedTurn lost its `source` field — the classifier verdict will die in "
            "_hmwa_prepare_turn and every reply falls back to DEFAULT_MODE='light' (240 chars)"
        )
        return
    raise AssertionError("_PreparedTurn class not found in run_turn.py")


def test_caller_adopts_prepared_source_after_prepare():
    """AST guard: `source` is rebound from the prepared turn, AFTER the prepare await."""
    tree = ast.parse(RUN_TURN.read_text(encoding="utf-8"), filename=str(RUN_TURN))

    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_message_with_agent":
            func = node
            break
    assert func is not None, "_handle_message_with_agent not found"

    prepare_line = None
    adopt_line = None
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name == "_hmwa_prepare_turn":
                prepare_line = node.lineno
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "source" in targets and _reads_prepared_source(node.value):
                adopt_line = node.lineno

    assert prepare_line, "no call to _hmwa_prepare_turn in _handle_message_with_agent"
    assert adopt_line, (
        "caller never rebinds `source` from the prepared turn (SCAR-2026-09-28-008 regression)"
    )
    assert adopt_line > prepare_line, (
        f"source adopted at line {adopt_line} BEFORE prepare at {prepare_line} — ordering bug"
    )


def _reads_prepared_source(value: ast.AST) -> bool:
    """Match `prepared.source`, `prepared.source or source`, `... if prepared.source ...`."""
    for sub in ast.walk(value):
        if isinstance(sub, ast.Attribute) and sub.attr == "source":
            base = sub.value
            if isinstance(base, ast.Name) and base.id == "prepared":
                return True
    return False


try:  # xfail when pytest is present; a WARN flag for the stdlib runner
    import pytest  # type: ignore

    _known_open = pytest.mark.xfail(
        run=False,
        reason="DEFAULT_MODE='light' still punishes an absent verdict; durable fix is "
        "_send_boundary.py:99 (unknown → HOLD/announce, not the tightest cap)",
    )
except ImportError:  # the gateway runtime has no pytest — runner prints WARN instead

    def _known_open(fn):
        fn.__hermes_warn__ = True
        return fn


def test_analyst_mode_preserves_a_12_row_enumeration():
    b = _load_boundary()
    v = b.apply_mode_shape("analyst", ENUMERATION)
    assert not isinstance(v, tuple), f"boundary reported import failure: {v}"
    kept = [ln for ln in v.shaped_text.splitlines() if ln.startswith("| 12 ")]
    assert kept, (
        f"analyst cap clipped the enumeration: {len(ENUMERATION)}→{len(v.shaped_text)} chars; "
        "row 12 missing means the analyst cap is below a 12-row table"
    )
    assert v.original_len == len(ENUMERATION)


def test_length_caps_are_advisory_not_jails():
    """F13 order 2026-09-28 (~22:26 MYT): no character/length limit may silently delete substance.

    This replaces the earlier assertion that `light` provably destroyed a 12-row table —
    that destruction was the defect (84.9% of the day's chars lost, 47 empty replies), not
    a feature to pin. The cap is still measured: it must appear in the verdict as a WAIVED
    telemetry event, so the witness that found this bug keeps working.
    """
    b = _load_boundary()
    v = b.apply_mode_shape("light", ENUMERATION)
    assert not isinstance(v, tuple)
    rows = [ln for ln in v.shaped_text.splitlines() if "Model-" in ln]
    assert len(rows) == 12, f"light still clipped the enumeration: {len(rows)}/12 rows survived"
    assert any(x.startswith("length_cap_waived") for x in v.violations), (
        f"cap waiver not recorded — telemetry lost: {v.violations}"
    )


def test_jargon_stripping_survives_the_cap_removal():
    """What the caps were FOR. Substance flows; internal machinery still does not reach the human."""
    b = _load_boundary()
    draft = (
        "⚒️ REALITY > EVERYTHING · DITEMPA BUKAN DIBERI\n\n"
        "## 🪞 DECODE:\nAku tak pasti jam berapa.\n\n"
        "Option A: jalan\nOption B: tahan\nOption C: tunggu\nOption D: bubar\n"
    )
    v = b.apply_mode_shape("light", draft)
    assert not isinstance(v, tuple)
    assert "REALITY > EVERYTHING" not in v.shaped_text, "boot signature leaked to the human"
    assert "DECODE" not in v.shaped_text, "decoder header leaked to the human"
    assert "Option A" not in v.shaped_text, "ABCD menu leaked to the human"
    assert "Aku tak pasti jam berapa" in v.shaped_text, "the actual sentence was deleted too"


def test_never_ship_an_empty_reply():
    """47 replies shipped empty on 2026-09-28: stripping consumed the whole message."""
    b = _load_boundary()
    v = b.apply_mode_shape("light", "## ⚠️ Reality check\n")
    assert not isinstance(v, tuple)
    assert v.shaped_text.strip() or not v.original_len, (
        "stripping produced an empty outbound reply — silence is the worse failure"
    )


def test_none_mode_is_not_silently_the_tightest_cap():
    """An absent verdict must not buy a substance deletion.

    Before SCAR-2026-09-28-008: mode=None → DEFAULT_MODE='light' → 240 chars, i.e. the
    *absence* of a classifier answer was punished with the tightest cap in the system.
    Length caps are now advisory, so no mode can delete substance; stripping posture may
    still be the conservative one, which is the correct reading of 'unknown'.
    """
    b = _load_boundary()
    v = b.apply_mode_shape(None, ENUMERATION)
    assert not isinstance(v, tuple)
    rows = [ln for ln in v.shaped_text.splitlines() if "Model-" in ln]
    assert len(rows) == 12, (
        f"unknown mode still clipped ({len(rows)}/12 rows) — fail-safe must not truncate"
    )


ADAPTER = REPO / "plugins" / "platforms" / "telegram" / "adapter.py"
AUTHZ = REPO / "gateway" / "authz_mixin.py"


def test_boundary_call_site_passes_lane():
    """SCAR-2026-09-28-006 was dead for 5 hours: F13 ratified a room-aware ceiling, but the
    call site passed only (mode, content), so `_lane_should_clamp(None)` never matched.
    AST-checked (3 args) rather than grepped, so reformatting cannot hide a regression."""
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"), filename=str(ADAPTER))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == "apply_mode_shape"
    ]
    assert calls, "no apply_mode_shape() call found in the Telegram adapter"
    for c in calls:
        assert len(c.args) >= 3 or any(k.arg == "lane" for k in c.keywords), (
            f"apply_mode_shape at line {c.lineno} does not pass a lane — SCAR-006 unreachable"
        )


class _Skipped(Exception):
    """Engine deps (ruamel et al) are absent from the gateway's standalone runtime."""

# NOTE: an earlier version of this file refused bot-authored DMs before generation. It broke
# tests/gateway/test_bot_loop_guard.py::test_admitted_bot_traffic_is_cut_at_the_budget, which
# asserts bot DMs ARE admitted and metered by budget — a deliberate design, not an oversight.
# The undeliverable-reply waste (34 Forbidden sends / 3h, 55,676 chars abandoned) is therefore
# a bot_loop_guard BUDGET question (KVM8 config.yaml carries no `gateway.bot_loop_guard` block
# at all, so the guard runs on defaults), not an authorization question. That test is the pin.


if __name__ == "__main__":
    import traceback

    def _is_known_open(fn) -> bool:
        if getattr(fn, "__hermes_warn__", False):
            return True
        return any("xfail" in repr(getattr(m, "name", m)) for m in getattr(fn, "pytestmark", []))

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = warned = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except _Skipped as exc:
            skipped += 1
            print(f"  SKIP  {name}: {exc}")
        except AssertionError as exc:
            if _is_known_open(fn):
                warned += 1
                print(f"  WARN  {name}: {exc}")
            else:
                failed += 1
                print(f"  FAIL  {name}: {exc}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}:\n{traceback.format_exc()}")
    status = "ALL-GREEN" if failed == 0 else f"{failed} FAILED — DO NOT RESTART"
    print(f"{status} (mode propagation, SCAR-2026-09-28-008) · {warned} known-open WARN · {skipped} SKIP")
    sys.exit(1 if failed else 0)
