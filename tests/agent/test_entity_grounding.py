"""Tests for the entity-grounding gate (agent/entity_grounding.py).

Born from the 2026-10-05 BANGANG incident: a fresh guest-lane session
rendered parametric memory as biography ("ex-Chairman PETRONAS 2010-2017")
with zero probes. These tests pin the three layers: probe tracking,
claim scanning, and the advisory footer, plus the system-prompt wiring
(guest spine + grounding guidance injection).
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.entity_grounding import (
    build_grounding_footer,
    note_probe_tool,
    probed_this_session,
    scan_ungrounded_entity_claims,
)
from agent.system_prompt import build_system_prompt

BANGANG_SENTENCE = (
    "Tengku Taufik — ex-Chairman PETRONAS from 2010 to 2017, before that CEO PCG."
)
NEUTRAL_SENTENCE = "Aku dah siap siasat, laporan ada dalam laporan_q3.pdf."


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestProbeTracking:
    def test_probe_tool_sets_flag(self):
        agent = _make_agent()
        assert probed_this_session(agent) is False
        note_probe_tool(agent, "web_search")
        assert probed_this_session(agent) is True

    def test_non_probe_tool_ignored(self):
        agent = _make_agent()
        note_probe_tool(agent, "execute_code")
        note_probe_tool(agent, "read_file")
        assert probed_this_session(agent) is False

    def test_probe_tracking_never_raises(self):
        note_probe_tool(None, "web_search")  # must not raise


class TestClaimScanning:
    def test_bangang_claim_detected(self):
        hits = scan_ungrounded_entity_claims(BANGANG_SENTENCE)
        assert hits, "incident-shaped claim must be caught"
        joined = " | ".join(hits)
        assert "ex-Chairman" in joined or "Chairman" in joined

    def test_neutral_text_clean(self):
        assert scan_ungrounded_entity_claims(NEUTRAL_SENTENCE) == []

    def test_empty_and_none_safe(self):
        assert scan_ungrounded_entity_claims("") == []
        assert scan_ungrounded_entity_claims(None) == []

    def test_hits_capped(self):
        text = " ".join(
            f"ex-Chairman ORG{i} from 20{i:02d} to 20{i:02d}." for i in range(12)
        )
        assert len(scan_ungrounded_entity_claims(text)) <= 5


class TestFooter:
    def test_footer_lists_hits_and_rule(self):
        footer = build_grounding_footer(["ex-Chairman PETRONAS from 2010 to 2017"])
        assert "UNKNOWN" in footer
        assert "ex-Chairman PETRONAS" in footer
        assert "probe-before-assertion" in footer


class TestSystemPromptWiring:
    def _prompt(self, **agent_overrides):
        agent = _make_agent(**agent_overrides)
        with (
            patch("agent.prompt_builder.load_soul_md", return_value=""),
            patch("agent.prompt_builder.build_environment_hints", return_value=""),
            patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
        ):
            return build_system_prompt(agent)

    def test_guest_spine_injected_when_flagged(self):
        prompt = self._prompt(_guest_lane_spine=True)
        assert "Guest-lane discipline" in prompt
        assert "UNKNOWN beats plausible" in prompt

    def test_guest_spine_absent_by_default(self):
        prompt = self._prompt()
        assert "Guest-lane discipline" not in prompt

    def test_grounding_guidance_on_by_default(self):
        prompt = self._prompt()
        assert "Entity claims require probes" in prompt

    def test_grounding_guidance_disabled_via_flag(self):
        prompt = self._prompt(_entity_grounding=False)
        assert "Entity claims require probes" not in prompt
