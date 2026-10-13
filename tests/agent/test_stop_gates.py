"""Unit tests for entity-grounding stop gate (Genesis SCAR-007 §3.2 — F13 directive 2026-10-05).

The entity-grounding gate forces one probe-first continuation when the final
response carries confident specifics (date range, tenure length, title chain,
or quantitative claim) but no probe flag has been set on this agent instance
yet. This test file pins the detector/nudge contract that the gate depends on.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# The detector and nudge live in agent/turn_stop_gates. Importing at module
# scope avoids the agent.* path-cycling import that turn_stop_gates itself
# guards (its module docstring says: "Nothing here imports agent.conversation_loop
# at module level (cycle)."). We re-import lazily so the test still works when
# turn_stop_gates is being modified in-place during the fix/entity-grounding-gate
# branch.
def _gates():
    from agent import turn_stop_gates as gates
    return gates


def _agent(flag: bool = False, prior_nudges: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        _entity_grounding_probed=flag,
        _entity_grounding_nudges=prior_nudges,
    )


# ---------------------------------------------------------------------------
# needs_probe — true cases
# ---------------------------------------------------------------------------

class TestNeedsProbeTriggers:
    def test_date_range_triggers(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(
            _agent(), "Tenure Chairman PETRONAS: 1 April 2010 - 1 April 2017"
        ) is True

    def test_tenure_length_triggers(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(
            _agent(), "Served 7 tahun on the board"
        ) is True

    def test_title_chain_triggers(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(
            _agent(), "Tan Sri Tengku bin Kamadjaja Aziz is now CEO"
        ) is True

    def test_quantitative_triggers(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(
            _agent(), "Stake worth RM 5.3 million worth"
        ) is True

    def test_combined_patterns_trigger(self):
        gates = _gates()
        text = "Tengku Muhammad Taufik served 7 years from April 2010 - April 2017"
        assert gates._entity_grounding_needs_probe(_agent(), text) is True


# ---------------------------------------------------------------------------
# needs_probe — false cases
# ---------------------------------------------------------------------------

class TestNeedsProbeNoTrigger:
    def test_empty_text(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(_agent(), "") is False

    def test_none_text(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(_agent(), None) is False

    def test_non_string_text(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(_agent(), 12345) is False

    def test_prose_without_specifics(self):
        gates = _gates()
        assert gates._entity_grounding_needs_probe(
            _agent(), "Tengku Taufik is a real person we should investigate."
        ) is False

    def test_already_probed_blocks(self):
        gates = _gates()
        text = "1 April 2010 - 1 April 2017 served 7 tahun"
        assert gates._entity_grounding_needs_probe(_agent(flag=True), text) is False


# ---------------------------------------------------------------------------
# nudge — produces a probe-first instruction
# ---------------------------------------------------------------------------

class TestNudgeShape:
    def test_nudge_includes_partial_name(self):
        gates = _gates()
        text = "Tan Sri Tengku bin Kamadjaja Aziz served 7 tahun as chairman"
        nudge = gates._entity_grounding_nudge(_agent(), text)
        assert nudge is not None
        assert "Tan Sri Tengku" in nudge
        assert "probe" in nudge.lower()
        assert "scar-007" in nudge
        assert "zai_search" in nudge.lower() or "web_search" in nudge.lower()

    def test_nudge_uses_h_class_form(self):
        gates = _gates()
        text = "Tengku Taufik 1 April 2010 - 1 April 2017 chairman"
        nudge = gates._entity_grounding_nudge(_agent(), text)
        assert nudge is not None
        # Per scar-007 §3.1 + F13 explicit preference (today's directive)
        assert "hipotesis" in nudge.lower() or "h-class" in nudge.lower()

    def test_nudge_none_without_partial_name(self):
        gates = _gates()
        # Prose without any partial-name pattern — no extractable identifier
        nudge = gates._entity_grounding_nudge(_agent(), "The economy grew 5% last year.")
        assert nudge is None

    def test_nudge_none_for_empty_text(self):
        gates = _gates()
        assert gates._entity_grounding_nudge(_agent(), "") is None
        assert gates._entity_grounding_nudge(_agent(), None) is None


# ---------------------------------------------------------------------------
# Gate wiring — apply_stop_gates should trigger entity-grounding
# ---------------------------------------------------------------------------

class TestGateWiring:
    def test_gate_triggers_continue_with_nudge(self):
        gates = _gates()
        agent = _agent(flag=False)
        # A final response that contains both a partial name + confident specifics
        # without primary source — the gate must trigger.
        final_response = (
            "Tengku Muhammad Taufik served 7 tahun as CEO from "
            "April 2010 - April 2017 and was Tan Sri Datuk Seri Utama."
        )
        final_msg = {"role": "assistant", "content": final_response}
        messages: list = []
        verdict = gates.apply_stop_gates(
            agent,
            final_msg,
            final_response=final_response,
            messages=messages,
            conversation_history=[],
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        )
        assert verdict.continue_turn is True
        assert verdict.final_response is None
        assert any(
            isinstance(msg, dict) and msg.get("_entity_grounding_synthetic")
            for msg in messages
        )
        assert agent._entity_grounding_probed is True
        assert agent._entity_grounding_nudges >= 1
        # The "scar-007" reference lives in the nudge message that goes back
        # to the loop, not in the discarded final_msg content.
        sentinel = next(
            (
                m.get("content", "")
                for m in messages
                if isinstance(m, dict) and m.get("_entity_grounding_synthetic")
            ),
            "",
        )
        assert "scar-007" in sentinel

    def test_gate_skips_when_already_probed(self):
        gates = _gates()
        agent = _agent(flag=True)
        # Even with confident specifics, the gate must NOT re-trigger.
        final_response = "Tan Sri served 7 tahun from April 2010 - April 2017"
        final_msg = {"role": "assistant", "content": final_response}
        messages: list = []
        verdict = gates.apply_stop_gates(
            agent,
            final_msg,
            final_response=final_response,
            messages=messages,
            conversation_history=[],
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        )
        # Should fall through to the no-continue default — no entity-grounding row
        assert not any(
            isinstance(msg, dict) and msg.get("_entity_grounding_synthetic")
            for msg in messages
        )

    def test_gate_passes_prose_without_specifics(self):
        gates = _gates()
        agent = _agent(flag=False)
        # Prose answer, no specifics → no entity-grounding trigger
        final_response = "Tengku Taufik appears to be a real person."
        final_msg = {"role": "assistant", "content": final_response}
        messages: list = []
        verdict = gates.apply_stop_gates(
            agent,
            final_msg,
            final_response=final_response,
            messages=messages,
            conversation_history=[],
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        )
        assert not any(
            isinstance(msg, dict) and msg.get("_entity_grounding_synthetic")
            for msg in messages
        )


# ---------------------------------------------------------------------------
# discard helper — ungrounded draft must be marked not promotable
# ---------------------------------------------------------------------------

class TestDiscardHelper:
    def test_discard_clears_content_and_marks_finish_reason(self):
        gates = _gates()
        final_msg = {
            "role": "assistant",
            "content": "Tengku Taufik 1 April 2010 - 1 April 2017 7 tahun",
            "reasoning": "made up date",
            "reasoning_content": "stuff",
        }
        gates._discard_entity_grounding_draft(final_msg)
        assert final_msg["content"] == "[discarded: entity-grounding probe required]"
        assert final_msg["finish_reason"] == "entity_grounding_required"
        assert final_msg["_entity_grounding_discarded"] is True
        # Reasoning content must be cleared so it cannot be promoted to fallback
        assert "reasoning" not in final_msg
        assert "reasoning_content" not in final_msg


# ---------------------------------------------------------------------------
# Regex sanity — the detector regexes must match what scar-007 calls out
# ---------------------------------------------------------------------------

class TestRegexSanity:
    def test_date_range_regex_matches_scar007_example(self):
        # Scar-007 §3.2 calls out: "1 April 2010 → 1 April 2017"
        from agent.turn_stop_gates import _ENTITY_DATE_RANGE_RE
        assert _ENTITY_DATE_RANGE_RE.search("1 April 2010 - 1 April 2017") is not None
        assert _ENTITY_DATE_RANGE_RE.search("1 Apr 2010 to 1 Apr 2017") is None

    def test_tenure_regex_matches_english_and_bm(self):
        from agent.turn_stop_gates import _ENTITY_TENURE_LEN_RE
        assert _ENTITY_TENURE_LEN_RE.search("7 tahun") is not None
        assert _ENTITY_TENURE_LEN_RE.search("7 years") is not None
        assert _ENTITY_TENURE_LEN_RE.search("7 yrs") is not None

    def test_partial_name_regex_captures_honorifics(self):
        from agent.turn_stop_gates import _ENTITY_PARTIAL_NAME_RE
        m = _ENTITY_PARTIAL_NAME_RE.search("Tengku Taufik was chairman")
        assert m is not None
        assert m.group(1) == "Tengku Taufik"
        m = _ENTITY_PARTIAL_NAME_RE.search("Tan Sri Sidek Hassan served")
        assert m is not None
        assert m.group(1).startswith("Tan Sri Sidek")