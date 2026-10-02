"""Regression tests for repetition-dominated response handling.

A response dominated by verbatim repeated text must be discarded whether it
ends normally or at the output cap. The turn aborts with a clear user-facing
error instead of persisting and delivering the pathological fragment.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.repetition_guard import STOP_PATH_MIN_CHARS
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID

# The exact sentence from the #86581 incident.
_INCIDENT_ECHO = "好，你幫我更改成 Google Gemini 4 31B。"


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _response(
    content,
    *,
    finish_reason=FINISH_REASON_LENGTH,
    response_id=PARTIAL_STREAM_STUB_ID,
):
    from tests.agent.test_run_agent import _mock_assistant_msg

    return SimpleNamespace(
        id=response_id,
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content),
            finish_reason=finish_reason,
        )],
        usage=None,
    )


def _run(agent, message):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


class TestContinuationRepetitionGuard:
    def test_repetition_dominated_truncation_aborts(self, loop_agent):
        echo = _INCIDENT_ECHO * 2000
        loop_agent.client.chat.completions.create.side_effect = [_response(echo)]

        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert echo not in (result["final_response"] or "")
        # The pathological fragment must NOT be appended to the history.
        assert not any(
            isinstance(m, dict) and m.get("_length_continuation_fragment")
            for m in result["messages"]
        )
        # Exactly one API call — no continuation was attempted.
        assert loop_agent.client.chat.completions.create.call_count == 1

    @pytest.mark.parametrize("requested_repeat", [False, True], ids=["loop", "repeat-on-request"])
    def test_repetition_dominated_stop_response_aborts(self, loop_agent, requested_repeat):
        if requested_repeat:
            # Asked-for repetition (identical lines, ~3.6k chars) is below runaway scale: a
            # completed answer must be delivered, not discarded.
            echo = "Hello world, this is a sentence the user asked me to repeat many times.\n" * 50
        else:
            paragraph = (
                "A long paragraph that should never be delivered hundreds of times "
                "when a model enters a repetition loop.\n"
                "The second line makes this a multiline repeating unit.\n"
            )
            echo = paragraph * 500
            assert len(echo) >= STOP_PATH_MIN_CHARS
        loop_agent.client.chat.completions.create.side_effect = [
            _response(echo, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "write me a long report")

        assert loop_agent.client.chat.completions.create.call_count == 1
        if requested_repeat:
            assert result["completed"] is True
            assert result["final_response"] == echo.strip()
            return
        assert result["completed"] is False
        assert result["partial"] is True
        assert (result["failure_reason"], result["failure_retryable"]) == ("truncated", True)
        assert "Repetition" in (result["final_response"] or "")
        assert not any(
            isinstance(m, dict) and m.get("content") == echo
            for m in result["messages"]
        )

    def test_legit_truncation_still_continues(self, loop_agent):
        # Ordinary short truncated fragments still get continuation retries.
        loop_agent.client.chat.completions.create.side_effect = [
            _response("part one "), _response("part two "),
            _response("part three "), _response("part four."),
        ]

        result = _run(loop_agent, "write me a long report")

        assert result["partial"] is True
        assert loop_agent.client.chat.completions.create.call_count == 4


def _locked(total_chars: int, run: int, token: str = "punya") -> str:
    """A reply carrying an intra-line single-word lock, in the measured incident line shape.

    Several distinct prose lines plus one line holding the whole lock — the shape that let the
    132 KB incident through: >=5 non-empty lines that are >50% distinct, so the distinct-line
    ratio in is_runaway_repetition reads it as ordinary prose.
    """
    lock_line = "Aku nampak apa yang hang rasa. " + " ".join([token] * run) + " itu sahaja."
    body = "\n".join([
        "OK, dah cukup probe. Ini ringkasan sebenar.",
        "---",
        "**Apa data nampak, dan apa dia tak nampak.**",
        "Panjang chat: 26 Ogos hingga sekarang.",
        lock_line,
        "Nota: scope creep perlu diasingkan.",
        "Kesimpulan: tiga perkara boleh dibuat sekarang.",
    ])
    if len(body) < total_chars:
        pad, i = [], 0
        while len(body) + sum(len(p) + 1 for p in pad) < total_chars:
            pad.append(f"Baris penambah nombor {i} dengan kandungan unik-{i}.")
            i += 1
        body += "\n" + "\n".join(pad)
    return body[:total_chars]


class TestAttractorLockStopPath:
    """Wiring regression for the attractor rule on the finish_reason="stop" path.

    Both cases below were DELIVERED to humans before this guard existed: the short one because
    it sat under STOP_PATH_MIN_CHARS and was never examined, the long one because it cleared the
    length gate and was then waved through by the distinct-line shape check.
    """

    def test_short_lock_under_the_length_gate_is_blocked(self, loop_agent):
        echo = _locked(409, 14)
        assert len(echo) < STOP_PATH_MIN_CHARS, "fixture must sit under the old length gate"
        loop_agent.client.chat.completions.create.side_effect = [
            _response(echo, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "kenapa seat depan paling murah")

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert result["completed"] is False
        assert result["partial"] is True
        assert (result["failure_reason"], result["failure_retryable"]) == ("truncated", True)
        assert echo not in (result["final_response"] or "")
        assert "Repetition" in (result["final_response"] or "")

    def test_long_lock_that_defeats_the_shape_check_is_blocked(self, loop_agent):
        echo = _locked(132_625, 12_479, token="sayang")
        assert len(echo) >= STOP_PATH_MIN_CHARS
        loop_agent.client.chat.completions.create.side_effect = [
            _response(echo, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "cross-audit semula")

        assert result["completed"] is False
        assert result["partial"] is True
        assert echo not in (result["final_response"] or "")
        assert "Repetition" in (result["final_response"] or "")

    def test_locked_text_never_becomes_durable_history(self, loop_agent):
        """The looped bytes must not be persisted — replaying them re-seeds the next turn."""
        echo = _locked(3_993, 22)
        loop_agent.client.chat.completions.create.side_effect = [
            _response(echo, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "kenapa seat depan paling murah")

        assert not any(
            isinstance(m, dict) and m.get("content") == echo for m in result["messages"]
        )

    def test_prose_with_normal_word_frequency_still_delivers(self, loop_agent):
        """Precision control: ordinary BM prose must not be blocked by the new rule."""
        text = "\n".join(
            f"Ayat nombor {i} menerangkan perkara berbeza supaya tiada perkataan "
            f"berulang secara berturut-turut, walaupun perkataan punya dan aku muncul "
            f"sekali-sekala macam orang biasa cakap."
            for i in range(40)
        )
        loop_agent.client.chat.completions.create.side_effect = [
            _response(text, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "cerita pasal orchestra")

        assert result["completed"] is True
        assert result["final_response"] == text.strip()
