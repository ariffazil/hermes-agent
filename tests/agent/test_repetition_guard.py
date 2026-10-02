"""Unit tests for repetition-dominated model output detection."""

from __future__ import annotations

import random
import time

import pytest

from agent.repetition_guard import (
    ATTRACTOR_MIN_RUN,
    MIN_FRAGMENT_LENGTH,
    STOP_PATH_MIN_CHARS,
    is_attractor_locked,
    is_repetition_dominated,
    is_runaway_repetition,
    token_run,
)

# The exact sentence from the #86581 incident (echoed hundreds of times by
# the model before the provider cut it off at finish_reason=length).
_INCIDENT_ECHO = "好，你幫我更改成 Google Gemini 4 31B。"


class TestRepetitionGuard:
    def test_incident_shape_flags_repetition(self):
        # Narration + the echoed sentence on its own line, repeated (line path).
        text = ("We need to verify the model setting.\n" + _INCIDENT_ECHO + "\n") * 800
        assert is_repetition_dominated(text) is True

    def test_repeated_sentence_without_line_breaks_flags(self):
        # Repetition loop with no line breaks — exercises the window path.
        text = _INCIDENT_ECHO * 2000
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is True

    def test_multiline_paragraph_run_uses_true_period_coverage(self):
        rng = random.Random(11)
        paragraph = "\n".join(
            "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(151))
            for _ in range(5)
        ) + "\n"

        # The incident unit was approximately 764 characters. Detection must
        # remain scale-free as the number of exact repeats grows.
        for repeat_count in (100, 1_000, 10_000):
            assert is_repetition_dominated(paragraph * repeat_count) is True

    @pytest.mark.parametrize("shape", ["unique_prefix_suffix", "counter_loop"])
    def test_dominant_run_with_unique_prefix_and_suffix_flags(self, shape):
        if shape == "counter_loop":
            # A changing counter breaks exact periodicity; main's window scan (#86581) must
            # still flag it.
            text = "".join(
                f"Step {i}: I will now carefully re-check the configuration file for the error again.\n"
                for i in range(200)
            )
            assert is_repetition_dominated(text) is True
            return
        paragraph = (
            "A deliberately long repeated paragraph has enough distinct text "
            "to make its period exceed the guard's minimum anchor length.\n"
            "It also spans multiple lines, matching the real incident shape.\n"
        )
        repeated = paragraph * 20
        text = ("unique introduction " * 20) + repeated + (" unique ending" * 20)

        assert len(repeated) > len(text) * 0.5
        assert is_repetition_dominated(text) is True

    def test_long_legitimate_text_not_flagged(self):
        # Long, unique prose — no 60-char window ever repeats.
        text = " ".join(
            f"Sentence number {i} describes a distinct topic with unique words "
            f"such as quasar-{i} and nebula-{i} to keep every window distinct."
            for i in range(1200)
        )
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is False

    def test_short_fragment_never_flagged(self):
        # Below MIN_FRAGMENT_LENGTH the guard fails open — short truncations
        # are legitimately continued even if they look repetitive.
        assert is_repetition_dominated("A. " * 50) is False
        assert is_repetition_dominated("hello ") is False

    def test_repeat_not_dominant_not_flagged(self):
        # A repeated sentence scattered through a long unique text: repeated
        # windows exist but cover far less than half of the fragment.
        filler = " ".join(f"unique filler token {i}" for i in range(3000))
        text = filler + ("\n" + _INCIDENT_ECHO + "\n") * 30
        assert is_repetition_dominated(text) is False

    def test_non_string_inputs(self):
        assert is_repetition_dominated("") is False
        assert is_repetition_dominated(None) is False
        assert is_repetition_dominated(12345) is False


# ---------------------------------------------------------------------------
# Attractor lock — one WORD repeating. Calibrated 2026-10-02 (FI-003) against 9 confirmed
# production incidents and 22,246 real assistant rows: the shipped stop-path guard blocked
# 0 of 9; this rule blocks 7 of 9 with 0 false positives. Fixtures below are SYNTHETIC and
# reproduce the measured (char count, run length) shape of each incident — the real content is
# private human chat and must not be committed to a repo with a public upstream.
_INCIDENT_SHAPES = [
    # (label, total_chars, consecutive_run, attractor_token)
    ("dm-one-word-132k", 132_625, 12_479, "sayang"),
    ("group-lock-14k", 14_017, 2_326, "ekos"),
    ("group-lock-11k", 11_487, 1_827, "ekos"),
    ("name-lock-7k", 7_188, 1_325, "arif"),
    ("enclitic-9k", 9_570, 218, "punya"),
    ("enclitic-4k", 3_993, 22, "punya"),
    ("enclitic-409", 409, 14, "punya"),
]


def _synthesize(total_chars: int, run: int, token: str) -> str:
    """Reproduce the measured incident shape, including its LINE structure.

    Several distinct prose lines plus exactly one line carrying the whole lock. Line structure
    is load-bearing, not decoration: ``is_runaway_repetition`` only misses the real incidents
    because they have >=5 non-empty lines that are >50% distinct (measured on the 132 KB case:
    43 non-empty lines, 42 distinct = 98%), with the entire lock inside a single 87,420-char
    line. A fixture that put everything on one line would be caught by the old guard and would
    therefore not reproduce the bug being guarded against.
    """
    lock_line = (
        "Aku nampak apa yang hang rasa. "
        + " ".join([token] * run)
        + " itu sahaja yang aku boleh katakan."
    )
    body = "\n".join([
        "OK, dah cukup probe. Ini ringkasan sebenar, bukan inventori pusing.",
        "---",
        "**Apa data nampak, dan apa dia tak nampak.**",
        "Panjang chat: 26 Ogos hingga sekarang, tiga lapisan berbeza.",
        lock_line,
        "Nota: scope creep perlu diasingkan daripada isu asal.",
        "Kesimpulan: tiga perkara boleh dibuat sekarang, satu perlu keputusan hang.",
    ])
    if len(body) < total_chars:
        pad, i = [], 0
        while len(body) + sum(len(p) + 1 for p in pad) < total_chars:
            pad.append(f"Baris penambah nombor {i} dengan kandungan unik-{i} supaya tiada ulang.")
            i += 1
        body += "\n" + "\n".join(pad)
    return body[:total_chars]


class TestAttractorLock:
    @pytest.mark.parametrize("label,total,run,token", _INCIDENT_SHAPES)
    def test_every_measured_incident_scale_is_caught(self, label, total, run, token):
        text = _synthesize(total, run, token)
        assert token_run(text)[0] >= run - 1, label
        assert is_attractor_locked(text) is True, label

    def test_catches_incidents_the_length_gate_never_examined(self):
        """8 of the 9 real incidents were under STOP_PATH_MIN_CHARS and so were never checked.

        The 7 fixtures here span 6 sub-threshold scales plus the one that cleared the gate.
        """
        short = [s for s in _INCIDENT_SHAPES if s[1] < STOP_PATH_MIN_CHARS]
        assert len(short) == 6, "fixture set drifted from the measured incident population"
        for label, total, run, token in short:
            assert is_attractor_locked(_synthesize(total, run, token)) is True, label

    def test_132k_case_defeats_runaway_shape_check_but_not_attractor(self):
        """Regression: the ONE incident above the length gate was waved through by shape.

        A one-token lock is a single enormous line, so the distinct-line ratio sees "all lines
        distinct" and returns False even though is_repetition_dominated already said True.
        """
        text = _synthesize(132_625, 12_479, "sayang")
        assert len(text) >= STOP_PATH_MIN_CHARS
        assert is_repetition_dominated(text) is True
        assert is_runaway_repetition(text) is False, "if this now passes, re-check the composite"
        assert is_attractor_locked(text) is True

    def test_threshold_boundary(self):
        assert is_attractor_locked(" ".join(["x"] * (ATTRACTOR_MIN_RUN - 1))) is False
        assert is_attractor_locked(" ".join(["x"] * ATTRACTOR_MIN_RUN)) is True

    def test_punctuation_and_case_do_not_break_the_run(self):
        # Real incident text alternates case and punctuation between repeats.
        assert is_attractor_locked("Punya. punya punya, PUNYA punya — punya; punya punya punya punya punya punya") is True

    def test_newline_breaks_the_run_by_design(self):
        """The attractor rule owns INTRA-line word locks; cross-line repeats belong elsewhere.

        Measured across the 9 incidents: 8 had global run == per-line run exactly, the ninth
        lost 3 of 1,325 to a line break — so the newline rule costs no recall. What it buys is
        sparing repeated markdown/checklist rows, which are newline-separated by construction.
        Cross-line repetition is already owned by _line_repetition_dominated on the length path.
        """
        assert is_attractor_locked("\n".join(["sama sama sama"] * 80)) is False
        assert is_attractor_locked(" ".join(["sama"] * 80)) is True

    def test_token_run_reports_the_attractor_for_diagnostics(self):
        run, tok = token_run("prose here " + "sayang " * 20 + " more prose")
        assert tok == "sayang"
        assert run >= 20

    @pytest.mark.parametrize(
        "label,text",
        [
            ("markdown_table", "\n".join(f"| row {i} | value | ok |" for i in range(80))),
            ("identical_table_rows", "\n".join(["| sama | sama | sama |"] * 80)),
            ("rule_lines", ("=" * 70 + "\n") * 40),
            ("code_block", "\n".join(f"    if x == {i % 3}:\n        pass" for i in range(60))),
            ("bullet_list", "\n".join(f"- item {i} bermaksud sesuatu yang berbeza" for i in range(60))),
            ("yaml", "\n".join(f"  key{i}: value" for i in range(100))),
            ("long_unique_prose", " ".join(
                f"Ayat {i} menerangkan topik berbeza dengan perkataan unik-{i} supaya tiada "
                f"perkataan berulang secara berturut-turut dalam teks panjang ini."
                for i in range(400))),
        ],
    )
    def test_legitimate_structured_output_is_not_blocked(self, label, text):
        assert is_attractor_locked(text) is False, label

    def test_accepted_false_positive_is_intentional(self):
        """A literal 'repeat this word' request IS blocked. Deliberate trade — see docstring.

        Asserted so a future lane reads this as a decision, not as a bug to 'fix' by
        reintroducing a length floor (which is what let 8 of 9 incidents through).
        """
        assert is_attractor_locked("OK " * 50) is True

    def test_known_residual_gap_is_documented_not_hidden(self):
        """2 of 9 incidents locked only loosely (run 4 and 5) and are NOT caught.

        Their token-frequency skew overlaps ordinary Malay prose: at the thresholds that would
        catch them, ~1-2% of 22,246 clean rows would be blocked. Suppressing legitimate output
        is worse than the mild garble, so the gap stays open and visible.
        """
        assert is_attractor_locked(_synthesize(7_310, 4, "punya")) is False
        assert is_attractor_locked(_synthesize(632, 5, "punya")) is False

    def test_non_string_and_empty_inputs(self):
        assert token_run("") == (0, "")
        assert token_run(None) == (0, "")
        assert token_run(12345) == (0, "")
        assert is_attractor_locked("") is False
        assert is_attractor_locked(None) is False

    def test_linear_on_the_largest_measured_incident(self):
        text = _synthesize(132_625, 12_479, "sayang")
        start = time.perf_counter()
        assert is_attractor_locked(text) is True
        assert time.perf_counter() - start < 1.0, "detector must stay cheap on the hot path"
