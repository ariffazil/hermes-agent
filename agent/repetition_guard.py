"""Cheap content-sanity checks for completed model output.

A model in a degenerate repetition loop can spend its ENTIRE output budget echoing one fragment;
the ``finish_reason=length`` continuation would then stitch it into the final response with a
"continue" nudge (one incident: a 60k-char turn delivered as 31 Discord messages). This detects
repetition-dominated fragments BEFORE the nudge so the turn aborts with a clear error. Deliberately
conservative: only LONG verbatim repeats (60+ chars) covering a majority of the fragment trip it.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Below this length the check doesn't run: short truncations trivially
# contain repeated tokens and are legitimately continued.
MIN_FRAGMENT_LENGTH = 400
# Exact-repeat window; far beyond ordinary phrasing reuse (citations, headings, similar code).
_REPEAT_WINDOW = 60
# A window repeating at least this often is a signal even for short fragments.
_MIN_REPEAT_COUNT = 5
# "Repetition-dominated" = repeated windows cover at least this fraction.
_DOMINANCE_RATIO = 0.5

# What an interrupt checkpoint says INSTEAD of a repetition-dominated partial. Replaying the
# looped bytes (as the redirect's api_content or as the interrupted assistant row) re-seeds the
# loop on the next request and the corruption survives restarts (#112764); the model only needs
# to know the reply degenerated and was cut off.
REPETITION_LOOP_INTERRUPTED = "[the reply degenerated into a repetition loop and was interrupted]"

# Sampling bounds keep the general path linear in output size with a small,
# fixed multiplier. A dominant contiguous run necessarily crosses many of
# these evenly spaced anchors.
_MAX_ANCHOR_SAMPLES = 32
_MAX_ANCHOR_MATCHES = 8


# ``is_runaway_repetition``: a multi-line partial must be mostly copies of a few lines. Batch-style
# output (distinct INSERT rows, similar table rows) shares long prefixes and trips the window
# scan, but every line is distinct; a loop re-emits the same line(s).
_RUNAWAY_DISTINCT_LINE_RATIO = 0.5

# The finish_reason="stop" path discards a COMPLETED answer, so it only aborts at runaway scale:
# real stop-path loops (#100716) run 80k-350k chars, while asked-for repeats ("say X 50 times",
# identical table rows, templated YAML) stay in the low KB and must be delivered.
STOP_PATH_MIN_CHARS = 16_000


def is_repetition_dominated(text: str) -> bool:
    """True when a contiguous run of at least five exact repetitions covers at least half
    of ``text`` — the signature of a model repetition loop (issue #86581). Coverage is measured
    with the true period, so long multi-line loop units count fully. Fail-open for short input.
    """
    if not isinstance(text, str):
        return False
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return False

    # Fast path: one normalized line duplicated enough to cover half the fragment (the common echo shape).
    if _line_repetition_dominated(text, n):
        return True

    # Window-count scan (#86581) catches loops whose repeats differ by a counter or noise token;
    # the periodic scan catches long exact units whose 60-char windows each recur too rarely.
    return _window_count_dominated(text, n) or _periodic_run_dominated(text, n)


def _window_count_dominated(text: str, n: int) -> bool:
    """True when one 60-char window recurs often enough to cover half of ``text``."""
    window = _REPEAT_WINDOW
    needed = max(_MIN_REPEAT_COUNT, math.ceil(n * _DOMINANCE_RATIO / window))
    counts: dict[str, int] = {}
    for i in range(n - window + 1):
        key = text[i : i + window]
        c = counts.get(key, 0) + 1
        if c >= needed:
            return True
        counts[key] = c
    return False


def _periodic_run_dominated(text: str, n: int) -> bool:
    """Detect a dominant exact periodic run from evenly spaced anchors.

    Matching a 60-character anchor at a later position supplies a candidate
    period. Expanding the equality ``text[i] == text[i + period]`` in both
    directions recovers the full run, so coverage is measured using the true
    repeating unit rather than crediting every occurrence with only 60 chars.
    """
    window = _REPEAT_WINDOW
    max_start = n - window
    if max_start < 1:
        return False

    sample_step = max(
        1,
        (max_start + _MAX_ANCHOR_SAMPLES - 2) // (_MAX_ANCHOR_SAMPLES - 1),
    )
    sample_starts = list(range(0, max_start + 1, sample_step))
    if sample_starts[-1] != max_start:
        sample_starts.append(max_start)

    # Runs already expanded and rejected, as (left, right, period). A later anchor inside one of
    # them whose period is a multiple of that run's period would re-walk the same run.
    rejected: list[tuple[int, int, int]] = []
    for start in sample_starts:
        anchor = text[start : start + window]
        search_from = start + 1
        for _ in range(_MAX_ANCHOR_MATCHES):
            match = text.find(anchor, search_from)
            if match < 0:
                break
            period = match - start
            search_from = match + 1
            if any(lo <= start < hi and period % p == 0 for lo, hi, p in rejected):
                continue
            left, right = _expand_run(text, n, start, period, window)
            if right - left >= _MIN_REPEAT_COUNT * period and right - left >= n * _DOMINANCE_RATIO:
                return True
            rejected.append((left, right, period))
    return False


def _expand_run(text: str, n: int, start: int, period: int, matched: int) -> tuple[int, int]:
    """Expand one known equal window to the ``[left, right)`` bounds of its exact periodic run."""
    left = start
    while left > 0 and text[left - 1] == text[left - 1 + period]:
        left -= 1

    right = start + matched
    while right + period < n and text[right] == text[right + period]:
        right += 1
    return left, right + period


def is_runaway_repetition(text: str) -> bool:
    """Stricter than :func:`is_repetition_dominated`: also require the runaway shape.

    An interrupt checkpoint DROPS the partial when this fires, so a legitimately repetitive but
    correct reply (distinct batch rows) must not qualify: repeated windows have to dominate AND,
    when the text has line structure, at most half of its non-empty lines may be distinct.
    """
    if not is_repetition_dominated(text):
        return False
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < _MIN_REPEAT_COUNT:
        return True  # no line structure to judge by: a dominated single-line loop
    return len(set(lines)) <= len(lines) * _RUNAWAY_DISTINCT_LINE_RATIO


def _line_repetition_dominated(text: str, n: int) -> bool:
    """True when a single normalized line covers half the fragment via repeats."""
    counts = Counter(norm for norm in (line.strip() for line in text.splitlines()) if norm)
    return any(c >= _MIN_REPEAT_COUNT and c * len(line) >= n * _DOMINANCE_RATIO for line, c in counts.items())


# ---------------------------------------------------------------------------
# Attractor lock — one WORD repeating, not one window or one line.
#
# Both detectors above look for structure: a 60-char verbatim window, or a duplicated LINE.
# A single-token attractor defeats them both. The degenerate text is one enormous line, so
# ``is_runaway_repetition``'s distinct-line ratio reads it as *non*-repetitive and returns
# False even when ``is_repetition_dominated`` already said True.
#
# Measured 2026-10-02 (FI-003) against 9 confirmed incidents and 22,246 real assistant rows:
# the stop-path guard as shipped blocked **0 of 9**. Eight incidents were shorter than
# STOP_PATH_MIN_CHARS so they were never examined; the ninth (132,625 chars, one token
# repeated 12,479 times) cleared the length gate and was then waved through by the
# distinct-line shape check. The same run-length rule below blocked 7 of 9 with **0 false
# positives** on all 22,246 rows.
#
# The attractor token is whatever the persona register makes most salient — observed locking
# onto four different tokens across incidents ('sayang', 'ekos', 'arif', 'punya') — so no word
# list is used or would survive.
#
# Deliberately length-independent: run length, not total length, is what separates a loop from
# prose. The 22,246-row calibration scored the GLOBAL longest run and found 0 false positives;
# token_run below scores per line, which can only lower a score, so that 0 holds a fortiori
# while also sparing repeated markdown/checklist rows. Recall is unchanged: every incident kept
# a per-line run >= 14 (measured 12479 / 2326 / 1827 / 1322 / 218 / 22 / 14).
ATTRACTOR_MIN_RUN = 12
_ATTRACTOR_TOKEN_RE = re.compile(r"[0-9A-Za-z']+")


def token_run(text: str) -> tuple[int, str]:
    """Longest run of one identical word repeated consecutively *within a line*, and that word.

    Single linear pass over the text. Punctuation and whitespace between repeats do not break
    the run, so ``"punya. Punya punya,"`` counts as 3. A newline DOES break it.

    The newline rule is measured, not stylistic. Across the 9 production incidents the lock was
    intra-line in every case (8 had global run == per-line run exactly; the ninth lost 3 of
    1,325 to a line break). Meanwhile the shape it must not be confused with — a markdown table
    or checklist whose rows repeat — is newline-separated by construction, so scoring per line
    drops that from a run of 240 to a run of 3. Cross-line repetition is already owned by
    :func:`_line_repetition_dominated`; this function owns the intra-line word lock, and the
    two do not overlap.

    Returns ``(0, "")`` for empty or non-string input.
    """
    if not isinstance(text, str) or not text:
        return 0, ""
    best = 0
    best_tok = ""
    for line in text.split("\n"):
        cur = 0
        prev = None
        for tok in _ATTRACTOR_TOKEN_RE.findall(line):
            low = tok.lower()
            cur = cur + 1 if low == prev else 1
            prev = low
            if cur > best:
                best, best_tok = cur, low
    return best, best_tok


def is_attractor_locked(text: str, min_run: int = ATTRACTOR_MIN_RUN) -> bool:
    """True when one word repeats consecutively ``min_run`` or more times within a line.

    Unlike :func:`is_repetition_dominated` this does not require a minimum text length, so a
    409-char reply that locked is caught as readily as a 132 KB one.

    Known accepted false positive: a user who literally asks for the same word a dozen or more
    times on one line ("say OK 50 times") is blocked. That request is rare on these lanes and
    re-askable; the alternative measured in the wild is 132 KB of one word delivered to a human.
    The trade is deliberate — do not "fix" it by reintroducing a length floor, which is exactly
    what let 8 of the 9 incidents through.
    """
    return token_run(text)[0] >= min_run

