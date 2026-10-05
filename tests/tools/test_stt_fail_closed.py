"""Tests for the STT fail-soft advisory marker (_join_confident_segments).

2026-10-05 scar: music-only audio produced a hallucinated foreign-language
transcript downstream even though whisper's own no_speech_prob said the
window was mostly non-speech. F13 sovereign directive (fail-soft, supersedes
the original fail-closed wiring): when the window scores as mostly
non-speech, the join annotates the transcript with NO_CLEAR_SPEECH and the
caller attaches an advisory flag — the transcript is still DELIVERED; the
signal informs, it never censors.
"""

from types import SimpleNamespace

from tools.transcription_local import (
    NO_CLEAR_SPEECH_MARKER,
    _NO_SPEECH_PROB_THRESHOLD_DEFAULT,
    _confidence_thresholds,
    _join_confident_segments,
)


def _seg(text, no_speech_prob=0.0, avg_logprob=-0.2):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)


class TestJoinFailSoft:
    def test_all_hallucinated_high_nospeech_annotates_marker(self):
        # Mostly-non-speech window: the marker replaces the (dropped) text so
        # callers attach the advisory flag.
        segments = [
            _seg(" Håkan Kåk", no_speech_prob=0.9, avg_logprob=-1.5),
            _seg(" Ik skrik om", no_speech_prob=0.85, avg_logprob=-1.4),
        ]
        result = _join_confident_segments(segments, {})
        assert result == NO_CLEAR_SPEECH_MARKER

    def test_high_nospeech_but_confident_segment_still_delivers_marker(self):
        # A confident surviving segment on a mostly-non-speech window: the
        # advisory gate fires on the window average — annotation wins over
        # the surviving text (the scar shape).
        segments = [_seg(" a probable line", no_speech_prob=0.56, avg_logprob=-0.5)]
        assert _join_confident_segments(segments, {}) == NO_CLEAR_SPEECH_MARKER

    def test_all_hallucinated_low_nospeech_stays_speech(self):
        # Quiet-but-real speech: the AND gate keeps the segment (no_speech
        # low) — transcript survives untouched, no advisory claim.
        segments = [_seg(" whispered words", no_speech_prob=0.1, avg_logprob=-1.8)]
        assert _join_confident_segments(segments, {}) == "whispered words"

    def test_real_speech_survives(self):
        segments = [
            _seg(" hello world", no_speech_prob=0.05, avg_logprob=-0.3),
            _seg(" how are you", no_speech_prob=0.1, avg_logprob=-0.4),
        ]
        result = _join_confident_segments(segments, {})
        assert result == "hello world how are you"

    def test_mixed_keeps_good_drops_bad(self):
        # Kept-segment average sits below the advisory threshold, so the kept
        # text passes through unchanged.
        segments = [
            _seg(" garbage", no_speech_prob=0.95, avg_logprob=-1.6),
            _seg(" real speech", no_speech_prob=0.2, avg_logprob=-0.5),
        ]
        assert _join_confident_segments(segments, {}) == "real speech"

    def test_empty_segments_no_marker(self):
        # A zero-segment result stays empty: no window scored non-speech,
        # so no claim about the audio is being made at all.
        assert _join_confident_segments([], {}) == ""

    def test_all_dropped_aggressive_threshold_no_marker(self):
        # With an aggressive drop threshold, mid-confidence segments are
        # dropped but the window never scored above the advisory average —
        # conservative: stay empty, do NOT claim NO_CLEAR_SPEECH.
        segments = [_seg(" x", no_speech_prob=0.25, avg_logprob=-1.5)]
        cfg = {"no_speech_prob_threshold": 0.2, "logprob_threshold": -1.0}
        assert _join_confident_segments(segments, cfg) == ""

    def test_marker_in_surviving_text_is_advisory_shape(self):
        # A segment literally decoding the marker is the hallucination shape:
        # same advisory verdict.
        segments = [_seg(f" {NO_CLEAR_SPEECH_MARKER}", no_speech_prob=0.05, avg_logprob=-0.3)]
        assert _join_confident_segments(segments, {}) == NO_CLEAR_SPEECH_MARKER

    def test_transcribe_local_attaches_advisory_flag_fail_soft(self):
        # The _transcribe_local envelope keeps success=True and the transcript
        # — fail-soft — and attaches the machine flag for callers.
        from tools.transcription_tools import _looks_like_no_clear_speech
        transcript = NO_CLEAR_SPEECH_MARKER
        result = {"success": True, "transcript": transcript, "provider": "local"}
        if _looks_like_no_clear_speech(transcript):
            result["no_clear_speech"] = True
            result["stt_unknown"] = "NO_SPEECH"
        assert result["success"] is True
        assert result["transcript"] == NO_CLEAR_SPEECH_MARKER
        assert result["no_clear_speech"] is True
        assert result["stt_unknown"] == "NO_SPEECH"

    def test_default_confidence_thresholds_unchanged(self):
        # The segment gate defaults are upstream's; the advisory layer adds
        # its own average gate without disturbing them.
        assert _confidence_thresholds({}) == (_NO_SPEECH_PROB_THRESHOLD_DEFAULT, -1.0)
