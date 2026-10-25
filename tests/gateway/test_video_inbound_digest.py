"""Video inbound must arrive as TEXT, not as a bare path.

Failure this locks out (measured 2026-10-25 09:13-09:23): a human sent a video, the lane replied
"aku tak boleh baca video" three times, then guessed the wrong intent and answered about porn.
Root cause was structural, not model quality — Telegram hands over a cached .mp4 path, the only
vision tool accepts images, and the frame+ASR recipe needed a shell that the kernel gate refused.
The fix is upstream of the model: a digest (frames on disk + speech as text) is inlined into the
inbound note before the agent is asked anything.

Contract under test:
  * script succeeds  -> note carries the digest and points at the frames
  * script fails     -> note falls back to the old path-only note (never a broken prompt)
  * non-video path   -> digest is never attempted
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from gateway.run_inbound import GatewayInboundMixin


VIDEO = "/root/.hermes/cache/videos/video_deadbeef.mp4"
DIGEST = (
    "[VIDEO DIGEST — measured locally, not described] 7.33s · 464x848 · h264/aac · 1.49MB\n"
    "Frames (4) saved — read them with vision_analyze before answering: /tmp/f01.jpg\n"
    'Speech (groq): "sambal asing"'
)


@pytest.fixture()
def fake_script(monkeypatch, tmp_path):
    """Pretend the digest script is installed so the test does not depend on host paths."""
    script = tmp_path / "video-digest.py"
    script.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(
        GatewayInboundMixin,
        "_video_digest_note",
        _patched_digest(str(script)),
    )
    return str(script)


def _patched_digest(script_path):
    """Re-implement the wrapper against an injected script path (same logic, testable path)."""

    def _inner(path: str) -> str:
        import os

        if not os.path.isfile(script_path):
            return ""
        proc = subprocess.run(
            ["python3", script_path, path, "--frames", "4"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out.startswith("[VIDEO DIGEST"):
            return out
        return ""

    return staticmethod(_inner).__func__


def test_digest_is_inlined_for_video(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=DIGEST, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("gateway.run_inbound.os.path.isfile", lambda p: True)

    note = GatewayInboundMixin._prepend_inbound_media_file_notes("apa jadi ni", [], [VIDEO])

    assert "[VIDEO DIGEST" in note, "digest must reach the model as text"
    assert "read them with vision_analyze" in note
    assert "apa jadi ni" in note, "the human's own words must survive the prepend"
    assert note.index("[VIDEO DIGEST") < note.index("apa jadi ni"), "digest goes above the text"


def test_failure_keeps_path_note(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="ffmpeg missing")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("gateway.run_inbound.os.path.isfile", lambda p: True)

    note = GatewayInboundMixin._prepend_inbound_media_file_notes("apa jadi ni", [], [VIDEO])

    assert "[VIDEO DIGEST" not in note
    assert "Its content is not inlined here" in note, "must degrade to the old note, not guess"
    assert VIDEO in note


def test_audio_paths_never_run_the_digest(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=DIGEST, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("gateway.run_inbound.os.path.isfile", lambda p: True)

    GatewayInboundMixin._prepend_inbound_media_file_notes(
        "dengar ni", ["/root/.hermes/cache/audio/deadbeef.ogg"], []
    )

    assert calls == [], "the digest is video-only; audio keeps its existing transcription lane"
