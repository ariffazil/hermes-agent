"""Curator refusals are ledgered — audit-only rows, never rollback targets.

The autonomous background-review fork may only write curator-managed skills. Every
ownership/protection refusal now appends one ``refused`` row to the skill ledger
(~/.hermes/skills/.curator_ledger.jsonl), so a lesson whose only natural home is a
protected skill is still visible after the review ends — instead of vanishing with
the chat line. The in-process counter is per-review-session only and breaks a single
session's retry loop; cross-session repetition is read from the ledger rows.
"""

import json

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME/skills dir + a guard that believes it is the review fork."""
    from agent import skill_utils
    from tools import skill_ledger, skill_manager_guards, skill_manager_tool, skill_usage

    home = tmp_path / "home"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)

    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_usage, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr(skill_manager_guards, "_is_background_review", lambda: True)
    monkeypatch.setattr(skill_usage, "load_usage", lambda: {})
    for pred in ("is_protected_builtin", "is_hub_installed", "is_bundled"):
        monkeypatch.setattr(skill_usage, pred, lambda name: False)

    def _rows():
        path = skill_ledger.ledger_path()
        if not path.exists():
            return []
        return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    return {"guards": skill_manager_guards, "ledger": skill_ledger,
            "skills_dir": skills_dir, "rows": _rows}


def _refuse(curator_env, name="user-owned-skill", action="patch"):
    return curator_env["guards"]._background_review_write_guard(
        name, curator_env["skills_dir"] / name / "SKILL.md", action)


def test_refusal_is_written_to_the_ledger(curator_env):
    result = _refuse(curator_env)

    assert result and result["success"] is False
    assert "not curator-managed" in result["error"]
    refused = [r for r in curator_env["rows"]() if r["action"] == "refused"]
    assert len(refused) == 1
    row = refused[0]
    assert row["actor"] == "curator"
    assert row["skill"] == "user-owned-skill"
    assert row["before"] == [] and row["after"] == []
    assert "not curator-managed" in row["evidence"]["reason"]


def test_bundled_skill_refusal_is_ledgered(curator_env, monkeypatch):
    from tools import skill_usage
    monkeypatch.setattr(skill_usage, "is_bundled", lambda name: name == "hermes-agent")

    result = _refuse(curator_env, name="hermes-agent")

    assert result and "bundled" in result["error"]
    assert [r["skill"] for r in curator_env["rows"]() if r["action"] == "refused"] == ["hermes-agent"]


def test_refused_row_is_not_a_rollback_target(curator_env):
    _refuse(curator_env)
    row_id = next(r["id"] for r in curator_env["rows"]() if r["action"] == "refused")

    ok, message = curator_env["ledger"].rollback_entry(row_id)

    assert ok is False and "audit row" in message


def test_in_session_counter_hard_stops_after_the_limit(curator_env):
    guards = curator_env["guards"]
    token = guards._background_review_rejection_counts_var.set({})
    try:
        first = _refuse(curator_env)
        assert first and not first["error"].startswith("STOP:")
        counts = guards._background_review_rejection_counts_var.get()
        assert counts == {"user-owned-skill": 1}

        guards._background_review_rejection_counts_var.set({"user-owned-skill": 3})
        stopped = _refuse(curator_env)
        assert stopped and stopped["error"].startswith("STOP:")
    finally:
        guards._background_review_rejection_counts_var.reset(token)


def test_curator_managed_skill_still_writes(curator_env, monkeypatch):
    from tools import skill_usage
    monkeypatch.setattr(skill_usage, "load_usage",
                        lambda: {"my-umbrella": {"created_by": "agent", "pinned": False}})

    assert _refuse(curator_env, name="my-umbrella") is None
    assert [r for r in curator_env["rows"]() if r["action"] == "refused"] == []


def test_ledger_failure_never_blocks_the_refusal(curator_env, monkeypatch):
    def _boom(*a, **k):
        raise OSError("ledger unwritable")

    monkeypatch.setattr(curator_env["ledger"], "append_entry", _boom)

    result = _refuse(curator_env)

    assert result and result["success"] is False
