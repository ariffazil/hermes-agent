"""Delivery-evidence contract for the delivery ledger (SCAR-2026-09-28-008 follow-through).

The ledger used to store the produced text and flip state='delivered' with no evidence of what
left the process. On 2026-09-28 the outbound shape boundary destroyed 84.9% of a day's
characters (713,924 produced → 107,868 sent, 47 replies shipped empty) between those two
points, and every "did it reach the human?" query answered `delivered`.

These tests pin what replaced that: produced_len / sent_len / delivered_receipt_id, written in
the SAME transaction as the state flip, with NULL meaning unknown and never zero.

    venv/bin/python -m pytest tests/gateway/test_delivery_ledger_evidence.py -q
"""

import sqlite3

import pytest

from gateway import delivery_ledger as dl

OLD_DDL = """CREATE TABLE delivery_obligations (
    obligation_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    platform TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    thread_id TEXT,
    content TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    last_error TEXT,
    adapter_profile TEXT
)"""

EVIDENCE_COLS = ("produced_len", "sent_len", "delivered_receipt_id")


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _cols(conn):
    return {r[1] for r in conn.execute("PRAGMA table_info(delivery_obligations)")}


def _insert(conn, oid, state="attempting", content="hello"):
    conn.execute(
        "INSERT INTO delivery_obligations (obligation_id, session_key, platform, chat_id,"
        " content, state, attempts, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (oid, "sk", "telegram", "42", content, state, 1, 1.0, 1.0))


def test_fresh_schema_carries_the_evidence_columns():
    conn = dl._connect()
    try:
        missing = [c for c in EVIDENCE_COLS if c not in _cols(conn)]
        assert not missing, f"new ledger is missing evidence columns: {missing}"
    finally:
        conn.close()


def test_existing_v1_table_is_upgraded_without_losing_rows():
    path = dl._db_path()
    conn = sqlite3.connect(path)
    conn.execute(OLD_DDL)
    _insert(conn, "legacy-1", state="delivered", content="produced text")
    conn.commit()
    assert "sent_len" not in _cols(conn), "fixture must start on the old shape"
    conn.close()

    conn = dl._connect()  # opening the ledger runs _initialize_schema
    try:
        assert set(EVIDENCE_COLS) <= _cols(conn)
        row = conn.execute("SELECT state, produced_len, sent_len, delivered_receipt_id FROM "
                           "delivery_obligations WHERE obligation_id='legacy-1'").fetchone()
        assert row[0] == "delivered", "migration must not rewrite history"
        assert row[1:] == (None, None, None), (
            f"a pre-evidence row must read UNKNOWN, not zero: {row}"
        )
    finally:
        conn.close()


def test_mark_delivered_writes_state_and_evidence_together():
    conn = dl._connect()
    _insert(conn, "ob-1")
    conn.commit()
    conn.close()

    dl.mark_delivered("ob-1", produced_len=8780, sent_len=167, delivered_receipt_id="5512")

    conn = dl._connect()
    row = conn.execute("SELECT state, produced_len, sent_len, delivered_receipt_id,"
                       " attempts FROM delivery_obligations WHERE obligation_id='ob-1'").fetchone()
    conn.close()
    assert row == ("delivered", 8780, 167, "5512", 1)
    assert row[2] < row[1], "the trimmed case must be visible as sent < produced"


def test_success_without_a_receipt_stays_a_visible_gap():
    conn = dl._connect()
    _insert(conn, "ob-2")
    conn.commit()
    conn.close()

    dl.mark_delivered("ob-2")

    conn = dl._connect()
    row = conn.execute("SELECT state, produced_len, sent_len, delivered_receipt_id "
                       "FROM delivery_obligations WHERE obligation_id='ob-2'").fetchone()
    conn.close()
    assert row[0] == "delivered"
    assert row[1:] == (None, None, None), (
        f"a delivery with no evidence must record UNKNOWN, never a fabricated number: {row}"
    )


def test_mark_failed_still_records_its_error():
    conn = dl._connect()
    _insert(conn, "ob-3")
    conn.commit()
    conn.close()

    dl.mark_failed("ob-3", "Forbidden: the bot can't send messages to the bot")

    conn = dl._connect()
    row = conn.execute("SELECT state, last_error, delivered_receipt_id FROM delivery_obligations "
                       "WHERE obligation_id='ob-3'").fetchone()
    conn.close()
    assert row[0] == "failed" and "Forbidden" in row[1] and row[2] is None


def test_send_result_carries_the_evidence_fields():
    from gateway.platforms.base import SendResult

    for name in ("produced_len", "sent_len"):
        assert name in SendResult.__dataclass_fields__, f"SendResult lost `{name}`"
    r = SendResult(success=True, message_id="9")
    assert r.produced_len is None and r.sent_len is None, "fields must default to unknown"
    assert "sent_chunks" not in SendResult.__dataclass_fields__, (
        "an unread field is decoration — delete it or wire it"
    )


def test_finalize_passes_evidence_into_the_ledger(tmp_path):
    """The wire between transport and ledger: without this, the columns decay into decoration."""
    import re

    src = open("/usr/local/lib/hermes-agent/gateway/platforms/base.py").read()
    fn = re.search(r"async def _finalize_delivery_obligation\(.*?\n    async def ", src, re.S)
    assert fn, "_finalize_delivery_obligation not found"
    body = fn.group(0)
    assert "mark_delivered" in body
    for kw in ("produced_len=", "sent_len=", "delivered_receipt_id="):
        assert kw in body, f"finalize stopped passing {kw} — evidence chain broken"
